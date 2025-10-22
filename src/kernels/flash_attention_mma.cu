#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>
#include "../include/attention/PTX.h"
#include "../include/attention/utils.h"
namespace attention {

// BHND格式的索引计算: [B, H, N, D]
__device__ __forceinline__ int index_BHND(int b, int h, int n, int d, int B, int H, int N, int D) {
    return ((b * H + h) * N + n) * D + d;
}

template<
    const int PAD,
    const int NumThreads,
    const int KMmaAtomM,
    const int KMmaAtomN,
    const int KMmaAtomK,
    const int kMmaTileSeqLenQ,  // 4, more MMA(warp), M=16*4=64, Q@K^T=[Br(M),
                                // d(K)]@[d(K),  Bc(N)]
    const int kMmaTileSeqLenK,  // 1, more MMA(warp), N=8*1 =8,  Q@K^T=[Br(M),
                                // d(K)]@[d(K),  Bc(N)]
    const int kMmaTileSeqLenP,  // 4, more MMA(warp), M=16*4=64, P@V
                                // =[Br(M),Bc(K)]@[Bc(K), d(N) ]
    const int kMmaTileHeadDimV, // 1, more MMA(warp), N=8*1 =8,  P@V
                                // =[Br(M),Bc(K)]@[Bc(K), d(N) ]
    const int kWarpTileSeqLenQ, // 1, more values, M, Br=64*1=64, matmul M
    const int kWarpTileSeqLenK, // 8, more values, N, Bc=8*8 =64, matmul N
    const int kWarpTileSeqLenP, // 1, more values, M, Br=64*1=64, matmul M
    const int kWarpTileHeadDimV // 8, more values, N,
                                // d=8*(1|2|3|4|...)=8|...|32|64|96|128|...
>
__global__ void flash_attention_mma_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,   // restrict 表示通过该指针访问的内存区域不会被其他指针别名（alias）访问
    const int B, const int H, const int N, const int D,
    const float scale, 
    const int Br, const int Bc, const int Tr, const int Tc) {
    
    const int b = blockIdx.y;
    const int h = blockIdx.z;
    const int tx = threadIdx.x;
    const int warp_id = tx / 32;
    const int lane_id = tx % 32;

    extern __shared__ float s_data[];
    float *smem_q = s_data;
    float *smem_k = smem_q + Br * (D + PAD);
    float *smem_v = smem_k + Bc * (D + PAD);
    
    const int i = blockIdx.x;

    const float *q_start = Q + index_BHND(b, h, i * Br, 0, B, H, N, D);

    for (int k = 0; k < (D * Br) / NumThreads; k += 4) {
        FLOAT4(smem_q[NumThreads * k + tx * 4]) = 
            CONST_FLOAT4(q_start[NumThreads * k + tx * 4]);
    }

    uint32_t R_Q[kWarpTileSeqLenQ][4];
    uint32_t R_K[kWarpTileSeqLenK][2];
    uint32_t R_V[kWarpTileHeadDimV][2];

    float R_O[kWarpTileSeqLenP][kWarpTileHeadDimV][4];
    fill_3D_regs<float, kWarpTileSeqLenP, kWarpTileHeadDimV, 4>(R_O, 0.0f);

    float lane_row_max_new[kWarpTileSeqLenQ][2];
    float lane_row_sum_new[kWarpTileSeqLenQ][2];
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_sum_new, 0.0f);

    for (int j = 0; j < Tc; j++) {

        float R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][4];
        fill_3D_regs<float, kWarpTileSeqLenQ, kWarpTileSeqLenK, 4>(R_S, 0.0f);

        const float *k_start = K + index_BHND(b, h, j * Bc, 0, B, H, N, D);
        const float *v_start = V + index_BHND(b, h, j * Bc, 0, B, H, N, D);

        for (int k = 0; k < D * Bc / NumThreads; k += 4) {
            FLOAT4(smem_k[NumThreads * k + tx * 4]) = CONST_FLOAT4(k_start[NumThreads * k + tx * 4]);
            FLOAT4(smem_v[NumThreads * k + tx * 4]) = CONST_FLOAT4(v_start[NumThreads * k + tx * 4]);
        }

        __syncthreads();
        
        float lane_row_max_old[kWarpTileSeqLenQ][2];
        lane_row_max_old[0][0] = lane_row_max_new[0][0];
        lane_row_max_old[0][1] = lane_row_max_new[0][1];

        for (int d = 0; d < D; d += KMmaAtomK) {
            const int smem_regQ_addr_y = lane_id % 16;
            const int smem_regQ_addr_x = (lane_id / 16) * 4;

            uint32_t smem_regQ_ptr = 
                __cvta_generic_to_shared(&smem_q[(smem_regQ_addr_y + warp_id * KMmaAtomM) * (D + PAD) 
                                        + smem_regQ_addr_x + d]);

            LDMATRIX_X4(R_Q[0][0], R_Q[0][1], R_Q[0][2], R_Q[0][3], smem_regQ_ptr);

            for (int k = 0; k < kWarpTileSeqLenK; k++) {  
                const int smem_regK_addr_y = lane_id / 4 + k * 8;
                const int smem_regK_addr_x = lane_id % 4 + d;

                R_K[k][0] = __float_as_uint(smem_k[smem_regK_addr_y * (D + PAD) + smem_regK_addr_x]);
                R_K[k][1] = __float_as_uint(smem_k[smem_regK_addr_y * (D + PAD) + smem_regK_addr_x + 4]);

                SMMA1688(R_S[0][k][0], R_S[0][k][1], R_S[0][k][2], R_S[0][k][3], R_Q[0][0], R_Q[0][1], R_Q[0][2], R_Q[0][3], 
                    R_K[k][0], R_K[k][1], R_S[0][k][0], R_S[0][k][1], R_S[0][k][2], R_S[0][k][3]);
            }
        }
        
        // if (i == 0 && b == 0 && h == 0 && tx == 1)
        //     printf("R_S[0][0][0]: %f, R_S[0][0][1]: %f\n", 
        //         R_S[0][0][0], R_S[0][0][1]);

        __syncthreads();

        for (int k = 0; k < kWarpTileSeqLenK; k++) {
            float tmp_max_0 = max(R_S[0][k][0], R_S[0][k][1]) * scale;
            float tmp_max_1 = max(R_S[0][k][2], R_S[0][k][3]) * scale;
            lane_row_max_new[0][0] = max(lane_row_max_new[0][0], tmp_max_0);
            lane_row_max_new[0][1] = max(lane_row_max_new[0][1], tmp_max_1);
        }

        lane_row_max_new[0][0] = warp_reduce_max<float, 4>(lane_row_max_new[0][0]);
        lane_row_max_new[0][1] = warp_reduce_max<float, 4>(lane_row_max_new[0][1]);

        float acc[kWarpTileSeqLenQ][2];
        fill_2D_regs<float, kWarpTileSeqLenQ, 2>(acc, 0.0f);

        for (int k = 0; k < kWarpTileSeqLenK; k++) {
            R_S[0][k][0] = __expf(__fmaf_rn(R_S[0][k][0], scale, -lane_row_max_new[0][0]));
            R_S[0][k][1] = __expf(__fmaf_rn(R_S[0][k][1], scale, -lane_row_max_new[0][0]));
            R_S[0][k][2] = __expf(__fmaf_rn(R_S[0][k][2], scale, -lane_row_max_new[0][1]));
            R_S[0][k][3] = __expf(__fmaf_rn(R_S[0][k][3], scale, -lane_row_max_new[0][1]));
            acc[0][0] += (R_S[0][k][0] + R_S[0][k][1]);
            acc[0][1] += (R_S[0][k][2] + R_S[0][k][3]);
        }
        
        acc[0][0] = warp_reduce_sum<float, 4>(acc[0][0]);
        acc[0][1] = warp_reduce_sum<float, 4>(acc[0][1]);
         
        // if (i == 0 && b == 0 && h == 0 && tx == 0)
        //     printf("acc[0][0]: %f, acc[0][1]: %f\n", acc[0][0], acc[0][1]);

        lane_row_sum_new[0][0] = __expf(lane_row_max_old[0][0] - lane_row_max_new[0][0]) * lane_row_sum_new[0][0] + acc[0][0];
        lane_row_sum_new[0][1] = __expf(lane_row_max_old[0][1] - lane_row_max_new[0][1]) * lane_row_sum_new[0][1] + acc[0][1];
                 
        // if (i == 0 && b == 0 && h == 0 && tx == 0)
        //     printf("lane_row_sum_new[0][0]: %f, lane_row_sum_new[0][1]: %f\n", lane_row_sum_new[0][0], lane_row_sum_new[0][1]);
            
        __syncthreads();

        float R_D[kWarpTileSeqLenP][kWarpTileHeadDimV][4];
        fill_3D_regs<float, kWarpTileSeqLenP, kWarpTileHeadDimV, 4>(R_D, 0.0f);

        for (int k = 0; k < kWarpTileSeqLenK; k++) {
            // Reinterpret S tile elements to b32 for TF32 A operands
            uint32_t RS0 = __float_as_uint(R_S[0][k][0]); // 这里Permute了 S也就是P矩阵的K方向，所以V矩阵对应位置也需要映射
            uint32_t RS1 = __float_as_uint(R_S[0][k][2]);
            uint32_t RS2 = __float_as_uint(R_S[0][k][1]);
            uint32_t RS3 = __float_as_uint(R_S[0][k][3]);

            for (int v = 0; v < kWarpTileHeadDimV; v++) {

                const int smem_regV_addr_y = (lane_id % 4) * 2 + k * 8;
                const int smem_regV_addr_x = lane_id / 4 + v * 8;

                R_V[v][0] = __float_as_uint(smem_v[smem_regV_addr_y * (D + PAD) + smem_regV_addr_x]);
                R_V[v][1] = __float_as_uint((smem_v[(smem_regV_addr_y + 1) * (D + PAD) + smem_regV_addr_x]));

                SMMA1688(R_D[0][v][0], R_D[0][v][1], R_D[0][v][2], R_D[0][v][3], RS0, RS1, RS2, RS3, 
                    R_V[v][0], R_V[v][1], R_D[0][v][0], R_D[0][v][1], R_D[0][v][2], R_D[0][v][3]);
            }
        }

        __syncthreads();

        // if (i == 0 && b == 0 && h == 0 && tx == 1)
        //     printf("R_D[0][0][0]: %f, R_D[0][0][1]: %f, R_D[0][0][2]: %f, R_D[0][0][3]: %f\n", 
        //         R_D[0][0][0], R_D[0][0][1], R_D[0][0][2], R_D[0][0][3]);

        for (int v = 0; v < kWarpTileHeadDimV; v++) {
            R_O[0][v][0] = __expf(lane_row_max_old[0][0] - lane_row_max_new[0][0]) * R_O[0][v][0] + R_D[0][v][0];
            R_O[0][v][1] = __expf(lane_row_max_old[0][0] - lane_row_max_new[0][0]) * R_O[0][v][1] + R_D[0][v][1];
            R_O[0][v][2] = __expf(lane_row_max_old[0][1] - lane_row_max_new[0][1]) * R_O[0][v][2] + R_D[0][v][2];
            R_O[0][v][3] = __expf(lane_row_max_old[0][1] - lane_row_max_new[0][1]) * R_O[0][v][3] + R_D[0][v][3];
        }

        __syncthreads();
    }

    // if (i == 0 && b == 0 && h == 0 && tx == 0)
    //     printf("lane_row_sum_new[0][0]: %f, lane_row_sum_new[0][1]: %f\n", lane_row_sum_new[0][0], lane_row_sum_new[0][1]);

    float *O_start = O + index_BHND(b, h, i * Br, 0, B, H, N, D);
    const int reg_global_y = warp_id * KMmaAtomM + lane_id / 4;
    const int reg_global_x = (lane_id % 4) * 2;
    for (int v = 0; v < kWarpTileHeadDimV; v++) {
        O_start[reg_global_y * D + reg_global_x + v * KMmaAtomN] = R_O[0][v][0] / lane_row_sum_new[0][0];
        O_start[reg_global_y * D + reg_global_x + 1 + v * KMmaAtomN] = R_O[0][v][1] / lane_row_sum_new[0][0];
        O_start[(reg_global_y + 8) * D + reg_global_x + v * KMmaAtomN] = R_O[0][v][2] / lane_row_sum_new[0][1];
        O_start[(reg_global_y + 8) * D + reg_global_x + 1 + v * KMmaAtomN] = R_O[0][v][3] / lane_row_sum_new[0][1];
    }
     
}

void launch_flash_attention_mma_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B, int H, int N, int D,
    float scale,
    cudaStream_t stream) {
    
    const int Br = 64;
    const int Bc = 64; 
    const int Tr = ceil((float)N / Br);
    const int Tc = ceil((float)N / Bc);

    dim3 grid(Tr, B, H);
    dim3 block(128);
    
    const int PAD = 0;

    auto smem_size = (Br * (D + PAD) + 2 * Bc * (D + PAD)) * sizeof(float);

    // Enable opt-in larger dynamic shared memory if available
    int device = 0;
    cudaGetDevice(&device);
    int max_optin = 0;
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // Prefer shared memory in cache config
    cudaFuncSetCacheConfig(
        flash_attention_mma_forward_kernel<PAD, 128, 16, 8, 8, 4, 1, 4, 1, 1, 8, 1, 8>, cudaFuncCachePreferShared);
    // Set attribute to requested size if within opt-in limit
    if ((int)smem_size <= max_optin) {
        cudaFuncSetAttribute(
            flash_attention_mma_forward_kernel<PAD, 128, 16, 8, 8, 4, 1, 4, 1, 1, 8, 1, 8>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_size));
    } else {
        // If requested exceeds opt-in cap, it's safer to early return or adjust tiles
        // Here we early return to avoid invalid launch
        fprintf(stderr, "Requested shared memory %zu exceeds opt-in limit %d. Reduce Br/Bc or D.\n", (size_t)smem_size, max_optin);
        return;
    }
 
    flash_attention_mma_forward_kernel<PAD, 128, 16, 8, 8, 4, 1, 4, 1, 1, 8, 1, 8><<<grid, block, smem_size, stream>>>(
        Q, K, V, O, B, H, N, D, scale, Br, Bc, Tr, Tc);

    // cudaFree(d_l);
    // cudaFree(d_m);
}

} // namespace attention


