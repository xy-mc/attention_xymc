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

static __host__ __device__ __forceinline__
int swizzle_QK(const int y, const int x) {
    // x >>= 2;
    // return ((y & 7) ^ x) << 2;
    return ((y & 7) ^ (x >> 2)) << 2 | (x & 3);
    // return x;
}

static __host__ __device__ __forceinline__
int swizzle_V(const int y, const int x) {
    return (((y & 7) >> 1) ^ (x >> 3)) << 3 | (x & 7);
    // return x;
}

template<
    const int PAD,
    const int NumThreads,
    const int KMmaAtomM,  // 16
    const int KMmaAtomN,  // 8
    const int KMmaAtomK,  // 8
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
    const int kWarpTileHeadDimV, // 8, more values, N,
                                // d=8*(1|2|3|4|...)=8|...|32|64|96|128|...
    const int Kstage
>
__global__ void flash_attention_mma_Kstage_forward_kernel(
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
    const int warp_id = tx / WARP_SIZE;
    const int lane_id = tx % WARP_SIZE;

    extern __shared__ __align__(16) float s_data[];
    float *smem_q = s_data;
    float *smem_k = smem_q + Br * (D + PAD);
    float *smem_v = smem_k + Bc * (D + PAD) * Kstage;
    
    const int i = blockIdx.x;

    const float *q_start = Q + index_BHND(b, h, i * Br, 0, B, H, N, D);
    
    const int qkv_addr = tx * 4;
    const int global_smem_offset_y = (NumThreads * 4) / (D + PAD);
    const int global_smem_qkv_y = (qkv_addr) / (D + PAD);
    const int global_smem_qkv_x = (qkv_addr) % (D + PAD);
    
    #pragma unroll
    for (int k = 0; k < (D * Br) / (NumThreads * 4); k++) {

        uint32_t smem_q_ptr = __cvta_generic_to_shared(&smem_q[qkv_addr + NumThreads * k * 4]);

        const int global_smem_q_y = global_smem_qkv_y + k * global_smem_offset_y;

        CP_ASYNC_CG(smem_q_ptr, 
            &q_start[global_smem_q_y * (D + PAD) + 
                swizzle_QK(global_smem_q_y, global_smem_qkv_x)], 16);

    }

    CP_ASYNC_COMMIT_GROUP();
    __syncthreads();


    if constexpr (Kstage > 1) {

        #pragma unroll
        for (int stage = 0; stage < (Kstage - 1); stage++) {

            const float *k_start = K + index_BHND(b, h, stage * Bc, 0, B, H, N, D);

            uint32_t smem_k_base_ptr = __cvta_generic_to_shared(smem_k + stage * Bc * (D + PAD));

            #pragma unroll
            for (int k = 0; k < (D * Bc) / (NumThreads * 4); k++) {
                
                uint32_t smem_k_ptr = smem_k_base_ptr + (qkv_addr + NumThreads * k * 4) * sizeof(float);

                const int global_smem_k_y = global_smem_qkv_y + k * global_smem_offset_y;

                CP_ASYNC_CG(smem_k_ptr, 
                    &k_start[global_smem_k_y * (D + PAD) + swizzle_QK(global_smem_k_y, global_smem_qkv_x)], 16);

            }

            CP_ASYNC_COMMIT_GROUP();
        }
    }

    if constexpr (Kstage > 1) {
        CP_ASYNC_WAIT_GROUP(Kstage - 2);
        __syncthreads();
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

    #pragma unroll
    for (int j = 0; j < Tc; j++) {

        if constexpr (Kstage > 1) {

            const float *v_start = V + index_BHND(b, h, j * Bc, 0, B, H, N, D);
            #pragma unroll
            for (int k = 0; k < (D * Bc) / (NumThreads * 4); k++) {
                uint32_t smem_v_ptr = __cvta_generic_to_shared(&smem_v[(qkv_addr + NumThreads * k * 4)]);

                const int global_smem_v_y = global_smem_qkv_y + k * global_smem_offset_y;

                CP_ASYNC_CG(smem_v_ptr, 
                    &v_start[global_smem_v_y * (D + PAD) + swizzle_V(global_smem_v_y, global_smem_qkv_x)], 16);
            }
            CP_ASYNC_COMMIT_GROUP();

            if (j + Kstage - 1 < Tc) {
                const float *k_start = K + index_BHND(b, h, (j + Kstage - 1) * Bc, 0, B, H, N, D);
                uint32_t smem_k_base_ptr = __cvta_generic_to_shared(smem_k + ((j + Kstage - 1) % Kstage) * Bc * (D + PAD));
                #pragma unroll
                for (int k = 0; k < (D * Bc) / (NumThreads * 4); k++) {
                    uint32_t smem_k_ptr = smem_k_base_ptr + (qkv_addr + NumThreads * k * 4) * sizeof(float);

                    const int global_smem_k_y = global_smem_qkv_y + k * global_smem_offset_y;

                    CP_ASYNC_CG(smem_k_ptr, 
                        &k_start[global_smem_k_y * (D + PAD) + swizzle_QK(global_smem_k_y, global_smem_qkv_x)], 16);
                }
                CP_ASYNC_COMMIT_GROUP();
            }
        }
        else {

            const float *k_start = K + index_BHND(b, h, j * Bc, 0, B, H, N, D);
            #pragma unroll
            for (int k = 0; k < (D * Bc) / (NumThreads * 4); k++) {
                uint32_t smem_k_ptr = __cvta_generic_to_shared(&smem_k[(qkv_addr + NumThreads * k * 4)]);

                const int global_smem_k_y = global_smem_qkv_y + k * global_smem_offset_y;

                CP_ASYNC_CG(smem_k_ptr, 
                    &k_start[global_smem_k_y * (D + PAD) + swizzle_QK(global_smem_k_y, global_smem_qkv_x)], 16);
            }
            CP_ASYNC_COMMIT_GROUP();

            const float *v_start = V + index_BHND(b, h, j * Bc, 0, B, H, N, D);
            #pragma unroll
            for (int k = 0; k < (D * Bc) / (NumThreads * 4); k++) {
                uint32_t smem_v_ptr = __cvta_generic_to_shared(&smem_v[(qkv_addr + NumThreads * k * 4)]);

                const int global_smem_v_y = global_smem_qkv_y + k * global_smem_offset_y;

                CP_ASYNC_CG(smem_v_ptr, &v_start[global_smem_v_y * (D + PAD) + swizzle_V(global_smem_v_y, global_smem_qkv_x)], 16);
            }
            CP_ASYNC_COMMIT_GROUP();

            CP_ASYNC_WAIT_GROUP(1);
            __syncthreads();
        }

        float R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][4];
        fill_3D_regs<float, kWarpTileSeqLenQ, kWarpTileSeqLenK, 4>(R_S, 0.0f);
        
        float lane_row_max_old[kWarpTileSeqLenQ][2];
        
        #pragma unroll
        for (int q = 0; q < kWarpTileSeqLenQ; q++) {
            lane_row_max_old[q][0] = lane_row_max_new[q][0];
            lane_row_max_old[q][1] = lane_row_max_new[q][1];
        }

        float* smem_k_current = smem_k + (j % Kstage) * Bc * (D + PAD);
        
        // if (i == 0 && j == 0 && b == 0 && h == 0 && tx == 0) {
        //     printf("mma_Kstage: R_Q values:\n");
        //     for (int mma = 0; mma < kWarpTileSeqLenQ; mma++) {
        //         for (int q = 0; q < 4; q++) {
        //             printf("  R_Q(%d, %d) = %f\n", mma, q, (float)R_Q[mma][q]);
        //         }   
        //     }
        // }

        for (int d = 0; d < D; d += KMmaAtomK) {
            #pragma unroll
            for (int q = 0; q < kWarpTileSeqLenQ; q++) {
                const int smem_regQ_addr_y = lane_id % 16 + warp_id * KMmaAtomM + q * KMmaAtomM * kMmaTileSeqLenQ;
                const int smem_regQ_addr_x = (lane_id / 16) * 4 + d;

                uint32_t smem_regQ_ptr = 
                    __cvta_generic_to_shared(&smem_q[smem_regQ_addr_y * (D + PAD) 
                                            + swizzle_QK(smem_regQ_addr_y, smem_regQ_addr_x)]);
                
                // if (i == 0 && b == 0 && h == 0 && j == 0 && d == 0 && tx == 2)
                //         printf("smem_regQ_ptr_y: %d, smem_regQ_ptr_x: %d\n", 
                //                 smem_regQ_addr_y, swizzle_Q(smem_regQ_addr_y, smem_regQ_addr_x));
                LDMATRIX_X4(R_Q[q][0], R_Q[q][1], R_Q[q][2], R_Q[q][3], smem_regQ_ptr);
                
                for (int k = 0; k < kWarpTileSeqLenK; k++) {  
                    const int smem_regK_addr_y = lane_id / 4 + k * 8;
                    const int smem_regK_addr_x = lane_id % 4 + d;
    
                    R_K[k][0] = __float_as_uint(smem_k_current[smem_regK_addr_y * (D + PAD) + 
                        swizzle_QK(smem_regK_addr_y, smem_regK_addr_x)]);
    
                    R_K[k][1] = __float_as_uint(smem_k_current[smem_regK_addr_y * (D + PAD) + 
                        swizzle_QK(smem_regK_addr_y, smem_regK_addr_x + 4)]);
                    
                    SMMA1688(R_S[q][k][0], R_S[q][k][1], R_S[q][k][2], R_S[q][k][3], R_Q[q][0], R_Q[q][1], R_Q[q][2], R_Q[q][3], 
                        R_K[k][0], R_K[k][1], R_S[q][k][0], R_S[q][k][1], R_S[q][k][2], R_S[q][k][3]);
                }
            }
        }
        
        // if (i == 0 && j == 0 && b == 0 && h == 0 && tx == 0) {
        //     for (int q = 0; q < kWarpTileSeqLenQ; q++) {
        //         for (int k = 0; k < kWarpTileSeqLenK; k++) {
        //             printf("mma_Kstage: R_S[%d][%d][0]: %f, R_S[%d][%d][1]: %f, R_S[%d][%d][2]: %f, R_S[%d][%d][3]: %f\n", 
        //                 q, k, (float)R_S[q][k][0], q, k, (float)R_S[q][k][1], q, k, (float)R_S[q][k][2], q, k, (float)R_S[q][k][3]);
        //         }
        //     }
        // }
        // __syncthreads();
        
        #pragma unroll
        for (int q = 0; q < kWarpTileSeqLenQ; q++) {
            #pragma unroll
            for (int k = 0; k < kWarpTileSeqLenK; k++) {
                float tmp_max_0 = max(R_S[q][k][0], R_S[q][k][1]) * scale;
                float tmp_max_1 = max(R_S[q][k][2], R_S[q][k][3]) * scale;
                // if (i == 0 && j == 0 && b == 0 && h == 0 && tx == 0) {
                //     printf("mma_Kstage: tmp_max_0: %f, tmp_max_1: %f\n", tmp_max_0, tmp_max_1);
                // }
                lane_row_max_new[q][0] = max(lane_row_max_new[q][0], tmp_max_0);
                lane_row_max_new[q][1] = max(lane_row_max_new[q][1], tmp_max_1);
            }
        }

        // if (i == 0 && j == 0 && b == 0 && h == 0 && tx == 0) {
        //     printf("mma_Kstage: lane_row_max_new: %f, %f\n", lane_row_max_new[0][0], lane_row_max_new[0][1]);
        // }

        #pragma unroll
        for (int q = 0; q < kWarpTileSeqLenQ; q++) {
            lane_row_max_new[q][0] = warp_reduce_max<float, 4>(lane_row_max_new[q][0]);
            lane_row_max_new[q][1] = warp_reduce_max<float, 4>(lane_row_max_new[q][1]);
            // if (i == 0 && j == 0 && b == 0 && h == 0 && tx == 0) {
            //     printf("mma_Kstage: lane_row_max_new: %f, %f\n", lane_row_max_new[q][0], lane_row_max_new[q][1]);
            // }
        }

        float acc[kWarpTileSeqLenQ][2];
        fill_2D_regs<float, kWarpTileSeqLenQ, 2>(acc, 0.0f);

        #pragma unroll
        for (int q = 0; q < kWarpTileSeqLenQ; q++) {
            #pragma unroll
            for (int k = 0; k < kWarpTileSeqLenK; k++) {
                R_S[q][k][0] = __expf(__fmaf_rn(R_S[q][k][0], scale, -lane_row_max_new[q][0]));
                R_S[q][k][1] = __expf(__fmaf_rn(R_S[q][k][1], scale, -lane_row_max_new[q][0]));
                R_S[q][k][2] = __expf(__fmaf_rn(R_S[q][k][2], scale, -lane_row_max_new[q][1]));
                R_S[q][k][3] = __expf(__fmaf_rn(R_S[q][k][3], scale, -lane_row_max_new[q][1]));
                acc[q][0] += (R_S[q][k][0] + R_S[q][k][1]);
                acc[q][1] += (R_S[q][k][2] + R_S[q][k][3]);
            }
        }
        
        #pragma unroll
        for (int q = 0; q < kWarpTileSeqLenQ; q++) {
            acc[q][0] = warp_reduce_sum<float, 4>(acc[q][0]);
            acc[q][1] = warp_reduce_sum<float, 4>(acc[q][1]);

            // if (i == 0 && j == 0 && b == 0 && h == 0 && tx == 0) {
            //     printf("mma_Kstage: acc: %f, %f\n", acc[q][0], acc[q][1]);
            // }
            lane_row_sum_new[q][0] = __fmaf_rn(__expf(lane_row_max_old[q][0] - lane_row_max_new[q][0]),
                                    lane_row_sum_new[q][0], acc[q][0]);
            lane_row_sum_new[q][1] = __fmaf_rn(__expf(lane_row_max_old[q][1] - lane_row_max_new[q][1]),
                                    lane_row_sum_new[q][1], acc[q][1]);

            // if (i == 0 && j == 1 && b == 0 && h == 0 && tx == 0) {
            //     printf("mma_Kstage: lane_row_sum_new: %f, %f\n", lane_row_sum_new[q][0], lane_row_sum_new[q][1]);
            // }
        }
            
        // __syncthreads();

        if constexpr (Kstage > 1) {
            if (j + Kstage - 1 < Tc) {
                CP_ASYNC_WAIT_GROUP(1);
            }
            else {
                CP_ASYNC_WAIT_GROUP(0);
            }
        }
        else {
            CP_ASYNC_WAIT_GROUP(0);
        }
        __syncthreads();

        float R_D[kWarpTileSeqLenP][kWarpTileHeadDimV][4];
        fill_3D_regs<float, kWarpTileSeqLenP, kWarpTileHeadDimV, 4>(R_D, 0.0f);

        #pragma unroll
        for (int p = 0; p < kWarpTileSeqLenP; p++) {
            #pragma unroll
            for (int k = 0; k < kWarpTileSeqLenK; k++) {
                // Reinterpret S tile elements to b32 for TF32 A operands
                uint32_t RS0 = __float_as_uint(R_S[p][k][0]); // 这里Permute了 S也就是P矩阵的K方向，所以V矩阵对应位置也需要映射
                uint32_t RS1 = __float_as_uint(R_S[p][k][2]);
                uint32_t RS2 = __float_as_uint(R_S[p][k][1]);
                uint32_t RS3 = __float_as_uint(R_S[p][k][3]);

                #pragma unroll
                for (int v = 0; v < kWarpTileHeadDimV; v++) {

                    const int smem_regV_addr_y = (lane_id % 4) * 2 + k * 8;
                    const int smem_regV_addr_x = lane_id / 4 + v * 8;

                    R_V[v][0] = __float_as_uint(smem_v[smem_regV_addr_y * (D + PAD) + 
                                swizzle_V(smem_regV_addr_y, smem_regV_addr_x)]);
                    R_V[v][1] = __float_as_uint(smem_v[(smem_regV_addr_y + 1) * (D + PAD) + 
                                swizzle_V(smem_regV_addr_y + 1, smem_regV_addr_x)]);
                    
                    // if (i == 0 && j == 0 && b == 0 && h == 0 && tx == 0) {
                    //     printf("mma_Kstage: R_V: %f, %f\n", __uint_as_float(R_V[v][0]), __uint_as_float(R_V[v][1]));
                    // }

                    SMMA1688(R_D[p][v][0], R_D[p][v][1], R_D[p][v][2], R_D[p][v][3], RS0, RS1, RS2, RS3, 
                        R_V[v][0], R_V[v][1], R_D[p][v][0], R_D[p][v][1], R_D[p][v][2], R_D[p][v][3]);
                }
            }
        }

        // __syncthreads();

        // if (i == 0 && b == 0 && h == 0 && tx == 1)
        //     printf("R_D[0][0][0]: %f, R_D[0][0][1]: %f, R_D[0][0][2]: %f, R_D[0][0][3]: %f\n", 
        //         R_D[0][0][0], R_D[0][0][1], R_D[0][0][2], R_D[0][0][3]);

        #pragma unroll
        for (int p = 0; p < kWarpTileSeqLenP; p++) {
            #pragma unroll
            for (int v = 0; v < kWarpTileHeadDimV; v++) {
                R_O[p][v][0] = __fmaf_rn(__expf(lane_row_max_old[p][0] - lane_row_max_new[p][0]), R_O[p][v][0], R_D[p][v][0]);
                R_O[p][v][1] = __fmaf_rn(__expf(lane_row_max_old[p][0] - lane_row_max_new[p][0]), R_O[p][v][1], R_D[p][v][1]);
                R_O[p][v][2] = __fmaf_rn(__expf(lane_row_max_old[p][1] - lane_row_max_new[p][1]), R_O[p][v][2], R_D[p][v][2]);
                R_O[p][v][3] = __fmaf_rn(__expf(lane_row_max_old[p][1] - lane_row_max_new[p][1]), R_O[p][v][3], R_D[p][v][3]);

                // if (i == 0 && j == 1 && b == 0 && h == 0 && tx == 0) {
                //     printf("mma_Kstage: R_O: %f, %f, %f, %f\n", R_O[p][v][0], R_O[p][v][1], R_O[p][v][2], R_O[p][v][3]);
                // }
            }
        }

        if constexpr (Kstage > 1) {
            if ((j + Kstage - 1) < Tc) {
                CP_ASYNC_WAIT_GROUP(0);
            }
        }
        // __syncthreads();
    }

    // if (i == 0 && b == 0 && h == 0 && tx == 0)
    //     printf("lane_row_sum_new[0][0]: %f, lane_row_sum_new[0][1]: %f\n", lane_row_sum_new[0][0], lane_row_sum_new[0][1]);
    float *O_start = O + index_BHND(b, h, i * Br, 0, B, H, N, D);

    #pragma unroll
    for (int p = 0; p < kWarpTileSeqLenP; p++) {

        const int reg_global_y = warp_id * KMmaAtomM + lane_id / 4 + p * kMmaTileSeqLenP * KMmaAtomM;
        const int reg_global_x = (lane_id % 4) * 2;

        #pragma unroll
        for (int v = 0; v < kWarpTileHeadDimV; v++) {
            
            R_O[p][v][0] = __fdividef(R_O[p][v][0], lane_row_sum_new[p][0]);
            R_O[p][v][1] = __fdividef(R_O[p][v][1], lane_row_sum_new[p][0]);
            R_O[p][v][2] = __fdividef(R_O[p][v][2], lane_row_sum_new[p][1]);
            R_O[p][v][3] = __fdividef(R_O[p][v][3], lane_row_sum_new[p][1]);
            
            // if (i == 0 && b == 0 && h == 0 && tx == 32) {
            //     printf("mma_Kstage: R_O: %f, %f, %f, %f\n", R_O[p][v][0], R_O[p][v][1], R_O[p][v][2], R_O[p][v][3]);
            // }

            LDST64BITS(O_start[reg_global_y * D + reg_global_x + v * KMmaAtomN])
                = LDST64BITS(R_O[p][v][0]);

            LDST64BITS(O_start[(reg_global_y + 8) * D + reg_global_x + v * KMmaAtomN])
                = LDST64BITS(R_O[p][v][2]);
        }
    }
}

template<int D>
void launch_flash_attention_mma_Kstage_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B, int H, int N,
    float scale,
    cudaStream_t stream) {
    
    constexpr int KMmaAtomM = 16;
    constexpr int KMmaAtomN = 8;
    constexpr int KMmaAtomK = 8;
    constexpr int kMmaTileSeqLenQ = 4;
    constexpr int kMmaTileSeqLenK = 1;
    constexpr int kMmaTileSeqLenP = 4;
    constexpr int kMmaTileHeadDimV = 1;
    constexpr int kWarpTileSeqLenQ = 2;
    constexpr int kWarpTileSeqLenK = 4;
    constexpr int kWarpTileSeqLenP = 2;
    constexpr int kWarpTileHeadDimV = D / (KMmaAtomN * kMmaTileHeadDimV);

    constexpr int NumThreads = WARP_SIZE * kMmaTileSeqLenQ;

    const int Br = KMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
    const int Bc = KMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; 
    const int Tr = ceil((float)N / Br);
    const int Tc = ceil((float)N / Bc);

    dim3 grid(Tr, B, H);
    dim3 block(NumThreads);
    
    constexpr int PAD = 0;
    constexpr int Kstage = 1;
    auto smem_size = (Br * (D + PAD) + Bc * (D + PAD) + Bc * (D + PAD) * Kstage) * sizeof(float);

    // Enable opt-in larger dynamic shared memory if available
    int device = 0;
    cudaGetDevice(&device);
    int max_optin = 0;
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // Prefer shared memory in cache config
    cudaFuncSetCacheConfig(
        flash_attention_mma_Kstage_forward_kernel<PAD, NumThreads, KMmaAtomM, KMmaAtomN, KMmaAtomK, kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ, kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV, Kstage>, cudaFuncCachePreferShared);
    // Set attribute to requested size if within opt-in limit
    if ((int)smem_size <= max_optin) {
        cudaFuncSetAttribute(
            flash_attention_mma_Kstage_forward_kernel<PAD, NumThreads, KMmaAtomM, KMmaAtomN, KMmaAtomK, kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ, kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV, Kstage>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_size));
    } else {
        // If requested exceeds opt-in cap, it's safer to early return or adjust tiles
        // Here we early return to avoid invalid launch
        fprintf(stderr, "Requested shared memory %zu exceeds opt-in limit %d. Reduce Br/Bc or D.\n", (size_t)smem_size, max_optin);
        return;
    }
 
    flash_attention_mma_Kstage_forward_kernel<PAD, NumThreads, KMmaAtomM, KMmaAtomN, KMmaAtomK, kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ, kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV, Kstage><<<grid, block, smem_size, stream>>>(
        Q, K, V, O, B, H, N, D, scale, Br, Bc, Tr, Tc);

    // cudaFree(d_l);
    // cudaFree(d_m);
}

} // namespace attention

template void attention::launch_flash_attention_mma_Kstage_forward<64>(
    const float*, const float*, const float*, float*,
    int, int, int, float, cudaStream_t);

template void attention::launch_flash_attention_mma_Kstage_forward<128>(
    const float*, const float*, const float*, float*,
    int, int, int, float, cudaStream_t);
