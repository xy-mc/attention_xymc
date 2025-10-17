#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>
#include "../include/attention/PTX.h"

namespace attention {

// BHND格式的索引计算: [B, H, N, D]
__device__ __forceinline__ int index_BHND(int b, int h, int n, int d, int B, int H, int N, int D) {
    return ((b * H + h) * N + n) * D + d;
}

/*
一个block有8个warp，一起处理64 * 64 的矩阵，为了处理online softmax
每个warp计算出8*64的结果， 这样每个warp中的结果可以覆盖8行，
可以用warp_shuffle来获得m和l
*/
template<typename T, const int KWarpSize = 32>
__device__ __forceinline__ T warp_reduce_max(T val) { 
    #pragma unroll
    for (int mask = KWarpSize >> 1; mask > 0; mask >>= 1) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask, KWarpSize));
    }
    return val;
}

template<typename T, const int KWarpSize = 32>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int mask = KWarpSize >> 1; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask, KWarpSize);
    }
    return val;
}

template<
    const int MMaAtomM,
    const int MMaAtomN,
    const int MMaAtomK,
    const int global_smem_numtx,
    const int PAD,
    const int O_warp_size> // 16 或者 32 依据 D 是64还是128
__global__ void flash_attention_v2_optimize_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int B, const int H, const int N, const int D,
    const float scale, 
    const int Br, const int Bc, const int Tr, const int Tc) {
    
    const int b = blockIdx.y;
    const int h = blockIdx.z;
    const int tx = threadIdx.x;
    const int warp_id = tx / 32;
    const int lane_id = tx % 32;
    extern __shared__ float s_data[];
    float *smem_k = s_data;
    float *smem_v = smem_k + Bc * (D * PAD);
    float *smem_q = smem_v + Bc * (D * PAD);
    float *smem_s = smem_q + Br * (D * PAD);
    float *smem_o = smem_s + Br * Bc;
    
    const int i = blockIdx.x;

    const float *q_start = Q + index_BHND(b, h, i * Br, 0, B, H, N, D);

    for (int k = 0; k < global_smem_numtx; k++) {
        const int global_smem_load = tx * 4 + blockDim.x * global_smem_numtx * 4;
        FLOAT4(smem_q[global_smem_load]) = CONST_FLOAT4(q_start[global_smem_load]);
    }
        
    // float *O_cur = smem_o + tx * Br;
    // for (int d = 0; d < D; d++) {
    //     O_cur[d] = 0.0f;
    // }

    float O_cur[8][2] = {0.0f};

    __syncthreads();

    float l_cur[8] = {0.0f};
    float m_cur[8] = {-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX};

    for (int j = 0; j < Tc; j++) {

        const float *k_start = K + index_BHND(b, h, j * Bc, 0, B, H, N, D);
        const float *v_start = V + index_BHND(b, h, j * Bc, 0, B, H, N, D);

        for (int k = 0; k < global_smem_numtx; k++) {
            const int global_smem_load = tx * 4 + blockDim.x * global_smem_numtx * 4;
            FLOAT4(smem_k[global_smem_load]) = CONST_FLOAT4(k_start[global_smem_load]);
            FLOAT4(smem_v[global_smem_load]) = CONST_FLOAT4(v_start[global_smem_load]);
        }

        __syncthreads();
                
        float m_prev[8] = m_cur;

        float p[8][2] = {0.0f};
        float reg_q[4];
        float reg_k[4];

        for (int d = 0; d < D; d += 4) {
            FLOAT4(reg_q[0]) = FLOAT4(smem_q[(D + PAD) * (warp_id * 8 + lane_id % 8) + d]);
            for (int a = 0; a < 8; a++) {
                for (int b = 0; b < 2; b++) {
                    FLOAT4(reg_k[0]) = FLOAT4(smem_k[(D + PAD) * (b * 32 + lane_id) + d]);
                    for (int k = 0; k < 4; k++) {
                        p[a][b] += reg_q[k] * reg_k[k];
                    }
                }
            }
        }
        
        for (int a = 0; a < 8; a++) {
            for (int b = 0; b < 2; b++) {
                p[a][b] *= scale;
                if (p[a][b] > m_cur[a]) m_cur[a] = p[a][b];
            }
            
            m_cur[a] = warp_reduce_max<float, 32>(m_cur[a]);
        }
        
        float acc[8] = {0.0f};
        for (int a = 0; a < 8; a++) {
            for (int b = 0; b < 2; b++) {
                p[a][b] = __expf(p[a][b] - m_cur[a]);
                acc[a] += p[a][b];
            }

            acc[a] = warp_reduce_sum<float, 32>(acc[a]);
        }

        for (int a = 0; a < 8; a++) {
            l_cur[a] = __expf(m_prev[a] - m_cur[a]) * l_cur[a] + acc[a];
        }
        

        for (int d = 0; d < D; d += 8) {

            float res[8] = {0.0f};
            for (int a = 0; a < 8; a++) {

                for (int b = 0; b < 2; b++) {
                    res[a] += p[a][b] * smem_v[(D + PAD) * (b * 32 + lane_id) + a + d];
                }
                
                res[a] = warp_reduce_sum<float, 32>(res[a]);
            }

            
        }

        for (int d = 0; d < D; d++) {
            float res = 0.0f;
            for (int k = 0; k < Bc; k++) {
                res += smem_s[tx * Bc + k] * smem_v[k * D + d];
            }

            O_cur[d] = __expf(m_prev - m_cur) * O_cur[d] + res;
        }
    }
        

    for (int d = 0; d < D; d++) {
        O[index_BHND(b, h, Br * i + tx, d, B, H, N, D)] = O_cur[d] / l_cur;
    }
     
}

// static __global__ void init_lm(float* l, float* m) {
//     const int i = blockIdx.x * blockDim.x + threadIdx.x;

//     l[i] = 0.0f;
//     m[i] = -FLT_MAX;
// }

void launch_flash_attention_v2_optimize_forward(
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
    dim3 block(256);

    // float *d_l, *d_m;
    // cudaMalloc(&d_l, B * H * N * sizeof(float));
    // cudaMalloc(&d_m, B * H * N * sizeof(float));
    // init_lm<<<B * H, N, 0, stream>>>(d_l, d_m);

    auto smem_size = (2 * Br * (D * PAD) + 2 * Bc * (D * PAD) + Br * Bc) * sizeof(float);

    // Enable opt-in larger dynamic shared memory if available
    int device = 0;
    cudaGetDevice(&device);
    int max_optin = 0;
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // Prefer shared memory in cache config
    cudaFuncSetCacheConfig(flash_attention_v2_optimize_forward_kernel, cudaFuncCachePreferShared);
    // Set attribute to requested size if within opt-in limit
    if ((int)smem_size <= max_optin) {
        cudaFuncSetAttribute(
            flash_attention_v2_optimize_forward_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_size));
    } else {
        // If requested exceeds opt-in cap, it's safer to early return or adjust tiles
        // Here we early return to avoid invalid launch
        fprintf(stderr, "Requested shared memory %zu exceeds opt-in limit %d. Reduce Br/Bc or D.\n", (size_t)smem_size, max_optin);
        return;
    }
 
    flash_attention_v2_optimize_forward_kernel<<<grid, block, smem_size, stream>>>(Q, K, V, O, B, H, N, D, 
    scale, Br, Bc, Tr, Tc);

    // cudaFree(d_l);
    // cudaFree(d_m);
}

} // namespace attention


