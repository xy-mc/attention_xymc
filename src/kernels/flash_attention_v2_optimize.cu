#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>
#include "../include/attention/PTX.h"
namespace attention {

// BHND格式的索引计算: [B, H, N, D]
__device__ __forceinline__ int index_BHND(int b, int h, int n, int d, int B, int H, int N, int D) {
    return ((b * H + h) * N + n) * D + d;
}

template<const int PAD,
        const int global_smem_numtx,
        const int BR>
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
    const int i = blockIdx.x;
    const int tx = threadIdx.x;
    extern __shared__ float s_data[];

    float *smem_q = s_data;
    float *smem_v = smem_q + Br * (D + PAD);
    float *smem_k = smem_v + Bc * D;
    float *smem_o = smem_k + Bc * D;

    // for (int i = 0; i < Tr; i++) {
    // const float *q_start = Q + index_BHND(b, h, i * Br, 0, B, H, N, D);

    // for (int d = 0; d < D / 4; d++) {
    //     FLOAT4(smem_q[(D + PAD) * (d * Br / global_smem_numtx + tx / global_smem_numtx) + tx % global_smem_numtx * 4]) = 
    //     CONST_FLOAT4(q_start[D * (d * Br / global_smem_numtx + tx / global_smem_numtx) + tx % global_smem_numtx * 4]);
    // }
    
    for (int d = 0; d < D; d++) {
        smem_q[tx * D + d] = Q[index_BHND(b, h, i * Br + tx, d, B, H, N, D)];
    }

    float *O_cur = smem_o + tx * (D + 1);
    for (int d = 0; d < D; d++) {
        O_cur[d] = 0.0f;
    }

    __syncthreads();

    // if (tx == 0 && i == 0 && b == 0 && h == 0) {
    //     for (int k = 0; k < Br; k++) {
    //         for (int d = 0; d < D; d++) {
    //             printf("smem_q[%d][%d]: %f  ", k, d, smem_q[k * (D + PAD) + d]);
    //         }
    //         printf("\n");
    //     }
    // }

    float l_cur = 0.0f;
    float m_cur = -FLT_MAX;

    for (int j = 0; j < Tc; j++) {

        // const float *k_start = K + index_BHND(b, h, j * Bc, 0, B, H, N, D);
        // const float *v_start = V + index_BHND(b, h, j * Bc, 0, B, H, N, D);
        
        // for (int d = 0; d < D / 4; d++) {
        //     FLOAT4(smem_k[D * (d * Bc / global_smem_numtx + tx / global_smem_numtx) + tx % global_smem_numtx * 4]) = 
        //     CONST_FLOAT4(k_start[D * (d * Bc / global_smem_numtx + tx / global_smem_numtx) + tx % global_smem_numtx * 4]);
            
        //     FLOAT4(smem_v[D * (d * Bc / global_smem_numtx + tx / global_smem_numtx) + tx % global_smem_numtx * 4]) = 
        //     CONST_FLOAT4(v_start[D * (d * Bc / global_smem_numtx + tx / global_smem_numtx) + tx % global_smem_numtx * 4]);
        // }

        for (int d = 0; d < D; d++) {
            smem_k[tx * D + d] = K[index_BHND(b, h, j * Bc + tx, d, B, H, N, D)];
            smem_v[tx * D + d] = V[index_BHND(b, h, j * Bc + tx, d, B, H, N, D)];
        }

        __syncthreads();
        
        float m_prev = m_cur;

        // __align__(16) float q[global_smem_numtx * 4];
        // // Load q into registers with scalar loads to avoid misaligned vector stores
        // for (int d = 0; d < D; ++d) {
        //     q[d] = smem_q[tx * D + d];
        // }

        // Debug: compare q[] and smem_q row and one dot product for the first thread/tile
        // if (b == 0 && h == 0 && i == 0 && j == 0 && tx == 0) {
        //     float max_abs_diff = 0.0f;
        //     for (int d = 0; d < D; ++d) {
        //         float diff = fabsf(q[d] - smem_q[tx * D + d]);
        //         if (diff > max_abs_diff) max_abs_diff = diff;
        //     }
        //     if (max_abs_diff > 1e-6f) {
        //         printf("[DEBUG] max |q - smem_q| = %e\n", max_abs_diff);
        //     }
        //     // Compare one dot product with k=0
        //     float res_q = 0.0f, res_s = 0.0f;
        //     for (int d = 0; d < D; ++d) {
        //         res_q += q[d] * smem_k[0 * D + d];
        //         res_s += smem_q[tx * D + d] * smem_k[0 * D + d];
        //     }
        //     float dp_diff = fabsf(res_q - res_s);
        //     if (dp_diff > 1e-6f) {
        //         printf("[DEBUG] dot diff (q vs smem_q) = %e, res_q=%e, res_s=%e\n", dp_diff, res_q, res_s);
        //     }
        // }
        
        float s[BR];
        for (int k = 0; k < Bc; k++) {
            float res = 0.0f;
            for (int d = 0; d < D; d++) {
                res += smem_q[tx * D + d] * smem_k[k * D + d];
            }

            // if (i == 0 && b == 0 && h == 0 && tx == 0 && (k == 2 || k == 3))
            //     printf("s[%d] %f\n", k, res);
            
            res *= scale;
            if (res > m_cur) m_cur = res;

            s[k] = res;
        }
        
        float acc = 0.0f;
        for (int k = 0; k < Bc; k++) {
            s[k] = __expf(s[k] - m_cur);
            acc += s[k];
        }

        l_cur = __expf(m_prev - m_cur) * l_cur + acc;
        
        // if (i == 0 && b == 0 && h == 0 && tx == 8)
        //     printf("l_cur: %f\n", l_cur);

        for (int d = 0; d < D; d++) {
            float res = 0.0f;
            for (int k = 0; k < Bc; k++) {
                res += s[k] * smem_v[k * D + d];
            }

            // if (i == 0 && b == 0 && h == 0 && tx == 0)
            //     printf("res[%d]: %f ", d, res);
            O_cur[d] = __expf(m_prev - m_cur) * O_cur[d] + res;
        }

        // if (i == 0 && b == 0 && h == 0 && tx == 0)
        //     printf("\n");
    }

    for (int d = 0; d < D; d++) {
        O[index_BHND(b, h, Br * i + tx, d, B, H, N, D)] = O_cur[d] / l_cur;
    }
    // }
    
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
    dim3 block(Br);

    // float *d_l, *d_m;
    // cudaMalloc(&d_l, B * H * N * sizeof(float));
    // cudaMalloc(&d_m, B * H * N * sizeof(float));
    // init_lm<<<B * H, N, 0, stream>>>(d_l, d_m);
    const int PAD = 0;

    auto smem_size = (Br * (D + PAD) + Br * (D + 1) + 2 * Bc * D) * sizeof(float);

    // Enable opt-in larger dynamic shared memory if available
    int device = 0;
    cudaGetDevice(&device);
    int max_optin = 0;
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

    if (D == 64) {
        // 使用 <4, 16> 配置
        cudaFuncSetCacheConfig(flash_attention_v2_optimize_forward_kernel<PAD, 16, Br>, cudaFuncCachePreferShared);
        
        if ((int)smem_size <= max_optin) {
            cudaFuncSetAttribute(
                flash_attention_v2_optimize_forward_kernel<PAD, 16, Br>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_size));
        }
        
        flash_attention_v2_optimize_forward_kernel<PAD, 16, Br><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, B, H, N, D, scale, Br, Bc, Tr, Tc);
            
    } else if (D == 128) {
        // 使用 <4, 32> 配置
        cudaFuncSetCacheConfig(flash_attention_v2_optimize_forward_kernel<PAD, 32, Br>, cudaFuncCachePreferShared);
        
        if ((int)smem_size <= max_optin) {
            cudaFuncSetAttribute(
                flash_attention_v2_optimize_forward_kernel<PAD, 32, Br>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_size));
        }
        
        flash_attention_v2_optimize_forward_kernel<PAD, 32, Br><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, B, H, N, D, scale, Br, Bc, Tr, Tc);

        }
    }
} // namespace attention


