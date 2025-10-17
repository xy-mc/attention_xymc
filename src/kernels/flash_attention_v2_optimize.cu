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
    const float *q_start = Q + index_BHND(b, h, i * Br, 0, B, H, N, D);

    for (int d = 0; d < D / 4; d++) {
        FLOAT4(smem_q[(D + PAD) * (d * Br / global_smem_numtx + tx / global_smem_numtx) + tx % global_smem_numtx * 4]) = 
        CONST_FLOAT4(q_start[D * (d * Br / global_smem_numtx + tx / global_smem_numtx) + tx % global_smem_numtx * 4]);
    }
    
    float *O_cur = smem_o + tx * (D + 1);
    for (int d = 0; d < D; d++) {
        O_cur[d] = 0.0f;
    }

    __syncthreads();

    float l_cur = 0.0f;
    float m_cur = -FLT_MAX;

    for (int j = 0; j < Tc; j++) {

        const float *k_start = K + index_BHND(b, h, j * Bc, 0, B, H, N, D);
        const float *v_start = V + index_BHND(b, h, j * Bc, 0, B, H, N, D);
        
        for (int d = 0; d < D / 4; d++) {
            FLOAT4(smem_k[D * (d * Bc / global_smem_numtx + tx / global_smem_numtx) + tx % global_smem_numtx * 4]) = 
            CONST_FLOAT4(k_start[D * (d * Bc / global_smem_numtx + tx / global_smem_numtx) + tx % global_smem_numtx * 4]);
            
            FLOAT4(smem_v[D * (d * Bc / global_smem_numtx + tx / global_smem_numtx) + tx % global_smem_numtx * 4]) = 
            CONST_FLOAT4(v_start[D * (d * Bc / global_smem_numtx + tx / global_smem_numtx) + tx % global_smem_numtx * 4]);
        }

        __syncthreads();
        
        float m_prev = m_cur;

        float q[global_smem_numtx * 4];
        for (int d = 0; d < D; d += 4) {
            FLOAT4(q[d]) = FLOAT4(smem_q[tx * (D + PAD) + d]);
        }
        
        float s[BR];
        for (int k = 0; k < Bc; k++) {
            float res = 0.0f;
            for (int d = 0; d < D; d++) {
                res += q[d] * smem_k[k * D + d];
            }
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
        
        for (int d = 0; d < D; d++) {
            float res = 0.0f;
            for (int k = 0; k < Bc; k++) {
                res += s[k] * smem_v[k * D + d];
            }

            O_cur[d] = __expf(m_prev - m_cur) * O_cur[d] + res;
        }
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

    auto smem_size = (Br * (D + 4) + Br * (D + 1) + 2 * Bc * D) * sizeof(float);

    // Enable opt-in larger dynamic shared memory if available
    int device = 0;
    cudaGetDevice(&device);
    int max_optin = 0;
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
   
    
    if (D == 64) {
        // 使用 <4, 16> 配置
        cudaFuncSetCacheConfig(flash_attention_v2_optimize_forward_kernel<4, 16, Br>, cudaFuncCachePreferShared);
        
        if ((int)smem_size <= max_optin) {
            cudaFuncSetAttribute(
                flash_attention_v2_optimize_forward_kernel<4, 16, Br>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_size));
        }
        
        flash_attention_v2_optimize_forward_kernel<4, 16, Br><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, B, H, N, D, scale, Br, Bc, Tr, Tc);
            
    } else if (D == 128) {
        // 使用 <4, 32> 配置
        cudaFuncSetCacheConfig(flash_attention_v2_optimize_forward_kernel<4, 32, Br>, cudaFuncCachePreferShared);
        
        if ((int)smem_size <= max_optin) {
            cudaFuncSetAttribute(
                flash_attention_v2_optimize_forward_kernel<4, 32, Br>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(smem_size));
        }
        
        flash_attention_v2_optimize_forward_kernel<4, 32, Br><<<grid, block, smem_size, stream>>>(
            Q, K, V, O, B, H, N, D, scale, Br, Bc, Tr, Tc);

        }
    }
} // namespace attention


