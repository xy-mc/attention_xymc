#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>
#include "../include/attention/PTX.h"

namespace attention {

// BHND格式的索引计算: [B, H, N, D]
__device__ __forceinline__ int index_BHND(int b, int h, int n, int d, int B, int H, int N, int D) {
    return ((b * H + h) * N + n) * D + d;
}


__global__ void flash_attention_v1_optimize_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N, int D,
    float scale, 
    const int Br, const int Bc, const int Tr, const int Tc,
    float* __restrict__ l,
    float* __restrict__ m) {
    
    const int b = blockIdx.y;
    const int h = blockIdx.z;
    const int tx = threadIdx.x;
    extern __shared__ float s_data[];

    float *smem_k = s_data;
    float *smem_v = smem_k + Bc * D;
    float *smem_q = smem_v + Bc * D;
    float *smem_s = smem_q + Br * D;
    
    const int i = blockIdx.x;

    for (int d = 0; d < D; d++) {
        smem_q[tx * D + d] = Q[index_BHND(b, h, i * Br + tx, d, B, H, N, D)];
    }

    for (int j = 0; j < Tc; j++) {

        for (int d = 0; d < D; d++) {
            smem_k[tx * D + d] = K[index_BHND(b, h, j * Bc + tx, d, B, H, N, D)];
            smem_v[tx * D + d] = V[index_BHND(b, h, j * Bc + tx, d, B, H, N, D)];
        }
            
        __syncthreads();

        float l_prev = l[index_BHND(b, h, i * Br + tx, 0, B, H, N, 1)];
        float m_prev = m[index_BHND(b, h, i * Br + tx, 0, B, H, N, 1)];
            
        // S_i,j <- smem_s
        float m_cur = -FLT_MAX;
        for (int k = 0; k < Bc; k++) {
            float res = 0.0f;
            for (int d = 0; d < D; d++) {
                res += smem_q[tx * D + d] * smem_k[k * D + d];
            }

            res *= scale;
            if (res > m_cur) m_cur = res;
                
            smem_s[tx * Bc + k] = res;
        }
            
        // P_i,j <- smem_s
        float l_cur = 0.0f;
        for (int k = 0; k < Bc; k++) {
            smem_s[tx * Bc + k] = __expf(smem_s[tx * Bc + k] - m_cur);
            l_cur += smem_s[tx * Bc + k];
        }

        float m_new = max(m_cur, m_prev);
        float l_new = __expf(m_prev - m_new) * l_prev + __expf(m_cur - m_new) * l_cur;

        for (int d = 0; d < D; d++) {
            float res = 0.0f;
            for (int k = 0; k < Bc; k++) {
                res += smem_s[tx * Bc + k] * smem_v[k * D + d];
            }

            O[index_BHND(b, h, Br * i + tx, d, B, H, N, D)] = 
                        (1.0f / l_new) * (l_prev * __expf(m_prev - m_new) * O[index_BHND(b, h, Br * i + tx, d, B, H, N, D)] 
                            + __expf(m_cur - m_new) * res);
        }
            

        l[index_BHND(b, h, i * Br + tx, 0, B, H, N, 1)] = l_new;
        m[index_BHND(b, h, i * Br + tx, 0, B, H, N, 1)] = m_new;
        
    }
}

static __global__ void init_lm(float* l, float* m) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    l[i] = 0.0f;
    m[i] = -FLT_MAX;
}

void launch_flash_attention_v1_optimize_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B, int H, int N, int D,
    float scale,
    cudaStream_t stream) {
    
    const int Br = D;
    const int Bc = D;

    const int Tr = ceil((float)N / Br);
    const int Tc = ceil((float)N / Bc);

    dim3 grid(Tr, B, H);
    dim3 block(Br);

    float *d_l, *d_m;
    cudaMalloc(&d_l, B * H * N * sizeof(float));
    cudaMalloc(&d_m, B * H * N * sizeof(float));
    init_lm<<<B * H, N, 0, stream>>>(d_l, d_m);

    auto smem_size = (Br * D + 2 * Bc * D + Br * Bc) * sizeof(float);

    // Enable opt-in larger dynamic shared memory if available
    int device = 0;
    cudaGetDevice(&device);
    int max_optin = 0;
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // Prefer shared memory in cache config
    cudaFuncSetCacheConfig(flash_attention_v1_optimize_forward_kernel, cudaFuncCachePreferShared);
    // Set attribute to requested size if within opt-in limit
    if ((int)smem_size <= max_optin) {
        cudaFuncSetAttribute(
            flash_attention_v1_optimize_forward_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_size));
    } else {
        // If requested exceeds opt-in cap, it's safer to early return or adjust tiles
        // Here we early return to avoid invalid launch
        fprintf(stderr, "Requested shared memory %zu exceeds opt-in limit %d. Reduce Br/Bc or D.\n", (size_t)smem_size, max_optin);
        return;
    }
 
    flash_attention_v1_optimize_forward_kernel<<<grid, block, smem_size, stream>>>(Q, K, V, O, B, H, N, D, scale, Br, Bc, Tr, Tc,
    d_l, d_m);

    cudaFree(d_l);
    cudaFree(d_m);
}

} // namespace attention


