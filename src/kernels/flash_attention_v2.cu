#include <cuda_runtime.h>
#include <float.h>
#include <cstdio>

namespace attention {

// BHND格式的索引计算: [B, H, N, D]
__device__ __forceinline__ int index_BHND(int b, int h, int n, int d, int B, int H, int N, int D) {
    return ((b * H + h) * N + n) * D + d;
}


__global__ void flash_attention_v2_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int B, const int H, const int N, const int D,
    const float scale, 
    const int Br, const int Bc, const int Tr, const int Tc) {
    
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int tx = threadIdx.x;
    extern __shared__ float s_data[];

    float *smem_k = s_data;
    float *smem_v = smem_k + Bc * D;
    float *smem_q = smem_v + Bc * D;
    float *smem_s = smem_q + Br * D;
    float *smem_o = smem_s + Br * Bc;

    for (int i = 0; i < Tr; i++) {

        for (int d = 0; d < D; d++) {
            smem_q[tx * D + d] = Q[index_BHND(b, h, i * Br + tx, d, B, H, N, D)];
        }
        
        float *O_cur = smem_o + tx * Br;
        for (int d = 0; d < D; d++) {
            O_cur[d] = 0.0f;
        }

        __syncthreads();

        // if (tx == 0 && i == 0 && b == 0 && h == 0) {
        //     for (int k = 0; k < Br; k++) {
        //         for (int d = 0; d < D; d++) {
        //             printf("smem_q[%d][%d]: %f  ", k, d, smem_q[k * D + d]);
        //         }
        //         printf("\n");
        //     }
        // }
        
        float l_cur = 0.0f;
        float m_cur = -FLT_MAX;

        for (int j = 0; j < Tc; j++) {

            for (int d = 0; d < D; d++) {
                smem_k[tx * D + d] = K[index_BHND(b, h, j * Bc + tx, d, B, H, N, D)];
                smem_v[tx * D + d] = V[index_BHND(b, h, j * Bc + tx, d, B, H, N, D)];
            }

            __syncthreads();
            
            float m_prev = m_cur;

            for (int k = 0; k < Bc; k++) {
                float res = 0.0f;
                for (int d = 0; d < D; d++) {
                    res += smem_q[tx * D + d] * smem_k[k * D + d];
                }
                res *= scale;
                if (res > m_cur) m_cur = res;
                smem_s[tx * Bc + k] = res;
            }
            
            float acc = 0.0f;
            for (int k = 0; k < Bc; k++) {
                smem_s[tx * Bc + k] = __expf(smem_s[tx * Bc + k] - m_cur);
                acc += smem_s[tx * Bc + k];
            }

            l_cur = __expf(m_prev - m_cur) * l_cur + acc;


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
    
}

// static __global__ void init_lm(float* l, float* m) {
//     const int i = blockIdx.x * blockDim.x + threadIdx.x;

//     l[i] = 0.0f;
//     m[i] = -FLT_MAX;
// }

void launch_flash_attention_v2_forward(
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
    dim3 grid(B, H);
    dim3 block(Br);

    // float *d_l, *d_m;
    // cudaMalloc(&d_l, B * H * N * sizeof(float));
    // cudaMalloc(&d_m, B * H * N * sizeof(float));
    // init_lm<<<B * H, N, 0, stream>>>(d_l, d_m);

    auto smem_size = (2 * Br * D + 2 * Bc * D + Br * Bc) * sizeof(float);

    // Enable opt-in larger dynamic shared memory if available
    int device = 0;
    cudaGetDevice(&device);
    int max_optin = 0;
    cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    // Prefer shared memory in cache config
    cudaFuncSetCacheConfig(flash_attention_v2_forward_kernel, cudaFuncCachePreferShared);
    // Set attribute to requested size if within opt-in limit
    if ((int)smem_size <= max_optin) {
        cudaFuncSetAttribute(
            flash_attention_v2_forward_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem_size));
    } else {
        // If requested exceeds opt-in cap, it's safer to early return or adjust tiles
        // Here we early return to avoid invalid launch
        fprintf(stderr, "Requested shared memory %zu exceeds opt-in limit %d. Reduce Br/Bc or D.\n", (size_t)smem_size, max_optin);
        return;
    }
 
    flash_attention_v2_forward_kernel<<<grid, block, smem_size, stream>>>(Q, K, V, O, B, H, N, D, 
    scale, Br, Bc, Tr, Tc);

    // cudaFree(d_l);
    // cudaFree(d_m);
}

} // namespace attention


