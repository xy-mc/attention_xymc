#include <cuda_runtime.h>
#include <float.h>

namespace attention {

// BHND格式的索引计算: [B, H, N, D]
__device__ __forceinline__ int index_BHND(int b, int h, int n, int d, int B, int H, int N, int D) {
    return ((b * H + h) * N + n) * D + d;
}

__global__ void naive_attention_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int B, const int H, const int N, const int D,
    const float scale) {
    
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    
    int n = threadIdx.x;

    // Compute max score for stability
    float max_score = -FLT_MAX;
    
    for (int m = 0; m < N; m++) {
        float res = 0.0f;
        for (int d = 0; d < D; d++) {
            const float qv = Q[index_BHND(b,h,n,d,B,H,N,D)];
            const float kv = K[index_BHND(b,h,m,d,B,H,N,D)];
            res += qv * kv;
        }
        res *= scale;
        if (res > max_score) max_score = res;
    }

    const float inv_denom = 1.0 / __expf(max_score);
    float acc = 0.0f;
    for (int m = 0; m < N; m++) {
        float res = 0.0f;
        for (int d = 0; d < D; d++) {
            const float qv = Q[index_BHND(b,h,n,d,B,H,N,D)];
            const float kv = K[index_BHND(b,h,m,d,B,H,N,D)];
            res += qv * kv;
        }
        res *= scale;
        acc += __expf(res - max_score);
    }

    for (int d = 0; d < D; d++) {
        float ans = 0.0f;
        for (int m = 0; m < N; m++) {
            float res = 0.0f;
            for (int dd = 0; dd < D; dd++) {
                const float qv = Q[index_BHND(b,h,n,dd,B,H,N,D)];
                const float kv = K[index_BHND(b,h,m,dd,B,H,N,D)];
                res += qv * kv;
            }
            
            res = __expf(res * scale) * inv_denom / acc;
            ans += res * V[index_BHND(b,h,m,d,B,H,N,D)];
        }
        O[index_BHND(b,h,n,d,B,H,N,D)] = ans;
    }
}

void launch_naive_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const int B, const int H, const int N, const int D,
    const float scale,
    cudaStream_t stream) {
    dim3 grid(B, H);
    dim3 block(N);
    naive_attention_forward_kernel<<<grid, block, 0, stream>>>(Q, K, V, O, B, H, N, D, scale);
}

} // namespace attention


