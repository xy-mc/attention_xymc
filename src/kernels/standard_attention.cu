#include <cuda_runtime.h>
#include <float.h>

namespace attention {

// BHND: [B, H, N, D]
__device__ __forceinline__ int index_BHND(int b, int h, int n, int d, int B, int H, int N, int D) {
    return ((b * H + h) * N + n) * D + d;
}

// BHNN: [B, H, N, N] for S and P
__device__ __forceinline__ int index_BHNN(int b, int h, int n, int m, int B, int H, int N) {
    return ((b * H + h) * N + n) * N + m;
}

// Pass 1: compute S = Q K^T * scale, materialized to HBM
__global__ void compute_scores_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    float* __restrict__ S,
    const int B, const int H, const int N, const int D,
    const float scale) {
    const int row = blockIdx.x; // (b,h,n)
    const int total_rows = B * H * N;
    if (row >= total_rows) return;

    const int bh = row / N;
    const int n = row % N;
    const int b = bh / H;
    const int h = bh % H;
    

    for (int m = threadIdx.x; m < N; m += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            const float qv = Q[index_BHND(b, h, n, d, B, H, N, D)];
            const float kv = K[index_BHND(b, h, m, d, B, H, N, D)];
            dot += qv * kv;
        }
        S[index_BHNN(b, h, n, m, B, H, N)] = dot * scale;
    }
}

// Pass 2: P = softmax(S) row-wise in place of S -> P
__global__ void softmax_rows_kernel(
    const float* __restrict__ S,
    float* __restrict__ P,
    const int B, const int H, const int N) {
    const int row = blockIdx.x; // (b,h,n)
    const int total_rows = B * H * N;
    if (row >= total_rows) return;

    const int bh = row / N;
    const int n = row % N;
    const int b = bh / H;
    const int h = bh % H;

    // compute max
    float max_val = -FLT_MAX;
    for (int m = 0; m < N; ++m) {
        float v = S[index_BHNN(b, h, n, m, B, H, N)];
        if (v > max_val) max_val = v;
    }
    // compute exp and sum
    float sum = 0.0f;
    for (int m = 0; m < N; ++m) {
        float e = __expf(S[index_BHNN(b, h, n, m, B, H, N)] - max_val);
        P[index_BHNN(b, h, n, m, B, H, N)] = e;
        sum += e;
    }
    // normalize
    const float inv_sum = 1.0f / sum;
    for (int m = 0; m < N; ++m) {
        P[index_BHNN(b, h, n, m, B, H, N)] *= inv_sum;
    }
}

// Pass 3: O = P V
__global__ void apply_values_kernel(
    const float* __restrict__ P,
    const float* __restrict__ V,
    float* __restrict__ O,
    const int B, const int H, const int N, const int D) {
    const int row = blockIdx.x; // (b,h,n)
    const int total_rows = B * H * N;
    if (row >= total_rows) return;

    const int bh = row / N;
    const int n = row % N;
    const int b = bh / H;
    const int h = bh % H;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int m = 0; m < N; ++m) {
            const float p = P[index_BHNN(b, h, n, m, B, H, N)];
            const float vv = V[index_BHND(b, h, m, d, B, H, N, D)];
            acc += p * vv;
        }
        O[index_BHND(b, h, n, d, B, H, N, D)] = acc;
    }
}

void launch_standard_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const int B, const int H, const int N, const int D,
    const float scale,
    cudaStream_t stream) {
    const int total_rows = B * H * N;
    const size_t size_scores = static_cast<size_t>(B) * H * N * N;

    float* S = nullptr; // scores
    float* P = nullptr; // probabilities
    cudaMalloc(&S, size_scores * sizeof(float));
    cudaMalloc(&P, size_scores * sizeof(float));

    // reasonable default block sizes
    dim3 grid(total_rows);
    dim3 block_scores(128);
    dim3 block_values(128);

    compute_scores_kernel<<<grid, block_scores, 0, stream>>>(Q, K, S, B, H, N, D, scale);
    softmax_rows_kernel<<<grid, 1, 0, stream>>>(S, P, B, H, N);
    apply_values_kernel<<<grid, block_values, 0, stream>>>(P, V, O, B, H, N, D);

    cudaFree(S);
    cudaFree(P);
}

} // namespace attention



