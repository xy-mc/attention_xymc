#include <cuda_runtime.h>
#include <float.h>

namespace attention {

// BHND格式的索引计算: [B, H, N, D]
__device__ __forceinline__ int index_BHND(int b, int h, int n, int d, int B, int H, int N, int D) {
    return ((b * H + h) * N + n) * D + d;
}

// A simple FlashAttention-style streaming kernel.
// One block processes one (b,h,n) row. For simplicity, a single thread does the work.
__global__ void flash_attention_forward_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int B, int H, int N, int D,
    float scale) {
    const int row = blockIdx.x;
    const int total_rows = B * H * N;
    if (row >= total_rows) return;

    const int bh = row / N;
    const int n = row % N;
    const int b = bh / H;
    const int h = bh % H;

    if (threadIdx.x == 0) {
        float m_i = -FLT_MAX; // running max
        float l_i = 0.0f;     // running sum of exp

        // Accumulator for output vector
        // Note: this uses local memory for D that may spill. For clarity over performance.
        extern __shared__ float shared_acc[]; // size D
        float* acc = shared_acc;
        for (int d = 0; d < D; ++d) {
            acc[d] = 0.0f;
        }

        // Stream over keys
        for (int j = 0; j < N; ++j) {
            // Compute score s = (q·k) * scale
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                const float qv = Q[index_BHND(b, h, n, d, B, H, N, D)];
                const float kv = K[index_BHND(b, h, j, d, B, H, N, D)];
                dot += qv * kv;
            }
            float s = dot * scale;

            // Update running max and sum in an online softmax manner
            float m_new = fmaxf(m_i, s);
            float l_new = __expf(m_i - m_new) * l_i + __expf(s - m_new);
            float alpha = (l_i == 0.0f) ? 0.0f : __expf(m_i - m_new) * (l_i / l_new);
            float beta = __expf(s - m_new) / l_new;

            // Update accumulator acc = alpha * acc + beta * v
            for (int d = 0; d < D; ++d) {
                const float vv = V[index_BHND(b, h, j, d, B, H, N, D)];
                acc[d] = alpha * acc[d] + beta * vv;
            }

            m_i = m_new;
            l_i = l_new;
        }

        // Write result
        for (int d = 0; d < D; ++d) {
            O[index_BHND(b, h, n, d, B, H, N, D)] = acc[d];
        }
    }
}

void launch_flash_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B, int H, int N, int D,
    float scale,
    cudaStream_t stream) {
    const int total_rows = B * H * N;
    dim3 grid(total_rows);
    dim3 block(1);
    size_t shared_bytes = static_cast<size_t>(D) * sizeof(float);
    flash_attention_forward_kernel<<<grid, block, shared_bytes, stream>>>(
        Q, K, V, O, B, H, N, D, scale);
}

} // namespace attention


