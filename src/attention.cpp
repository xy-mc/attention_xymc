#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "attention/attention.h"

namespace attention {

#define CUDA_CHECK(expr)                                                                 \
    do {                                                                                 \
        cudaError_t _err = (expr);                                                       \
        if (_err != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_err), __FILE__, __LINE__); \
        }                                                                                \
    } while (0)

// Forward declarations for kernel launchers defined in .cu files
void launch_naive_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const int B, const int H, const int N, const int D,
    const float scale,
    cudaStream_t stream);

void launch_flash_attention_v1_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const int B, const int H, const int N, const int D,
    const float scale,
    cudaStream_t stream);

static inline float compute_scale(int head_dim) {
    return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

void attention_naive_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const AttentionDims& dims,
    cudaStream_t stream) {
    const float scale = compute_scale(dims.D);
    launch_naive_attention_forward(
        Q, K, V, O,
        dims.B, dims.H, dims.N, dims.D,
        scale,
        stream);
    CUDA_CHECK(cudaGetLastError());
}

void flash_attention_v1_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const AttentionDims& dims,
    cudaStream_t stream) {
    const float scale = compute_scale(dims.D);
    launch_flash_attention_v1_forward(
        Q, K, V, O,
        dims.B, dims.H, dims.N, dims.D,
        scale,
        stream);
    CUDA_CHECK(cudaGetLastError());
}

void launch_standard_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const int B, const int H, const int N, const int D,
    const float scale,
    cudaStream_t stream);

void standard_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const AttentionDims& dims,
    cudaStream_t stream) {
    const float scale = compute_scale(dims.D);
    launch_standard_attention_forward(
        Q, K, V, O,
        dims.B, dims.H, dims.N, dims.D,
        scale,
        stream);
    CUDA_CHECK(cudaGetLastError());
}

void launch_flash_attention_v1_optimize_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const int B, const int H, const int N, const int D,
    const float scale,
    cudaStream_t stream);

// FlashAttention-style streaming forward with online softmax
void flash_attention_v1_optimize_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const AttentionDims& dims,
    cudaStream_t stream) {
    const float scale = compute_scale(dims.D);
    launch_flash_attention_v1_optimize_forward(
        Q, K, V, O,
        dims.B, dims.H, dims.N, dims.D,
        scale,
        stream);
    CUDA_CHECK(cudaGetLastError());
}

void launch_flash_attention_v2_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const int B, const int H, const int N, const int D,
    const float scale,
    cudaStream_t stream);
    
void flash_attention_v2_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const AttentionDims& dims,
    cudaStream_t stream) {
    const float scale = compute_scale(dims.D);
    launch_flash_attention_v2_forward(
        Q, K, V, O,
        dims.B, dims.H, dims.N, dims.D,
        scale,
        stream);
    CUDA_CHECK(cudaGetLastError());
}

void launch_flash_attention_v2_optimize_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const int B, const int H, const int N, const int D,
    const float scale,
    cudaStream_t stream);

// FlashAttention-style streaming forward with online softmax
void flash_attention_v2_optimize_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const AttentionDims& dims,
    cudaStream_t stream) {
    const float scale = compute_scale(dims.D);
    launch_flash_attention_v2_optimize_forward(
        Q, K, V, O,
        dims.B, dims.H, dims.N, dims.D,
        scale,
        stream);
    CUDA_CHECK(cudaGetLastError());
}

void launch_flash_attention_mma_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const int B, const int H, const int N, const int D,
    const float scale,
    cudaStream_t stream);

void flash_attention_mma_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const AttentionDims& dims,
    cudaStream_t stream) {
    const float scale = compute_scale(dims.D);
    launch_flash_attention_mma_forward(
        Q, K, V, O,
        dims.B, dims.H, dims.N, dims.D,
        scale,
        stream);
    CUDA_CHECK(cudaGetLastError());
}

void launch_flash_attention_mma_optimize_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const int B, const int H, const int N, const int D,
    const float scale,
    cudaStream_t stream);

void flash_attention_mma_optimize_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const AttentionDims& dims,
    cudaStream_t stream) {
    const float scale = compute_scale(dims.D);
    launch_flash_attention_mma_optimize_forward(
        Q, K, V, O,
        dims.B, dims.H, dims.N, dims.D,
        scale,
        stream);
    CUDA_CHECK(cudaGetLastError());
}

void launch_flash_attention_mma_Kstage_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const int B, const int H, const int N, const int D,
    const float scale,
    cudaStream_t stream);

void flash_attention_mma_Kstage_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const AttentionDims& dims,
    cudaStream_t stream) {
    const float scale = compute_scale(dims.D);
    launch_flash_attention_mma_Kstage_forward(
        Q, K, V, O,
        dims.B, dims.H, dims.N, dims.D,
        scale,
        stream);
    CUDA_CHECK(cudaGetLastError());
}

extern "C" void flash_attn_target_launch_half(const __half *Q, const __half *K,
                                               const __half *V, __half *O, int B,
                                               int H, int N, int D,
                                               int stages,
                                               cudaStream_t stream);

static inline void device_memcpy_float_to_half(const float* src, __half* dst, size_t n, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

static inline void device_memcpy_half_to_float(const __half* src, float* dst, size_t n, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToDevice, stream);
}

void flash_attention_target_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const AttentionDims& dims,
    cudaStream_t stream) {
    const int B = dims.B;
    const int H = dims.H;
    const int N = dims.N;
    const int D = dims.D;

    const size_t elems = static_cast<size_t>(B) * H * N * D;
    __half *hQ = nullptr, *hK = nullptr, *hV = nullptr, *hO = nullptr;
    CUDA_CHECK(cudaMalloc(&hQ, elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&hK, elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&hV, elems * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&hO, elems * sizeof(__half)));

    device_memcpy_float_to_half(Q, hQ, elems, stream);
    device_memcpy_float_to_half(K, hK, elems, stream);
    device_memcpy_float_to_half(V, hV, elems, stream);

    const int stages = 2;
    flash_attn_target_launch_half(hQ, hK, hV, hO, B, H, N, D, stages, stream);
    CUDA_CHECK(cudaGetLastError());

    device_memcpy_half_to_float(hO, O, elems, stream);
    CUDA_CHECK(cudaFree(hQ));
    CUDA_CHECK(cudaFree(hK));
    CUDA_CHECK(cudaFree(hV));
    CUDA_CHECK(cudaFree(hO));
}

} // namespace attention


