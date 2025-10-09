#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>

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
    int B, int H, int N, int D,
    float scale,
    cudaStream_t stream);

void launch_flash_attention_v1_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    int B, int H, int N, int D,
    float scale,
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
    int B, int H, int N, int D,
    float scale,
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
    int B, int H, int N, int D,
    float scale,
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




} // namespace attention


