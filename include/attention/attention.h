#pragma once

#include <cuda_runtime.h>

namespace attention {

struct AttentionDims {
    int B;  // batch_size
    int H;  // num_heads  
    int N;  // seq_len (查询和键/值长度相同，自注意力)
    int D;  // head_dim
};

// Naive attention forward: O = softmax(Q K^T / sqrt(d)) V
void attention_naive_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const AttentionDims& dims,
    cudaStream_t stream = nullptr);

// FlashAttention-style streaming forward with online softmax
void flash_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const AttentionDims& dims,
    cudaStream_t stream = nullptr);

// Standard attention with materialized S and P in HBM
// Algorithm:
// 1) S = Q K^T * scale
// 2) P = softmax(S)
// 3) O = P V
void standard_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
    float* O,
    const AttentionDims& dims,
    cudaStream_t stream = nullptr);

} // namespace attention


