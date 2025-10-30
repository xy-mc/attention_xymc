#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(expr)                                                                         \
    do {                                                                                          \
        cudaError_t _err = (expr);                                                                \
        if (_err != cudaSuccess) {                                                                \
            fprintf(stderr, "CUDA error %s at %s:%d -> %s\n", #expr, __FILE__, __LINE__,        \
                    cudaGetErrorString(_err));                                                    \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

static __host__ __device__ __forceinline__
uint32_t swizzle_Q(uint32_t y, uint32_t x) {
    x >>= 2;
    return ((y & 7u) ^ (x & 7u)) << 2;
}

__global__ void swizzle_kernel(const uint32_t* ys, const uint32_t* xs, uint32_t* outs, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        outs[tid] = swizzle_Q(ys[tid], xs[tid]);
    }
}

int main() {
    const int N = 64 * 64; // cover y in [0,15], x in [0,63]
    std::vector<uint32_t> h_y(N), h_x(N), h_out(N), h_ref(N);
    for (int i = 0; i < N; i += 4) {
        h_y[i] = static_cast<uint32_t>(i / 64);
        h_x[i] = static_cast<uint32_t>(i % 64);      // diverse x values
        h_ref[i] = swizzle_Q(h_y[i], h_x[i]);   // host reference
    }

    uint32_t *d_y = nullptr, *d_x = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_y, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_x, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(d_y, h_y.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));

    dim3 block(64);
    dim3 grid((N + block.x - 1) / block.x);
    swizzle_kernel<<<grid, block>>>(d_y, d_x, d_out, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != h_ref[i]) ++mismatches;
    }

    printf("swizzle_Q test (N=%d)\n", N);
    for (int i = 0; i < N; ++i) {
        printf("i=%2d  y=%2u  x=%2u  out(dev)=%2u  ref(host)=%2u%s\n", i,
               h_y[i], h_x[i], h_out[i], h_ref[i], (h_out[i]==h_ref[i]?"":"  <- MISMATCH"));
    }
    if (mismatches == 0) {
        printf("All results match.\n");
    } else {
        printf("Mismatches: %d\n", mismatches);
    }

    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}


