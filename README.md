# attention_xymc

CUDA implementations of naive Attention and a simple FlashAttention-style forward pass with an online softmax. Includes a small benchmark app.

## Build

Requirements:
- CUDA Toolkit 11.4+ (11.8/12.x recommended)
- CMake 3.20+
- GPU with sm_70+

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . -j
```

## Run benchmark

```bash
./bench_attention [B H M N D]
```
Defaults: `B=1 H=8 M=512 N=512 D=64`.

Example:
```bash
./bench_attention 1 8 1024 1024 64
```

## Layout
- `include/attention/attention.h`: Host API
- `src/attention.cpp`: API implementation and launchers
- `src/kernels/naive_attention.cu`: Baseline O(MND) kernel
- `src/kernels/flash_attention.cu`: Streaming FlashAttention-like forward (online softmax)
- `apps/bench_attention.cu`: Benchmark harness

## Notes
This is an educational scaffold focusing on forward pass logic and API shape. The FlashAttention kernel prioritizes clarity over performance and demonstrates the online softmax update pattern.

## License
MIT
