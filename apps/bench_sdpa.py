import argparse
import time
import math
from typing import Tuple

import torch
import torch.nn.functional as F


def compute_attention_flops(B: int, H: int, Nq: int, Nk: int, D: int) -> float:
    # Align with C++: qk: B*H*Nq*Nk*D; softmax ~ 2 ops per element; av: B*H*Nq*Nk*D
    qk = B * H * Nq * Nk * D
    softmax = B * H * Nq * Nk * 2
    av = B * H * Nq * Nk * D
    return float(qk + softmax + av)


def compute_bandwidth_bytes(B: int, H: int, N: int, D: int, bytes_per_elem: int) -> float:
    # Read Q,K,V and write O; match C++ estimate
    read = (B * H * N * D * 3) * bytes_per_elem
    write = (B * H * N * D) * bytes_per_elem
    return float(read + write)


def bench_one(q: torch.Tensor,
              k: torch.Tensor,
              v: torch.Tensor,
              causal: bool,
              warmup: int,
              iters: int,
              enable_flash: bool,
              enable_mem_efficient: bool,
              enable_math: bool) -> float:
    try:
        torch.cuda.synchronize()
        with torch.backends.cuda.sdp_kernel(enable_flash=enable_flash,
                                            enable_math=enable_math,
                                            enable_mem_efficient=enable_mem_efficient):
            for _ in range(warmup):
                F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(iters):
                F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)
            torch.cuda.synchronize()
            elapsed = (time.time() - start) / iters
        return elapsed * 1e3  # ms/iter
    except RuntimeError as e:
        # Gracefully indicate this kernel mode is unavailable
        if "No available kernel" in str(e):
            return float('nan')
        raise


def parse_dtype(s: str):
    s = s.lower()
    if s in ("fp32", "float32", "f32"): return torch.float32
    if s in ("fp16", "float16", "half", "f16"): return torch.float16
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {s}")


def dtype_nbytes(dtype: torch.dtype) -> int:
    if dtype is torch.float32: return 4
    if dtype is torch.float16: return 2
    if dtype is torch.bfloat16: return 2
    raise ValueError(f"Unsupported dtype for bytes: {dtype}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PyTorch SDPA as baseline (flash/mem_efficient/math)")
    parser.add_argument("--B", type=int, default=64)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--Nq", type=int, default=1024, help="query sequence length")
    parser.add_argument("--Nk", type=int, default=None, help="key/value sequence length; default Nq")
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16", "fp32", "fp16", "bf16"])
    parser.add_argument("--causal", action="store_true", help="use causal mask")
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    Nk = args.Nq if args.Nk is None else args.Nk
    dtype = parse_dtype(args.dtype)

    assert torch.cuda.is_available(), "CUDA is required"
    device = torch.device("cuda")

    torch.manual_seed(1234)
    q = torch.randn(args.B, args.H, args.Nq, args.D, device=device, dtype=dtype)
    k = torch.randn(args.B, args.H, Nk, args.D, device=device, dtype=dtype)
    v = torch.randn(args.B, args.H, Nk, args.D, device=device, dtype=dtype)

    ms_flash = bench_one(q, k, v, args.causal, args.warmup, args.iters, True, False, False)
    ms_mem = bench_one(q, k, v, args.causal, args.warmup, args.iters, False, True, False)
    ms_math = bench_one(q, k, v, args.causal, args.warmup, args.iters, False, False, True)

    flops = compute_attention_flops(args.B, args.H, args.Nq, Nk, args.D)
    bytes_total = compute_bandwidth_bytes(args.B, args.H, args.Nq, args.D, dtype_nbytes(dtype))

    def report(name: str, ms: float):
        if math.isnan(ms):
            print(f"{name:<25}{'N/A':>12}{'N/A':>15}{'N/A':>15}")
            return
        gflops = (flops / 1e9) / (ms / 1e3)
        gbps = (bytes_total / 1e9) / (ms / 1e3)
        print(f"{name:<25}{ms:>12.3f} ms{gflops:>15.2f} GFLOPS{gbps:>15.2f} GB/s")

    print("\nSDPA Baseline Results:")
    print("========================================")
    print(f"{'Implementation':<25}{'Time (ms)':>12}{'GFLOPS':>15}{'Bandwidth (GB/s)':>15}")
    print("----------------------------------------")
    report("SDPA flash", ms_flash)
    report("SDPA mem_efficient", ms_mem)
    report("SDPA math", ms_math)


if __name__ == "__main__":
    main()


