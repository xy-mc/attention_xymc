import argparse
import math
import time
from typing import Tuple

import torch
import torch.nn.functional as F


def compute_attention_flops(B: int, H: int, N: int, D: int) -> float:
    # QK^T: B*H*N*N*D; Softmax: B*H*N*N*2; P@V: B*H*N*N*D
    qk = float(B) * H * N * N * D
    sm = float(B) * H * N * N * 2
    pv = float(B) * H * N * N * D
    return qk + sm + pv


def compute_attention_bandwidth_bytes(B: int, H: int, N: int, D: int, dtype: torch.dtype) -> float:
    elem_size = torch.tensor([], dtype=dtype).element_size()
    reads = (B * H * N * D) * 3 * elem_size  # Q, K, V
    writes = (B * H * N * D) * elem_size     # O
    return float(reads + writes)


def make_qkv(B: int, H: int, N: int, D: int, dtype: torch.dtype, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(0)
    q = torch.randn((B, H, N, D), dtype=dtype, device=device, generator=generator)
    k = torch.randn((B, H, N, D), dtype=dtype, device=device, generator=generator)
    v = torch.randn((B, H, N, D), dtype=dtype, device=device, generator=generator)
    return q, k, v


def sdpa_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # torch >=2.0: scaled_dot_product_attention expects (..., L, E) shapes
    return F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)


def naive_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, H, N, D = q.shape
    scale = 1.0 / math.sqrt(D)
    q2 = q.reshape(B * H, N, D)
    k2 = k.reshape(B * H, N, D)
    v2 = v.reshape(B * H, N, D)
    scores = torch.matmul(q2, k2.transpose(-1, -2)) * scale
    p = torch.softmax(scores, dim=-1)
    o2 = torch.matmul(p, v2)
    return o2.reshape(B, H, N, D)


def benchmark_once(fn, q, k, v, iters: int, warmup: int) -> Tuple[float, torch.Tensor]:
    # CUDA events for accurate timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # warmup
    for _ in range(warmup):
        _ = fn(q, k, v)
    torch.cuda.synchronize()

    start.record()
    out = None
    for _ in range(iters):
        out = fn(q, k, v)
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = start.elapsed_time(end) / max(iters, 1)
    return float(elapsed_ms), out


def main():
    parser = argparse.ArgumentParser(description="PyTorch Attention Benchmark (SDPA vs Naive)")
    parser.add_argument("--B", type=int, default=1, help="batch size")
    parser.add_argument("--H", type=int, default=8, help="num heads")
    parser.add_argument("--N", type=int, default=1024, help="sequence length")
    parser.add_argument("--D", type=int, default=64, help="head dimension")
    parser.add_argument("--dtype", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="dtype")
    parser.add_argument("--iters", type=int, default=50, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=10, help="warmup iterations")
    parser.add_argument("--allow_tf32", action="store_true", help="enable TF32 for matmul")
    parser.add_argument("--which", type=str, default="both", choices=["both", "sdpa", "naive"], help="which implementation to run")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA is required"
    device = torch.device("cuda")

    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.allow_tf32)

    B, H, N, D = args.B, args.H, args.N, args.D
    q, k, v = make_qkv(B, H, N, D, dtype=dtype, device=device)

    gflops_total = compute_attention_flops(B, H, N, D) / 1e9
    gb_total = compute_attention_bandwidth_bytes(B, H, N, D, dtype) / 1e9

    print(f"Config: B={B}, H={H}, N={N}, D={D}, dtype={args.dtype}, TF32={args.allow_tf32}")
    print(f"Iters: {args.iters} (warmup={args.warmup})")
    print("========================================")
    print(f"Theoretical work: {gflops_total:.2f} GFLOPs, Traffic: {gb_total:.2f} GB")
    print("\nResults:")
    print(f"{'Impl':<18}{'Time (ms)':>12}{'GFLOPS':>12}{'GB/s':>12}")
    print("----------------------------------------")

    if args.which in ("sdpa", "both"):
        ms, _ = benchmark_once(sdpa_forward, q, k, v, iters=args.iters, warmup=args.warmup)
        gflops = gflops_total / (ms / 1000.0)
        gbps = gb_total / (ms / 1000.0)
        print(f"{'SDPA':<18}{ms:>12.3f}{gflops:>12.2f}{gbps:>12.2f}")

    if args.which in ("naive", "both"):
        ms, _ = benchmark_once(naive_forward, q, k, v, iters=args.iters, warmup=args.warmup)
        gflops = gflops_total / (ms / 1000.0)
        gbps = gb_total / (ms / 1000.0)
        print(f"{'Naive':<18}{ms:>12.3f}{gflops:>12.2f}{gbps:>12.2f}")


if __name__ == "__main__":
    main()
