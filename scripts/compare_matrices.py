#!/usr/bin/env python3
import argparse
import re
from typing import List, Tuple


FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def extract_numbers_from_line(line: str) -> List[float]:
    # Skip headers like: "Batch 0, Head 0:" or blank lines
    if line.strip().startswith("Batch ") or not line.strip():
        return []
    nums: List[float] = []
    # Prefer fast token parse; fallback to regex for robustness
    for token in line.strip().split():
        try:
            nums.append(float(token))
        except ValueError:
            # Token may contain trailing commas or other chars; scan with regex
            for m in FLOAT_RE.finditer(token):
                try:
                    nums.append(float(m.group(0)))
                except ValueError:
                    pass
    return nums


def read_all_numbers(path: str) -> List[float]:
    values: List[float] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            values.extend(extract_numbers_from_line(line))
    return values


def is_close(a: float, b: float, abs_tol: float, rel_tol: float) -> bool:
    # |a-b| <= max(abs_tol, rel_tol * max(|a|, |b|))
    diff = abs(a - b)
    scale = rel_tol * max(abs(a), abs(b))
    return diff <= (abs_tol if abs_tol > scale else scale)


def compare_numbers(
    a_vals: List[float],
    b_vals: List[float],
    abs_tol: float,
    rel_tol: float,
    topk: int,
) -> Tuple[int, int, int, List[Tuple[int, float, float, float, float]]]:
    n = min(len(a_vals), len(b_vals))
    diff_examples: List[Tuple[int, float, float, float, float]] = []
    equal_count = 0
    diff_count = 0
    for i in range(n):
        a = a_vals[i]
        b = b_vals[i]
        if is_close(a, b, abs_tol, rel_tol):
            equal_count += 1
        else:
            diff_count += 1
            if len(diff_examples) < topk:
                abs_diff = abs(a - b)
                denom = max(abs(b), 1e-12)
                rel_diff = abs_diff / denom
                diff_examples.append((i, a, b, abs_diff, rel_diff))
    return n, equal_count, diff_count, diff_examples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare numeric contents of two attention output files."
    )
    parser.add_argument("file_a", help="Path to first file (e.g., reference)")
    parser.add_argument("file_b", help="Path to second file (e.g., variant)")
    parser.add_argument(
        "--abs_tol", type=float, default=0.0,
        help="Absolute tolerance (default: 0.0)"
    )
    parser.add_argument(
        "--rel_tol", type=float, default=0.0,
        help="Relative tolerance (default: 0.0)"
    )
    parser.add_argument(
        "--topk", type=int, default=10,
        help="Show top-K differing examples (default: 10)"
    )
    args = parser.parse_args()

    a_vals = read_all_numbers(args.file_a)
    b_vals = read_all_numbers(args.file_b)

    total_a = len(a_vals)
    total_b = len(b_vals)
    if total_a == 0 or total_b == 0:
        msg = (
            f"Parsed counts -> {args.file_a}: {total_a}, "
            f"{args.file_b}: {total_b}"
        )
        print(msg)
        print("No numbers to compare.")
        return

    n, equal_count, diff_count, diff_examples = compare_numbers(
        a_vals, b_vals, args.abs_tol, args.rel_tol, args.topk
    )

    print("Comparison summary")
    print("===================")
    print(f"File A: {args.file_a}")
    print(f"File B: {args.file_b}")
    print(
        f"Numbers parsed -> A: {total_a}, B: {total_b}"
    )
    if n != total_a or n != total_b:
        print(f"Warning: Different lengths; compared first {n} numbers.")
    ratio = diff_count / n if n > 0 else 0.0
    print(f"Compared: {n}")
    print(f"Equal (within tol): {equal_count}")
    print(f"Different: {diff_count} ({ratio:.6%})")
    print()

    if diff_examples:
        print("Examples (index, A, B, abs_diff, rel_diff vs B)")
        for idx, a, b, ad, rd in diff_examples:
            print(f"{idx}\t{a:.10f}\t{b:.10f}\t{ad:.10e}\t{rd:.10e}")


if __name__ == "__main__":
    main()
