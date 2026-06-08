from __future__ import annotations

import argparse
import statistics
import time
from functools import partial
from itertools import product

import torch
import torch.nn.functional as F

from src.common import resolve_device, save_csv, save_json, synchronize
from src.model import fixed_tile_matmul


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def parse_shapes(value: str) -> list[tuple[str, int, int]]:
    shapes = []
    for item in value.split(","):
        name, in_features, out_features = item.split(":")
        shapes.append((name, int(in_features), int(out_features)))
    return shapes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark native and fixed-tile matmul.")
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 4, 8])
    parser.add_argument("--seq-lens", type=parse_int_list, default=[1, 128, 512])
    parser.add_argument(
        "--shapes",
        type=parse_shapes,
        default=[("hidden", 480, 480), ("mlp_up", 480, 1280), ("lm_head", 480, 259)],
        help="Comma-separated name:in_features:out_features entries.",
    )
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--tile-m", type=int, default=16)
    parser.add_argument("--tile-n", type=int, default=64)
    parser.add_argument("--k-block-size", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/matmul_benchmark.csv")
    parser.add_argument("--invariance-output", default="results/matmul_invariance.csv")
    return parser.parse_args()


def run_timed(function, warmup: int, iterations: int, device: torch.device) -> float:
    for _ in range(warmup):
        function()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / iterations


def run_backend(
    backend: str,
    values: torch.Tensor,
    weight: torch.Tensor,
    tile_m: int,
    tile_n: int,
    k_block_size: int,
) -> torch.Tensor:
    if backend == "native":
        return F.linear(values, weight)
    if backend == "fixed_tile":
        return fixed_tile_matmul(
            values,
            weight,
            tile_m=tile_m,
            tile_n=tile_n,
            k_block_size=k_block_size,
        )
    raise ValueError(f"unknown backend: {backend}")


def main() -> None:
    args = parse_args()
    if args.tile_m <= 0 or args.tile_n <= 0 or args.k_block_size <= 0:
        raise ValueError("tile-m, tile-n, and k-block-size must be positive")
    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32

    rows = []
    invariance_rows = []
    backends = ("native", "fixed_tile")
    for shape_name, in_features, out_features in args.shapes:
        weight = torch.randn(out_features, in_features, device=device, dtype=dtype)
        target = torch.randn(1, 1, in_features, device=device, dtype=dtype)
        mixed = torch.randn(8, 1, in_features, device=device, dtype=dtype)
        mixed[0] = target[0]
        for backend in backends:
            with torch.inference_mode():
                single = run_backend(
                    backend,
                    target,
                    weight,
                    args.tile_m,
                    args.tile_n,
                    args.k_block_size,
                )[0, 0]
                batched = run_backend(
                    backend,
                    mixed,
                    weight,
                    args.tile_m,
                    args.tile_n,
                    args.k_block_size,
                )[0, 0]
                difference = (single.float() - batched.float()).abs()
            invariance_rows.append({
                "shape": shape_name,
                "backend": backend,
                "single_batch_size": 1,
                "mixed_batch_size": mixed.shape[0],
                "in_features": in_features,
                "out_features": out_features,
                "dtype": str(dtype),
                "bitwise_equal": bool(torch.equal(single, batched)),
                "max_abs_diff": float(difference.max()),
                "mean_abs_diff": float(difference.mean()),
                "status": "ok",
                "reason": "",
            })

        for batch_size, seq_len in product(args.batch_sizes, args.seq_lens):
            values = torch.randn(batch_size, seq_len, in_features, device=device, dtype=dtype)
            for backend in backends:
                function = partial(
                    run_backend,
                    backend,
                    values,
                    weight,
                    args.tile_m,
                    args.tile_n,
                    args.k_block_size,
                )
                with torch.inference_mode():
                    reference = run_backend(
                        "native",
                        values,
                        weight,
                        args.tile_m,
                        args.tile_n,
                        args.k_block_size,
                    )
                    candidate = function()
                    error = (candidate.float() - reference.float()).abs()
                latencies = []
                memories = []
                for _ in range(args.repeats):
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    latencies.append(run_timed(function, args.warmup, args.iterations, device))
                    memories.append(
                        torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda" else 0.0
                    )
                mean_latency = statistics.mean(latencies)
                std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
                tokens = batch_size * seq_len
                for repeat, (latency_ms, memory_mb) in enumerate(
                    zip(latencies, memories), start=1
                ):
                    rows.append({
                        "shape": shape_name,
                        "backend": backend,
                        "batch_size": batch_size,
                        "seq_len": seq_len,
                        "in_features": in_features,
                        "out_features": out_features,
                        "dtype": str(dtype),
                        "tile_m": args.tile_m,
                        "tile_n": args.tile_n,
                        "k_block_size": args.k_block_size,
                        "repeat": repeat,
                        "latency_ms": latency_ms,
                        "mean_latency_ms": mean_latency,
                        "std_latency_ms": std_latency,
                        "tokens_per_second": tokens / (latency_ms / 1000.0),
                        "peak_memory_mb": memory_mb,
                        "max_abs_error_vs_native": float(error.max()),
                        "mean_abs_error_vs_native": float(error.mean()),
                        "status": "ok",
                        "reason": "",
                    })

    save_csv(rows, args.output)
    save_csv(invariance_rows, args.invariance_output)
    save_json(
        {
            "device": str(device),
            "requested_dtype": args.dtype,
            "tile_m": args.tile_m,
            "tile_n": args.tile_n,
            "k_block_size": args.k_block_size,
            "shapes": [
                {"name": name, "in_features": in_features, "out_features": out_features}
                for name, in_features, out_features in args.shapes
            ],
            "fixed_tile_note": (
                "fixed_tile uses fixed 2D output tiles and fixed K-block traversal. "
                "It validates batch invariance but is not a fused tensor-core GEMM."
            ),
        },
        args.output.rsplit(".", 1)[0] + ".json",
    )
    save_json(
        {
            "all_fixed_tile_bitwise_equal": all(
                row["bitwise_equal"]
                for row in invariance_rows
                if row["backend"] == "fixed_tile"
            ),
            "cases": invariance_rows,
        },
        args.invariance_output.rsplit(".", 1)[0] + ".json",
    )
    print(
        f"saved {len(rows)} matmul benchmark rows to {args.output}; "
        f"saved {len(invariance_rows)} invariance rows to {args.invariance_output}"
    )


if __name__ == "__main__":
    main()
