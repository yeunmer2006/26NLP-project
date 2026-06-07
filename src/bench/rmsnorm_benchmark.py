from __future__ import annotations

import argparse
import statistics
import time
from itertools import product

import torch

from src.common import resolve_device, save_csv, save_json, synchronize
from src.model import RMSNorm


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def run_timed(function, warmup: int, iterations: int, device: torch.device) -> float:
    for _ in range(warmup):
        function()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark native and fixed-tree RMSNorm.")
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 4, 8])
    parser.add_argument("--seq-lens", type=parse_int_list, default=[128, 256, 512])
    parser.add_argument("--hidden-size", type=int, default=480)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"],
                        default="float16")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/rmsnorm_benchmark.csv")
    parser.add_argument("--invariance-output", default="results/rmsnorm_invariance.csv")
    args = parser.parse_args()

    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    rows = []
    invariance = []
    for batch_size, seq_len in product(args.batch_sizes, args.seq_lens):
        values = torch.randn(
            batch_size, seq_len, args.hidden_size, device=device, dtype=dtype
        )
        target = values[:1, :1].clone()
        mixed = torch.cat((target, torch.randn_like(target).expand(7, -1, -1)), dim=0)
        for backend in ("native", "fixed_tree"):
            norm = RMSNorm(args.hidden_size, 1e-5, backend).to(device=device, dtype=dtype)
            with torch.inference_mode():
                single = norm(target)[0, 0]
                batched = norm(mixed)[0, 0]
                invariance.append({
                    "backend": backend,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "bitwise_equal": torch.equal(single, batched),
                    "max_abs_diff": float((single.float() - batched.float()).abs().max()),
                })
                latencies = []
                memories = []
                for _ in range(args.repeats):
                    if device.type == "cuda":
                        torch.cuda.reset_peak_memory_stats(device)
                    latencies.append(run_timed(
                        lambda: norm(values), args.warmup, args.iterations, device
                    ))
                    memories.append(
                        torch.cuda.max_memory_allocated(device) / 1024**2
                        if device.type == "cuda" else 0.0
                    )
            mean_latency = statistics.mean(latencies)
            std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
            for repeat, (latency, memory) in enumerate(
                zip(latencies, memories), start=1
            ):
                rows.append({
                    "backend": backend,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "hidden_size": args.hidden_size,
                    "dtype": str(dtype),
                    "repeat": repeat,
                    "latency_ms": latency,
                    "mean_latency_ms": mean_latency,
                    "std_latency_ms": std_latency,
                    "tokens_per_second": batch_size * seq_len / (latency / 1000.0),
                    "peak_memory_mb": memory,
                    "status": "ok",
                    "reason": "",
                })
    save_csv(rows, args.output)
    prefix = args.output.rsplit(".", 1)[0]
    save_json({
        "device": str(device),
        "requested_dtype": args.dtype,
        "hidden_size": args.hidden_size,
    }, prefix + ".json")
    save_csv(invariance, args.invariance_output)
    save_json({
        "all_fixed_tree_bitwise_equal": all(
            row["bitwise_equal"] for row in invariance
            if row["backend"] == "fixed_tree"
        ),
        "cases": invariance,
    }, args.invariance_output.rsplit(".", 1)[0] + ".json")


if __name__ == "__main__":
    main()
