from __future__ import annotations

import argparse
import time

import torch

from src.common import resolve_device, save_csv, save_json, set_seed, synchronize
from src.toy.reduction_order import blocked_sum, fixed_tree_sum


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare block-dependent and fixed reductions.")
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--block-sizes", type=parse_int_list, default=[16, 32, 64, 128, 256])
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float32")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/batch_invariant_reduction.csv")
    return parser.parse_args()


def measure(function, repeats: int, device: torch.device) -> tuple[float, float]:
    times = []
    result = None
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter()
        result = function()
        synchronize(device)
        times.append((time.perf_counter() - started) * 1000.0)
    return float(result), sum(times) / len(times)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator().manual_seed(args.seed)
    exponents = torch.randint(-4, 5, (args.size,), generator=generator)
    signs = torch.randint(0, 2, (args.size,), generator=generator) * 2 - 1
    source = signs.float() * torch.pow(10.0, exponents.float())
    values = source.to(device=device, dtype=dtype)
    reference = float(source.double().sum())
    rows = []
    ordinary_results = []
    fixed_results = []

    for block_size in args.block_sizes:
        ordinary, ordinary_ms = measure(
            lambda block_size=block_size: blocked_sum(values, block_size),
            args.repeats,
            device,
        )
        fixed, fixed_ms = measure(lambda: fixed_tree_sum(values), args.repeats, device)
        ordinary_results.append(ordinary)
        fixed_results.append(fixed)
        rows.extend(
            [
                {
                    "method": "block_dependent",
                    "block_size": block_size,
                    "result": ordinary,
                    "fp64_reference": reference,
                    "absolute_error": abs(ordinary - reference),
                    "elapsed_ms": ordinary_ms,
                    "matches_first_setting": ordinary == ordinary_results[0],
                },
                {
                    "method": "fixed_tree",
                    "block_size": block_size,
                    "result": fixed,
                    "fp64_reference": reference,
                    "absolute_error": abs(fixed - reference),
                    "elapsed_ms": fixed_ms,
                    "matches_first_setting": fixed == fixed_results[0],
                },
            ]
        )

    save_csv(rows, args.output)
    save_json(
        {
            "seed": args.seed,
            "device": str(device),
            "dtype": args.dtype,
            "ordinary_unique_results": len(set(ordinary_results)),
            "fixed_tree_unique_results": len(set(fixed_results)),
            "interpretation": (
                "A fixed tree preserves the arithmetic order across block settings, "
                "trading scheduling freedom for reproducibility."
            ),
        },
        args.output.rsplit(".", 1)[0] + ".json",
    )
    print(f"saved batch-invariant reduction results to {args.output}")


if __name__ == "__main__":
    main()
