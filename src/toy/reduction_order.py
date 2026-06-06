from __future__ import annotations

import argparse
import random
import time

import torch

from src.common import resolve_device, save_csv, save_json, set_seed, synchronize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare floating-point reduction orders.")
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="results/reduction_order.csv")
    return parser.parse_args()


def sequential_sum(values: torch.Tensor) -> torch.Tensor:
    result = torch.zeros((), dtype=values.dtype, device=values.device)
    for value in values:
        result = result + value
    return result


def blocked_sum(values: torch.Tensor, block_size: int) -> torch.Tensor:
    partials = [
        sequential_sum(values[start : start + block_size])
        for start in range(0, values.numel(), block_size)
    ]
    return sequential_sum(torch.stack(partials))


def fixed_tree_sum(values: torch.Tensor) -> torch.Tensor:
    level = values
    while level.numel() > 1:
        if level.numel() % 2:
            level = torch.cat((level, torch.zeros(1, dtype=level.dtype, device=level.device)))
        level = level[0::2] + level[1::2]
    return level[0]


def timed_sum(function, device: torch.device) -> tuple[float, float]:
    synchronize(device)
    started = time.perf_counter()
    result = function()
    synchronize(device)
    return float(result), (time.perf_counter() - started) * 1000.0


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    generator = torch.Generator().manual_seed(args.seed)
    # Mixed magnitudes amplify rounding-order effects while remaining reproducible.
    exponents = torch.randint(-4, 5, (args.size,), generator=generator)
    signs = torch.randint(0, 2, (args.size,), generator=generator) * 2 - 1
    base = signs.float() * torch.pow(10.0, exponents.float())
    reference = float(base.double().sum())
    permutation = list(range(args.size))
    random.Random(args.seed).shuffle(permutation)
    rows = []

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        try:
            values = base.to(device=device, dtype=dtype)
            methods = {
                "forward": lambda: sequential_sum(values),
                "reverse": lambda: sequential_sum(values.flip(0)),
                "blocked": lambda: blocked_sum(values, args.block_size),
                "random": lambda: sequential_sum(values[permutation]),
                "fixed_tree": lambda: fixed_tree_sum(values),
            }
            for method, function in methods.items():
                result, elapsed_ms = timed_sum(function, device)
                rows.append(
                    {
                        "dtype": str(dtype).removeprefix("torch."),
                        "method": method,
                        "size": args.size,
                        "block_size": args.block_size if method == "blocked" else "",
                        "result": result,
                        "fp64_reference": reference,
                        "absolute_error": abs(result - reference),
                        "elapsed_ms": elapsed_ms,
                        "status": "ok",
                        "reason": "",
                    }
                )
        except RuntimeError as error:
            rows.append(
                {
                    "dtype": str(dtype).removeprefix("torch."),
                    "method": "all",
                    "size": args.size,
                    "block_size": "",
                    "result": "",
                    "fp64_reference": reference,
                    "absolute_error": "",
                    "elapsed_ms": "",
                    "status": "skipped",
                    "reason": str(error),
                }
            )

    save_csv(rows, args.output)
    save_json(
        {"seed": args.seed, "device": str(device), "size": args.size},
        args.output.rsplit(".", 1)[0] + ".json",
    )
    print(f"saved reduction-order results to {args.output}")


if __name__ == "__main__":
    main()
