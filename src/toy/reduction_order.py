from __future__ import annotations

import argparse
import math
import random
import time

import torch

from src.common import resolve_device, save_csv, save_json, set_seed, synchronize

FP16_SAFE_SUM = 0.5 * torch.finfo(torch.float16).max


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare floating-point reduction orders.")
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeats", type=int, default=10)
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


def overflow_safe_scale(values: torch.Tensor) -> float:
    absolute_sum = float(values.double().abs().sum())
    if absolute_sum == 0.0:
        return 1.0
    return min(1.0, FP16_SAFE_SUM / absolute_sum)


def quantized_input_reference(values: torch.Tensor, input_scale: float) -> float:
    return math.fsum(values.detach().cpu().tolist()) / input_scale


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
    input_scale = overflow_safe_scale(base)
    scaled_base = base * input_scale
    permutation = list(range(args.size))
    random.Random(args.seed).shuffle(permutation)
    rows = []

    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        try:
            values = scaled_base.to(device=device, dtype=dtype)
            reference = quantized_input_reference(values, input_scale)
            methods = {
                "forward": lambda: sequential_sum(values),
                "reverse": lambda: sequential_sum(values.flip(0)),
                "blocked": lambda: blocked_sum(values, args.block_size),
                "random": lambda: sequential_sum(values[permutation]),
                "fixed_tree": lambda: fixed_tree_sum(values),
            }
            for method, function in methods.items():
                for repeat in range(1, args.repeats + 1):
                    scaled_result, elapsed_ms = timed_sum(function, device)
                    if not math.isfinite(scaled_result):
                        raise FloatingPointError(
                            f"{dtype} {method} produced {scaled_result} after "
                            f"overflow-safe input scaling"
                        )
                    result = scaled_result / input_scale
                    absolute_error = abs(result - reference)
                    relative_error = (
                        absolute_error / abs(reference)
                        if reference != 0.0
                        else (0.0 if absolute_error == 0.0 else math.inf)
                    )
                    rows.append({
                        "dtype": str(dtype).removeprefix("torch."),
                        "method": method,
                        "repeat": repeat,
                        "size": args.size,
                        "block_size": args.block_size if method == "blocked" else "",
                        "input_scale": input_scale,
                        "scaled_result": scaled_result,
                        "result": result,
                        "fp64_reference": reference,
                        "reference_basis": "dtype_quantized_input",
                        "absolute_error": absolute_error,
                        "relative_error": relative_error,
                        "elapsed_ms": elapsed_ms,
                        "status": "ok",
                        "reason": "",
                    })
        except RuntimeError as error:
            rows.append(
                {
                    "dtype": str(dtype).removeprefix("torch."),
                    "method": "all",
                    "repeat": "",
                    "size": args.size,
                    "block_size": "",
                    "input_scale": input_scale,
                    "scaled_result": "",
                    "result": "",
                    "fp64_reference": reference,
                    "reference_basis": "dtype_quantized_input",
                    "absolute_error": "",
                    "relative_error": "",
                    "elapsed_ms": "",
                    "status": "skipped",
                    "reason": str(error),
                }
            )

    save_csv(rows, args.output)
    save_json(
        {
            "seed": args.seed,
            "device": str(device),
            "size": args.size,
            "input_scale": input_scale,
            "fp16_safe_sum": FP16_SAFE_SUM,
        },
        args.output.rsplit(".", 1)[0] + ".json",
    )
    print(f"saved reduction-order results to {args.output}")


if __name__ == "__main__":
    main()
