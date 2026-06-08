from __future__ import annotations

import argparse
import importlib.util
import statistics
import time
from functools import partial
from itertools import product
from typing import Callable

import torch
import torch.nn.functional as F

from src.common import resolve_device, save_csv, save_json, synchronize
from src.model import flash_attention_2_batch_invariant


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark eager, SDPA, and FlashAttention-2.")
    parser.add_argument("--batch-sizes", type=parse_int_list, default=[1, 4, 8])
    parser.add_argument("--seq-lens", type=parse_int_list, default=[128, 512, 1024])
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument(
        "--num-kv-heads",
        type=int,
        default=None,
        help="KV heads for MQA/GQA. Defaults to --num-heads (standard MHA).",
    )
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--fixed-split-size", type=int, default=64)
    parser.add_argument("--backends", default="eager,sdpa,flash_attn_2")
    parser.add_argument(
        "--invariance-backends",
        default="eager,sdpa,flash_attn_2,flash_attn_2_bi",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/attention_benchmark.csv")
    parser.add_argument("--invariance-output", default="results/attention_invariance.csv")
    return parser.parse_args()


def validate_head_counts(num_heads: int, num_kv_heads: int) -> None:
    if num_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num-heads and num-kv-heads must be positive")
    if num_heads % num_kv_heads != 0:
        raise ValueError("num-heads must be divisible by num-kv-heads")


def repeat_kv(hidden_states: torch.Tensor, num_heads: int) -> torch.Tensor:
    num_kv_heads = hidden_states.shape[1]
    validate_head_counts(num_heads, num_kv_heads)
    return hidden_states.repeat_interleave(num_heads // num_kv_heads, dim=1)


def bottom_right_causal_mask(
    query_len: int, key_len: int, device: torch.device | None = None
) -> torch.Tensor:
    query_positions = torch.arange(query_len, device=device) + key_len - query_len
    key_positions = torch.arange(key_len, device=device)
    return key_positions[None, :] <= query_positions[:, None]


def eager_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal: bool
) -> torch.Tensor:
    key = repeat_kv(key, query.shape[1])
    value = repeat_kv(value, query.shape[1])
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    allowed = None
    if causal:
        query_len, key_len = query.shape[-2], key.shape[-2]
        allowed = bottom_right_causal_mask(query_len, key_len, query.device)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    probabilities = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    if allowed is not None:
        # FlashAttention-2 returns zero for a causal-mask row with no valid keys.
        probabilities = probabilities.masked_fill(~allowed, 0.0)
    return torch.matmul(probabilities, value)


def sdpa_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal: bool
) -> torch.Tensor:
    key = repeat_kv(key, query.shape[1])
    value = repeat_kv(value, query.shape[1])
    if not causal or query.shape[-2] == key.shape[-2]:
        return F.scaled_dot_product_attention(query, key, value, is_causal=causal)
    allowed = bottom_right_causal_mask(query.shape[-2], key.shape[-2], query.device)
    return F.scaled_dot_product_attention(
        query, key, value, attn_mask=allowed[None, None, :, :]
    )


def batch_invariant_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
    fixed_split_size: int = 64,
) -> torch.Tensor:
    key = repeat_kv(key, query.shape[1])
    value = repeat_kv(value, query.shape[1])
    past_len = key.shape[-2] - query.shape[-2] if causal else 0
    return flash_attention_2_batch_invariant(
        query,
        key,
        value,
        past_len=past_len,
        block_size=fixed_split_size,
        causal=causal,
    )


def run_backend(
    backend: str,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool,
    flash_function: Callable | None,
    fixed_split_size: int,
) -> torch.Tensor:
    if backend == "eager":
        return eager_attention(query, key, value, causal)
    if backend == "sdpa":
        return sdpa_attention(query, key, value, causal)
    if backend == "flash_attn_2_bi":
        return batch_invariant_attention(query, key, value, causal, fixed_split_size)
    if backend == "flash_attn_2":
        if flash_function is None:
            raise RuntimeError("flash_attn is unavailable")
        output = flash_function(
            query.transpose(1, 2).contiguous(),
            key.transpose(1, 2).contiguous(),
            value.transpose(1, 2).contiguous(),
            causal=causal,
        )
        return output.transpose(1, 2)
    raise ValueError(f"unknown attention backend: {backend}")


def run_timed(
    function: Callable[[], torch.Tensor],
    warmup: int,
    iterations: int,
    device: torch.device,
) -> float:
    for _ in range(warmup):
        function()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / iterations


def peak_memory_mb(device: torch.device, baseline_bytes: int) -> float | None:
    if device.type == "cuda":
        peak_bytes = torch.cuda.max_memory_allocated(device)
        return max(peak_bytes - baseline_bytes, 0) / 1024**2
    return None


def attention_invariance_rows(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    num_kv_heads: int,
    backends: list[str],
    flash_available: bool,
    flash_function: Callable | None,
    flash_reason: str,
) -> list[dict]:
    rows = []
    mixed_batch_size = max([2, *args.batch_sizes])
    for seq_len in args.seq_lens:
        for workload, query_len in (("prefill", seq_len), ("decode", 1)):
            single_key = torch.randn(
                (1, num_kv_heads, seq_len, args.head_dim),
                device=device,
                dtype=dtype,
            )
            single_value = torch.randn_like(single_key)
            single_query = torch.randn(
                (1, args.num_heads, query_len, args.head_dim),
                device=device,
                dtype=dtype,
            )
            mixed_key = torch.randn(
                (mixed_batch_size, num_kv_heads, seq_len, args.head_dim),
                device=device,
                dtype=dtype,
            )
            mixed_value = torch.randn_like(mixed_key)
            mixed_query = torch.randn(
                (mixed_batch_size, args.num_heads, query_len, args.head_dim),
                device=device,
                dtype=dtype,
            )
            mixed_key[0] = single_key[0]
            mixed_value[0] = single_value[0]
            mixed_query[0] = single_query[0]
            for backend in backends:
                row_base = {
                    "workload": workload,
                    "backend": backend,
                    "single_batch_size": 1,
                    "mixed_batch_size": mixed_batch_size,
                    "seq_len": seq_len,
                    "num_heads": args.num_heads,
                    "num_kv_heads": num_kv_heads,
                    "head_dim": args.head_dim,
                    "dtype": str(dtype),
                }
                if backend == "flash_attn_2" and not flash_available:
                    rows.append({
                        **row_base,
                        "bitwise_equal": "",
                        "max_abs_diff": "",
                        "mean_abs_diff": "",
                        "status": "skipped",
                        "reason": flash_reason,
                    })
                    continue
                try:
                    with torch.inference_mode():
                        single = run_backend(
                            backend,
                            single_query,
                            single_key,
                            single_value,
                            workload == "prefill",
                            flash_function,
                            args.fixed_split_size,
                        )
                        mixed = run_backend(
                            backend,
                            mixed_query,
                            mixed_key,
                            mixed_value,
                            workload == "prefill",
                            flash_function,
                            args.fixed_split_size,
                        )[0:1]
                        difference = (single.float() - mixed.float()).abs()
                    rows.append({
                        **row_base,
                        "bitwise_equal": bool(torch.equal(single, mixed)),
                        "max_abs_diff": float(difference.max()),
                        "mean_abs_diff": float(difference.mean()),
                        "status": "ok",
                        "reason": "",
                    })
                except RuntimeError as error:
                    rows.append({
                        **row_base,
                        "bitwise_equal": "",
                        "max_abs_diff": "",
                        "mean_abs_diff": "",
                        "status": "error",
                        "reason": str(error),
                    })
    return rows


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    num_kv_heads = args.num_heads if args.num_kv_heads is None else args.num_kv_heads
    validate_head_counts(args.num_heads, num_kv_heads)
    if args.fixed_split_size <= 0:
        raise ValueError("fixed-split-size must be positive")
    backends = [backend.strip() for backend in args.backends.split(",") if backend.strip()]
    valid_backends = {"eager", "sdpa", "flash_attn_2", "flash_attn_2_bi"}
    unknown_backends = sorted(set(backends) - valid_backends)
    if unknown_backends:
        raise ValueError(f"unknown backends: {', '.join(unknown_backends)}")
    invariance_backends = [
        backend.strip() for backend in args.invariance_backends.split(",")
        if backend.strip()
    ]
    unknown_invariance_backends = sorted(set(invariance_backends) - valid_backends)
    if unknown_invariance_backends:
        raise ValueError(
            f"unknown invariance backends: {', '.join(unknown_invariance_backends)}"
        )
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    flash_available = False
    flash_function = None
    flash_version = None
    flash_reason = "flash_attn unavailable or CUDA unsupported"
    if importlib.util.find_spec("flash_attn") is not None and device.type == "cuda":
        try:
            import flash_attn
            from flash_attn import flash_attn_func

            flash_function = flash_attn_func
            flash_version = flash_attn.__version__
            flash_available = True
            flash_reason = ""
        except (ImportError, OSError) as error:
            flash_reason = f"flash_attn import failed: {error}"

    rows = []
    for batch_size, seq_len in product(args.batch_sizes, args.seq_lens):
        kv_shape = (batch_size, num_kv_heads, seq_len, args.head_dim)
        key = torch.randn(kv_shape, device=device, dtype=dtype)
        value = torch.randn_like(key)
        for workload, query_len in (("prefill", seq_len), ("decode", 1)):
            query = torch.randn(
                (batch_size, args.num_heads, query_len, args.head_dim),
                device=device,
                dtype=dtype,
            )
            for backend in backends:
                row_base = {
                    "workload": workload,
                    "backend": backend,
                    "batch_size": batch_size,
                    "seq_len": seq_len,
                    "num_heads": args.num_heads,
                    "num_kv_heads": num_kv_heads,
                    "head_dim": args.head_dim,
                    "dtype": str(dtype),
                }
                if backend == "flash_attn_2" and not flash_available:
                    rows.append(
                        {
                            **row_base,
                            "repeat": "",
                            "latency_ms": "",
                            "mean_latency_ms": "",
                            "std_latency_ms": "",
                            "tokens_per_second": "",
                            "peak_memory_mb": "",
                            "status": "skipped",
                            "reason": flash_reason,
                        }
                    )
                    continue
                function = partial(
                    run_backend,
                    backend,
                    query,
                    key,
                    value,
                    workload == "prefill",
                    flash_function,
                    args.fixed_split_size,
                )
                try:
                    with torch.inference_mode():
                        reference = sdpa_attention(
                            query, key, value, workload == "prefill"
                        )
                        candidate = function()
                        error = (candidate.float() - reference.float()).abs()
                        max_abs_error = float(error.max())
                        mean_abs_error = float(error.mean())
                    latencies = []
                    memories = []
                    for repeat in range(1, args.repeats + 1):
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                            baseline_bytes = torch.cuda.memory_allocated(device)
                        else:
                            baseline_bytes = 0
                        latency_ms = run_timed(function, args.warmup, args.iterations, device)
                        latencies.append(latency_ms)
                        memories.append(peak_memory_mb(device, baseline_bytes) or 0.0)
                    mean_latency = statistics.mean(latencies)
                    std_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
                    processed_tokens = batch_size * (seq_len if workload == "prefill" else 1)
                    for repeat, (latency_ms, memory_mb) in enumerate(
                        zip(latencies, memories), start=1
                    ):
                        rows.append({
                            **row_base,
                            "repeat": repeat,
                            "latency_ms": latency_ms,
                            "mean_latency_ms": mean_latency,
                            "std_latency_ms": std_latency,
                            "tokens_per_second": processed_tokens / (latency_ms / 1000.0),
                            "peak_memory_mb": memory_mb,
                            "max_abs_error_vs_sdpa": max_abs_error,
                            "mean_abs_error_vs_sdpa": mean_abs_error,
                            "status": "ok",
                            "reason": "",
                        })
                except RuntimeError as error:
                    rows.append(
                        {
                            **row_base,
                            "repeat": "",
                            "latency_ms": "",
                            "mean_latency_ms": "",
                            "std_latency_ms": "",
                            "tokens_per_second": "",
                            "peak_memory_mb": "",
                            "max_abs_error_vs_sdpa": "",
                            "mean_abs_error_vs_sdpa": "",
                            "status": "error",
                            "reason": str(error),
                        }
                    )
            del query
        del key, value

    save_csv(rows, args.output)
    invariance = attention_invariance_rows(
        args,
        device,
        dtype,
        num_kv_heads,
        invariance_backends,
        flash_available,
        flash_function,
        flash_reason,
    )
    save_csv(invariance, args.invariance_output)
    means = {
        (row["workload"], row["backend"], row["batch_size"], row["seq_len"]):
        row["mean_latency_ms"]
        for row in rows if row["status"] == "ok"
    }
    comparisons = []
    for workload, batch_size, seq_len in product(
        ("prefill", "decode"), args.batch_sizes, args.seq_lens
    ):
        eager = means.get((workload, "eager", batch_size, seq_len))
        sdpa = means.get((workload, "sdpa", batch_size, seq_len))
        flash = means.get((workload, "flash_attn_2", batch_size, seq_len))
        comparisons.append({
            "workload": workload,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "sdpa_speedup_vs_eager": eager / sdpa if eager and sdpa else None,
            "flash_speedup_vs_eager": eager / flash if eager and flash else None,
            "flash_speedup_vs_sdpa": sdpa / flash if sdpa and flash else None,
        })
    save_json(
        {
            "device": str(device),
            "requested_dtype": args.dtype,
            "num_heads": args.num_heads,
            "num_kv_heads": num_kv_heads,
            "head_dim": args.head_dim,
            "fixed_split_size": args.fixed_split_size,
            "backends": backends,
            "invariance_backends": invariance_backends,
            "flash_attn_available": flash_available,
            "flash_attn_version": flash_version,
            "flash_attn_api": "flash_attn.flash_attn_func",
            "flash_attn_implementation": (
                "flash_attn_2 uses the official Dao-AILab package. flash_attn_2_bi is "
                "a local FlashAttention-2-style fixed-order online-softmax reference path."
            ),
            "batch_invariant_attention_backend": "flash_attn_2_bi",
            "attention_invariance_output": args.invariance_output,
            "comparisons": comparisons,
            "note": (
                "This is a forward-only inference microbenchmark, not the paper's combined "
                "forward+backward A100 benchmark. Prefill uses full causal attention. Decode "
                "uses one query token against the existing KV length with FlashAttention-2's "
                "bottom-right causal-mask alignment; it excludes KV-cache management overhead."
            ),
        },
        args.output.rsplit(".", 1)[0] + ".json",
    )
    print(
        f"saved {len(rows)} benchmark rows to {args.output}; "
        f"saved {len(invariance)} invariance rows to {args.invariance_output}"
    )


if __name__ == "__main__":
    main()
