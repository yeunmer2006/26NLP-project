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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/attention_benchmark.csv")
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


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    num_kv_heads = args.num_heads if args.num_kv_heads is None else args.num_kv_heads
    validate_head_counts(args.num_heads, num_kv_heads)
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
            for backend in ("eager", "sdpa", "flash_attn_2"):
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
                if backend == "eager":
                    function = partial(
                        eager_attention, query, key, value, workload == "prefill"
                    )
                elif backend == "sdpa":
                    function = partial(
                        sdpa_attention, query, key, value, workload == "prefill"
                    )
                else:
                    q = query.transpose(1, 2).contiguous()
                    k = key.transpose(1, 2).contiguous()
                    v = value.transpose(1, 2).contiguous()
                    function = partial(flash_function, q, k, v, causal=True)
                try:
                    with torch.inference_mode():
                        reference = sdpa_attention(
                            query, key, value, workload == "prefill"
                        )
                        candidate = function()
                        if backend == "flash_attn_2":
                            candidate = candidate.transpose(1, 2)
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
            "flash_attn_available": flash_available,
            "flash_attn_version": flash_version,
            "flash_attn_api": "flash_attn.flash_attn_func",
            "flash_attn_implementation": (
                "Official Dao-AILab/flash-attention package; no local FA2 kernel reimplementation."
            ),
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
    print(f"saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
