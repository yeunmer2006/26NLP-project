from __future__ import annotations

import argparse
import importlib.util
import statistics
import time
from functools import partial
from itertools import product

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
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/attention_benchmark.csv")
    return parser.parse_args()


def eager_attention(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, causal: bool
) -> torch.Tensor:
    scale = query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if causal:
        query_len, key_len = query.shape[-2], key.shape[-2]
        allowed = torch.ones(query_len, key_len, dtype=torch.bool, device=query.device)
        allowed = torch.tril(allowed, diagonal=key_len - query_len)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
    return torch.matmul(torch.softmax(scores.float(), dim=-1).to(query.dtype), value)


def run_timed(function, warmup: int, iterations: int, device: torch.device) -> float:
    for _ in range(warmup):
        function()
    synchronize(device)
    started = time.perf_counter()
    for _ in range(iterations):
        function()
    synchronize(device)
    return (time.perf_counter() - started) * 1000.0 / iterations


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1024**2
    return None


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype == torch.float16:
        dtype = torch.float32
    flash_available = False
    flash_function = None
    flash_reason = "flash_attn unavailable or CUDA unsupported"
    if importlib.util.find_spec("flash_attn") is not None and device.type == "cuda":
        try:
            from flash_attn import flash_attn_func

            flash_function = flash_attn_func
            flash_available = True
            flash_reason = ""
        except (ImportError, OSError) as error:
            flash_reason = f"flash_attn import failed: {error}"

    rows = []
    for batch_size, seq_len in product(args.batch_sizes, args.seq_lens):
        kv_shape = (batch_size, args.num_heads, seq_len, args.head_dim)
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
                        F.scaled_dot_product_attention,
                        query,
                        key,
                        value,
                        is_causal=workload == "prefill",
                    )
                else:
                    q = query.transpose(1, 2)
                    k = key.transpose(1, 2)
                    v = value.transpose(1, 2)
                    function = partial(flash_function, q, k, v, causal=True)
                try:
                    latencies = []
                    memories = []
                    for repeat in range(1, args.repeats + 1):
                        if device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(device)
                        latency_ms = run_timed(function, args.warmup, args.iterations, device)
                        latencies.append(latency_ms)
                        memories.append(peak_memory_mb(device) or 0.0)
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
                            "status": "error",
                            "reason": str(error),
                        }
                    )
            del query
        del key, value

    save_csv(rows, args.output)
    save_json(
        {
            "device": str(device),
            "requested_dtype": args.dtype,
            "flash_attn_available": flash_available,
            "note": (
                "Prefill uses full causal attention. Decode uses one query token against "
                "the existing KV length; it excludes model and KV-cache management overhead."
            ),
        },
        args.output.rsplit(".", 1)[0] + ".json",
    )
    print(f"saved {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
