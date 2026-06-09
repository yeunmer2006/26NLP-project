from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import time
from pathlib import Path

import torch

from src.common import resolve_device, save_csv, save_json, set_seed
from src.determinism.batch_sensitivity import (
    batch_greedy_generate,
    compositions,
    first_divergence,
    target_logits,
)
from src.infer.generate import load_model_and_tokenizer


LOGITS_COMPOSITIONS = ("A_target_only", "C_seven_short", "E_mixed_lengths")
GENERATION_COMPOSITIONS = (
    "A_target_only",
    "B_one_short",
    "C_seven_short",
    "D_one_long",
    "E_mixed_lengths",
)
DEFAULT_BATCH_SIZES = (1, 2, 4, 8)
MODEL_VARIANTS = ("native", "fixed_order")
FIXED_ORDER_VARIANT = "fixed_order"


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(","))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the lightweight full-model batch-invariance validation."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--candidates-input", required=True)
    parser.add_argument("--logits-candidates", type=int, default=20)
    parser.add_argument("--generation-rank", type=int, default=44)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--batch-sizes",
        type=parse_int_list,
        default=DEFAULT_BATCH_SIZES,
    )
    parser.add_argument("--attention-fixed-split-size", type=int, default=64)
    parser.add_argument("--linear-tile-m", type=int, default=16)
    parser.add_argument("--linear-tile-n", type=int, default=256)
    parser.add_argument("--linear-k-block-size", type=int, default=480)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output",
        default=(
            "results/main_experiments_30m_v2/determinism/"
            "improved_model_validation.csv"
        ),
    )
    return parser.parse_args()


def load_candidates(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def token_hash(token_ids: list[int]) -> str:
    payload = ",".join(str(token_id) for token_id in token_ids).encode()
    return hashlib.sha256(payload).hexdigest()[:12]


def read_existing_rows(path: Path, resume: bool) -> list[dict]:
    if not resume or not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row.setdefault("model_variant", FIXED_ORDER_VARIANT)
        if not row["model_variant"]:
            row["model_variant"] = FIXED_ORDER_VARIANT
    return rows


def completed_keys(rows: list[dict]) -> set[tuple[str, str, int, str]]:
    return {
        (
            row.get("model_variant") or FIXED_ORDER_VARIANT,
            row["stage"],
            int(row["candidate_rank"]),
            row["composition"],
        )
        for row in rows
        if row.get("status") == "ok"
    }


def variant_configuration(
    model_variant: str,
    args: argparse.Namespace,
) -> dict[str, str | int]:
    if model_variant == "native":
        return {
            "attention_backend": "sdpa",
            "rms_norm_backend": "native",
            "linear_backend": "native",
        }
    if model_variant == FIXED_ORDER_VARIANT:
        return {
            "attention_backend": "flash_attn_2_bi",
            "rms_norm_backend": "fixed_tree",
            "linear_backend": "fixed_tile",
            "attention_fixed_split_size": args.attention_fixed_split_size,
            "linear_tile_m": args.linear_tile_m,
            "linear_tile_n": args.linear_tile_n,
            "linear_k_block_size": args.linear_k_block_size,
        }
    raise ValueError(f"unknown model variant: {model_variant}")


def configure_model(
    model,
    model_variant: str,
    args: argparse.Namespace,
) -> None:
    configuration = variant_configuration(model_variant, args)
    model.set_batch_invariant_backends(**configuration)
    model.to(dtype=torch.float16)


def summarize_logits(rows: list[dict]) -> dict:
    return {
        "tested_cases": len(rows),
        "bitwise_equal_cases": sum(
            str(row["logits_bitwise_equal"]).lower() == "true" for row in rows
        ),
        "nonzero_difference_cases": sum(
            float(row["max_abs_diff"]) != 0.0 for row in rows
        ),
        "top1_changed_cases": sum(
            str(row["top1_changed"]).lower() == "true" for row in rows
        ),
        "top5_changed_cases": sum(
            str(row["top5_changed"]).lower() == "true" for row in rows
        ),
        "maximum_absolute_difference": max(
            (float(row["max_abs_diff"]) for row in rows),
            default=None,
        ),
    }


def build_summary(rows: list[dict], args: argparse.Namespace) -> dict:
    normalized_rows = []
    for row in rows:
        normalized = dict(row)
        normalized["model_variant"] = (
            normalized.get("model_variant") or FIXED_ORDER_VARIANT
        )
        normalized_rows.append(normalized)
    logits_rows_by_variant = {
        model_variant: [
            row
            for row in normalized_rows
            if row["stage"] == "logits"
            and row.get("status") == "ok"
            and row["model_variant"] == model_variant
        ]
        for model_variant in MODEL_VARIANTS
    }
    generation_rows = [
        row for row in normalized_rows
        if row["stage"] == "generation"
        and row.get("status") == "ok"
        and row["model_variant"] == FIXED_ORDER_VARIANT
    ]
    batch_size_rows = [
        row for row in normalized_rows
        if row["stage"] == "batch_size"
        and row.get("status") == "ok"
        and row["model_variant"] == FIXED_ORDER_VARIANT
    ]
    return {
        "checkpoint": args.checkpoint,
        "candidates_input": args.candidates_input,
        "device": args.device,
        "dtype": "float16",
        "configuration": {
            "model_variants": {
                model_variant: variant_configuration(model_variant, args)
                for model_variant in MODEL_VARIANTS
            },
            "logits_candidates": args.logits_candidates,
            "logits_compositions": list(LOGITS_COMPOSITIONS),
            "logits_positions": "last_valid_token_only",
            "generation_rank": args.generation_rank,
            "generation_compositions": list(GENERATION_COMPOSITIONS),
            "batch_sizes": list(args.batch_sizes),
            "max_new_tokens": args.max_new_tokens,
        },
        "logits_comparison": {
            model_variant: summarize_logits(logits_rows_by_variant[model_variant])
            for model_variant in MODEL_VARIANTS
        },
        "logits_summary": summarize_logits(
            logits_rows_by_variant[FIXED_ORDER_VARIANT]
        ),
        "generation_summary": {
            "tested_cases": len(generation_rows),
            "identical_cases": sum(
                str(row["output_identical"]).lower() == "true"
                for row in generation_rows
            ),
            "all_outputs_identical": bool(generation_rows) and all(
                str(row["output_identical"]).lower() == "true"
                for row in generation_rows
            ),
        },
        "generation_cases": [
            {
                "candidate_rank": int(row["candidate_rank"]),
                "composition": row["composition"],
                "batch_size": int(row["batch_size"]),
                "token_ids": json.loads(row["generated_token_ids"]),
                "token_hash": row["token_hash"],
                "generated_text": row["generated_text"],
                "output_identical": str(row["output_identical"]).lower() == "true",
                "first_divergence_token": row["first_divergence_token"],
            }
            for row in generation_rows
        ],
        "batch_size_summary": {
            "tested_cases": len(batch_size_rows),
            "bitwise_equal_cases": sum(
                str(row["logits_bitwise_equal"]).lower() == "true"
                for row in batch_size_rows
            ),
            "identical_generation_cases": sum(
                str(row["output_identical"]).lower() == "true"
                for row in batch_size_rows
            ),
            "all_logits_bitwise_equal": bool(batch_size_rows) and all(
                str(row["logits_bitwise_equal"]).lower() == "true"
                for row in batch_size_rows
            ),
            "all_outputs_identical": bool(batch_size_rows) and all(
                str(row["output_identical"]).lower() == "true"
                for row in batch_size_rows
            ),
        },
        "batch_size_cases": [
            {
                "candidate_rank": int(row["candidate_rank"]),
                "batch_size": int(row["batch_size"]),
                "logits_bitwise_equal": (
                    str(row["logits_bitwise_equal"]).lower() == "true"
                ),
                "max_abs_diff": float(row["max_abs_diff"]),
                "token_ids": json.loads(row["generated_token_ids"]),
                "token_hash": row["token_hash"],
                "generated_text": row["generated_text"],
                "output_identical": str(row["output_identical"]).lower() == "true",
                "first_divergence_token": row["first_divergence_token"],
            }
            for row in batch_size_rows
        ],
        "total_case_elapsed_seconds": sum(
            float(row["elapsed_seconds"])
            for row in normalized_rows
            if row.get("status") == "ok" and row.get("elapsed_seconds")
        ),
    }


def save_progress(rows: list[dict], output: Path, args: argparse.Namespace) -> None:
    save_csv(rows, output)
    save_json(build_summary(rows, args), output.with_suffix(".json"))


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=True)
    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError("the FP16 improved-model validation requires CUDA")
    args.device = str(device)

    candidates = load_candidates(Path(args.candidates_input))
    required = max(args.logits_candidates, args.generation_rank)
    if len(candidates) < required:
        raise ValueError(f"candidates input must contain at least {required} rows")

    output = Path(args.output)
    rows = read_existing_rows(output, args.resume)
    done = completed_keys(rows)
    started = time.perf_counter()

    fixed_model = None
    fixed_tokenizer = None
    for model_variant in MODEL_VARIANTS:
        model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)
        configure_model(model, model_variant, args)
        for rank, candidate in enumerate(candidates[:args.logits_candidates], start=1):
            pending_compositions = [
                composition
                for composition in LOGITS_COMPOSITIONS
                if (model_variant, "logits", rank, composition) not in done
            ]
            if not pending_compositions:
                continue
            prompt = candidate["prompt"]
            cases = compositions(prompt)
            baseline_logits = target_logits(
                model,
                tokenizer,
                cases["A_target_only"],
                device,
                select_logits=True,
            )
            baseline_top5 = torch.topk(baseline_logits.float(), 5).indices.tolist()
            for composition in pending_compositions:
                case_started = time.perf_counter()
                prompts = cases[composition]
                logits = (
                    baseline_logits
                    if composition == "A_target_only"
                    else target_logits(
                        model,
                        tokenizer,
                        prompts,
                        device,
                        select_logits=True,
                    )
                )
                difference = (logits.float() - baseline_logits.float()).abs()
                top5 = torch.topk(logits.float(), 5).indices.tolist()
                rows.append(
                    {
                        "model_variant": model_variant,
                        "stage": "logits",
                        "candidate_rank": rank,
                        "prompt": prompt,
                        "composition": composition,
                        "batch_size": len(prompts),
                        "dtype": "float16",
                        "model_dtype": str(next(model.parameters()).dtype),
                        "logits_dtype": str(logits.dtype),
                        "max_abs_diff": float(difference.max()),
                        "mean_abs_diff": float(difference.mean()),
                        "logits_bitwise_equal": torch.equal(logits, baseline_logits),
                        "top1_changed": top5[0] != baseline_top5[0],
                        "top5_changed": top5 != baseline_top5,
                        "output_identical": "",
                        "first_divergence_token": "",
                        "generated_token_ids": "",
                        "token_hash": "",
                        "generated_text": "",
                        "elapsed_seconds": time.perf_counter() - case_started,
                        "status": "ok",
                        "reason": "",
                    }
                )
                save_progress(rows, output, args)
            print(
                f"{model_variant} logits candidate "
                f"{rank}/{args.logits_candidates} completed",
                flush=True,
            )
        if model_variant == FIXED_ORDER_VARIANT:
            fixed_model = model
            fixed_tokenizer = tokenizer
        else:
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()

    if fixed_model is None or fixed_tokenizer is None:
        raise RuntimeError("fixed-order model was not initialized")
    model = fixed_model
    tokenizer = fixed_tokenizer

    generation_candidate = candidates[args.generation_rank - 1]
    generation_prompt = generation_candidate["prompt"]
    generation_cases = compositions(generation_prompt)
    pending_generation = [
        composition
        for composition in GENERATION_COMPOSITIONS
        if (
            FIXED_ORDER_VARIANT,
            "generation",
            args.generation_rank,
            composition,
        ) not in done
    ]
    baseline_generated = None
    if pending_generation:
        baseline_generated = batch_greedy_generate(
            model,
            tokenizer,
            generation_cases["A_target_only"],
            args.max_new_tokens,
            device,
            select_logits=True,
        )[0]
        for composition in pending_generation:
            case_started = time.perf_counter()
            prompts = generation_cases[composition]
            generated = (
                baseline_generated
                if composition == "A_target_only"
                else batch_greedy_generate(
                    model,
                    tokenizer,
                    prompts,
                    args.max_new_tokens,
                    device,
                    select_logits=True,
                )[0]
            )
            divergence = first_divergence(baseline_generated, generated)
            rows.append(
                {
                    "model_variant": FIXED_ORDER_VARIANT,
                    "stage": "generation",
                    "candidate_rank": args.generation_rank,
                    "prompt": generation_prompt,
                    "composition": composition,
                    "batch_size": len(prompts),
                    "dtype": "float16",
                    "model_dtype": str(next(model.parameters()).dtype),
                    "logits_dtype": "torch.float16",
                    "max_abs_diff": "",
                    "mean_abs_diff": "",
                    "logits_bitwise_equal": "",
                    "top1_changed": "",
                    "top5_changed": "",
                    "output_identical": divergence is None,
                    "first_divergence_token": (
                        "" if divergence is None else divergence
                    ),
                    "generated_token_ids": json.dumps(generated),
                    "token_hash": token_hash(generated),
                    "generated_text": tokenizer.decode(generated),
                    "elapsed_seconds": time.perf_counter() - case_started,
                    "status": "ok",
                    "reason": "",
                }
            )
            save_progress(rows, output, args)
            print(f"generation composition {composition} completed", flush=True)

    pending_batch_sizes = [
        batch_size
        for batch_size in args.batch_sizes
        if (
            FIXED_ORDER_VARIANT,
            "batch_size",
            args.generation_rank,
            f"same_prompt_x{batch_size}",
        ) not in done
    ]
    if pending_batch_sizes:
        baseline_logits = target_logits(
            model,
            tokenizer,
            [generation_prompt],
            device,
            select_logits=True,
        )
        if baseline_generated is None:
            baseline_generated = batch_greedy_generate(
                model,
                tokenizer,
                [generation_prompt],
                args.max_new_tokens,
                device,
                select_logits=True,
            )[0]
        baseline_top5 = torch.topk(baseline_logits.float(), 5).indices.tolist()
        for batch_size in pending_batch_sizes:
            case_started = time.perf_counter()
            prompts = [generation_prompt] * batch_size
            if batch_size == 1:
                logits = baseline_logits
                generated = baseline_generated
            else:
                logits = target_logits(
                    model,
                    tokenizer,
                    prompts,
                    device,
                    select_logits=True,
                )
                generated = batch_greedy_generate(
                    model,
                    tokenizer,
                    prompts,
                    args.max_new_tokens,
                    device,
                    select_logits=True,
                )[0]
            difference = (logits.float() - baseline_logits.float()).abs()
            top5 = torch.topk(logits.float(), 5).indices.tolist()
            divergence = first_divergence(baseline_generated, generated)
            rows.append(
                {
                    "model_variant": FIXED_ORDER_VARIANT,
                    "stage": "batch_size",
                    "candidate_rank": args.generation_rank,
                    "prompt": generation_prompt,
                    "composition": f"same_prompt_x{batch_size}",
                    "batch_size": batch_size,
                    "dtype": "float16",
                    "model_dtype": str(next(model.parameters()).dtype),
                    "logits_dtype": str(logits.dtype),
                    "max_abs_diff": float(difference.max()),
                    "mean_abs_diff": float(difference.mean()),
                    "logits_bitwise_equal": torch.equal(logits, baseline_logits),
                    "top1_changed": top5[0] != baseline_top5[0],
                    "top5_changed": top5 != baseline_top5,
                    "output_identical": divergence is None,
                    "first_divergence_token": (
                        "" if divergence is None else divergence
                    ),
                    "generated_token_ids": json.dumps(generated),
                    "token_hash": token_hash(generated),
                    "generated_text": tokenizer.decode(generated),
                    "elapsed_seconds": time.perf_counter() - case_started,
                    "status": "ok",
                    "reason": "",
                }
            )
            save_progress(rows, output, args)
            print(f"same-prompt batch size {batch_size} completed", flush=True)

    summary = build_summary(rows, args)
    summary["last_run_elapsed_seconds"] = time.perf_counter() - started
    save_csv(rows, output)
    save_json(summary, output.with_suffix(".json"))
    print(
        f"saved {len(rows)} validation rows to {output}; "
        f"elapsed {summary['last_run_elapsed_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()
