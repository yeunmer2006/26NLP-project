from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
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


def parse_int_list(value: str) -> list[int]:
    return [int(item) for item in value.split(",")]


def precision_context(device: torch.device, dtype: str):
    if dtype == "float32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=getattr(torch, dtype))


def configure_model(model, attention_backend: str, norm_backend: str) -> None:
    model.config.attention_backend = attention_backend
    model.config.rms_norm_backend = norm_backend
    model.norm.backend = norm_backend
    for layer in model.layers:
        layer.attention.backend = attention_backend
        layer.input_norm.backend = norm_backend
        layer.post_attention_norm.backend = norm_backend


def sampled_documents(path: Path, count: int, seed: int) -> list[str]:
    reservoir = []
    randomizer = random.Random(seed)
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            text = json.loads(line).get("text", "").strip()
            if not text:
                continue
            if len(reservoir) < count:
                reservoir.append(text)
            else:
                replacement = randomizer.randint(0, index)
                if replacement < count:
                    reservoir[replacement] = text
    return reservoir


@torch.inference_mode()
def rank_candidates(
    model,
    tokenizer,
    documents: list[str],
    prefix_lengths: list[int],
    keep: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict], int]:
    candidates = []
    seen = set()
    for document_index, text in enumerate(documents):
        token_ids = tokenizer.encode(text, add_bos=True)
        for prefix_length in prefix_lengths:
            if len(token_ids) <= prefix_length:
                continue
            prefix_ids = token_ids[:prefix_length]
            key = tuple(prefix_ids)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((document_index, prefix_length, prefix_ids))

    ranked = []
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start:start + batch_size]
        width = max(len(item[2]) for item in chunk)
        input_ids = torch.full(
            (len(chunk), width),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.zeros_like(input_ids)
        lengths = []
        for row, (_, _, ids) in enumerate(chunk):
            input_ids[row, :len(ids)] = torch.tensor(ids, device=device)
            attention_mask[row, :len(ids)] = 1
            lengths.append(len(ids))
        logits = model(input_ids, attention_mask=attention_mask)["logits"]
        row_ids = torch.arange(len(chunk), device=device)
        last = logits[row_ids, torch.tensor(lengths, device=device) - 1].float()
        top2 = torch.topk(last, 2, dim=-1)
        for item, values, indices in zip(chunk, top2.values.cpu(), top2.indices.cpu()):
            document_index, prefix_length, ids = item
            ranked.append({
                "document_index": document_index,
                "prefix_length": prefix_length,
                "prompt": tokenizer.decode(ids),
                "top1_token_id": int(indices[0]),
                "top2_token_id": int(indices[1]),
                "top1_logit": float(values[0]),
                "top2_logit": float(values[1]),
                "margin": float(values[0] - values[1]),
            })
    ranked.sort(key=lambda row: row["margin"])
    return ranked[:keep], len(candidates)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search low-margin prompts for greedy divergence.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--validation-jsonl",
        default="data/processed/tinystories/raw/validation.jsonl",
    )
    parser.add_argument("--documents", type=int, default=2000)
    parser.add_argument(
        "--prefix-lengths", type=parse_int_list, default=[8, 16, 32, 64, 128]
    )
    parser.add_argument("--keep", type=int, default=100)
    parser.add_argument("--ranking-batch-size", type=int, default=32)
    parser.add_argument("--backends", default="eager,sdpa")
    parser.add_argument("--dtypes", default="float32,float16")
    parser.add_argument("--norm-backends", default="native")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--candidates-output", default="results/prompt_margin_candidates.csv"
    )
    parser.add_argument("--output", default="results/divergence_search.csv")
    args = parser.parse_args()

    set_seed(args.seed, deterministic=True)
    device = resolve_device(args.device)
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)
    documents = sampled_documents(Path(args.validation_jsonl), args.documents, args.seed)
    configure_model(model, "sdpa", "native")
    ranking_dtype = "float16" if device.type == "cuda" else "float32"
    with precision_context(device, ranking_dtype):
        candidates, candidate_count = rank_candidates(
            model,
            tokenizer,
            documents,
            args.prefix_lengths,
            args.keep,
            args.ranking_batch_size,
            device,
        )
    save_csv(candidates, args.candidates_output)
    save_json({
        "checkpoint": args.checkpoint,
        "documents_requested": args.documents,
        "documents_loaded": len(documents),
        "prefix_lengths": args.prefix_lengths,
        "candidate_count": candidate_count,
        "kept": len(candidates),
        "minimum_margin": candidates[0]["margin"] if candidates else None,
        "seed": args.seed,
        "ranking_dtype": ranking_dtype,
    }, args.candidates_output.rsplit(".", 1)[0] + ".json")

    rows = []
    details = {}
    for norm_backend in args.norm_backends.split(","):
        for attention_backend in args.backends.split(","):
            configure_model(model, attention_backend, norm_backend)
            for dtype in args.dtypes.split(","):
                if device.type == "cpu" and dtype == "float16":
                    continue
                with precision_context(device, dtype):
                    for rank, candidate in enumerate(candidates, start=1):
                        prompt = candidate["prompt"]
                        cases = compositions(prompt)
                        baseline_logits = target_logits(
                            model, tokenizer, cases["A_target_only"], device
                        )
                        baseline_top5 = torch.topk(baseline_logits, 5).indices.tolist()
                        baseline_generated = batch_greedy_generate(
                            model,
                            tokenizer,
                            cases["A_target_only"],
                            args.max_new_tokens,
                            device,
                        )[0]
                        for composition, prompts in cases.items():
                            logits = target_logits(model, tokenizer, prompts, device)
                            top5 = torch.topk(logits, 5).indices.tolist()
                            difference = (logits - baseline_logits).abs()
                            generated = (
                                baseline_generated
                                if composition == "A_target_only"
                                else batch_greedy_generate(
                                    model,
                                    tokenizer,
                                    prompts,
                                    args.max_new_tokens,
                                    device,
                                )[0]
                            )
                            divergence = first_divergence(
                                baseline_generated, generated
                            )
                            row = {
                                "candidate_rank": rank,
                                "prompt": prompt,
                                "prefix_length": candidate["prefix_length"],
                                "baseline_margin": candidate["margin"],
                                "attention_backend": attention_backend,
                                "norm_backend": norm_backend,
                                "dtype": dtype,
                                "composition": composition,
                                "batch_size": len(prompts),
                                "max_abs_diff": float(difference.max()),
                                "mean_abs_diff": float(difference.mean()),
                                "top1_changed": top5[0] != baseline_top5[0],
                                "top5_changed": top5 != baseline_top5,
                                "output_identical": divergence is None,
                                "first_divergence_token": (
                                    "" if divergence is None else divergence
                                ),
                                "status": "ok",
                                "reason": "",
                            }
                            rows.append(row)
                            if divergence is not None:
                                key = (
                                    f"{rank}/{norm_backend}/{attention_backend}/"
                                    f"{dtype}/{composition}"
                                )
                                details[key] = {
                                    **row,
                                    "prompts": prompts,
                                    "baseline_token_ids": baseline_generated,
                                    "generated_token_ids": generated,
                                    "baseline_text": tokenizer.decode(baseline_generated),
                                    "generated_text": tokenizer.decode(generated),
                                }
                        if rank % 10 == 0 or rank == len(candidates):
                            print(
                                f"validated {rank}/{len(candidates)} candidates "
                                f"for {norm_backend}/{attention_backend}/{dtype}",
                                flush=True,
                            )
    save_csv(rows, args.output)
    divergences = [
        row for row in rows
        if row["status"] == "ok" and not row["output_identical"]
    ]
    save_json({
        "checkpoint": args.checkpoint,
        "device": str(device),
        "search_budget": {
            "documents": len(documents),
            "candidate_count": candidate_count,
            "kept": len(candidates),
            "prefix_lengths": args.prefix_lengths,
            "max_new_tokens": args.max_new_tokens,
        },
        "minimum_margin": candidates[0]["margin"] if candidates else None,
        "tested_cases": len(rows),
        "divergence_count": len(divergences),
        "found_divergence": bool(divergences),
        "divergences": details,
    }, args.output.rsplit(".", 1)[0] + ".json")
    print(
        f"tested {len(rows)} cases from {candidate_count} candidates; "
        f"found {len(divergences)} divergences"
    )


if __name__ == "__main__":
    main()
