from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path

import torch

from src.common import resolve_device, save_csv, save_json, set_seed
from src.infer.generate import load_model_and_tokenizer


DEFAULT_PROMPTS = [
    "Once upon a time",
    "The little girl opened the door and",
    "A friendly dragon lived",
    "Tom found a red ball",
    "The dog was afraid because",
    "In a small village",
    "Lily wanted to help her friend",
    "The old tree could speak",
    "One sunny morning",
    "The treasure was hidden under",
]


SHORT_DISTRACTORS = [
    "The sky is blue.",
    "Two plus two is four.",
    "A short unrelated sentence.",
    "Paris is in France.",
    "Water freezes at zero degrees.",
    "A model maps tokens to logits.",
    "This prompt is deliberately brief.",
]
LONG_DISTRACTOR = " ".join(
    [
        "This is a longer distractor prompt used to alter the shape and composition of a batch."
        " It contains several clauses and repeated context without sharing the target meaning."
    ]
    * 6
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure target logits across batch compositions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target", default="The capital of France is")
    parser.add_argument("--prompts-file")
    parser.add_argument("--backends", default="eager,sdpa")
    parser.add_argument("--attention-fixed-split-size", type=int, default=64)
    parser.add_argument("--dtypes", default="float32,float16")
    parser.add_argument("--norm-backends", default="native,fixed_tree")
    parser.add_argument("--linear-backends", default="native")
    parser.add_argument("--linear-tile-m", type=int, default=16)
    parser.add_argument("--linear-tile-n", type=int, default=64)
    parser.add_argument("--linear-k-block-size", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/batch_sensitivity.csv")
    return parser.parse_args()


def compositions(target: str) -> dict[str, list[str]]:
    return {
        "A_target_only": [target],
        "B_one_short": [target, SHORT_DISTRACTORS[0]],
        "C_seven_short": [target, *SHORT_DISTRACTORS],
        "D_one_long": [target, LONG_DISTRACTOR],
        "E_mixed_lengths": [target, SHORT_DISTRACTORS[1], LONG_DISTRACTOR, SHORT_DISTRACTORS[2]],
    }


def encode_batch(
    prompts: list[str],
    tokenizer,
    device: torch.device,
    max_length: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    sequences = [
        tokenizer.encode(prompt, add_bos=True)[:max_length]
        for prompt in prompts
    ]
    lengths = [len(sequence) for sequence in sequences]
    width = max(lengths)
    input_ids = torch.full(
        (len(sequences), width), tokenizer.pad_token_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(sequence, device=device)
        attention_mask[row, : len(sequence)] = 1
    return input_ids, attention_mask, lengths


@torch.inference_mode()
def target_logits(
    model,
    tokenizer,
    prompts: list[str],
    device: torch.device,
    select_logits: bool = False,
) -> torch.Tensor:
    input_ids, attention_mask, lengths = encode_batch(
        prompts, tokenizer, device, model.config.max_position_embeddings
    )
    if select_logits:
        positions = torch.tensor(lengths, device=device) - 1
        logits = model(
            input_ids,
            attention_mask=attention_mask,
            logits_positions=positions,
        )["logits"]
        return logits[0].cpu()
    logits = model(input_ids, attention_mask=attention_mask)["logits"]
    return logits[0, lengths[0] - 1].float().cpu()


@torch.inference_mode()
def batch_greedy_generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int,
    device: torch.device,
    select_logits: bool = False,
) -> list[list[int]]:
    prompt_limit = model.config.max_position_embeddings - max_new_tokens
    if prompt_limit < 1:
        raise ValueError("max_new_tokens must be smaller than max_position_embeddings")
    input_ids, attention_mask, lengths = encode_batch(
        prompts, tokenizer, device, prompt_limit
    )
    generated = [[] for _ in prompts]
    active = torch.ones(len(prompts), dtype=torch.bool, device=device)
    if select_logits:
        positions = torch.tensor(lengths, device=device) - 1
        output = model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_positions=positions,
        )
    else:
        output = model(input_ids, attention_mask=attention_mask, use_cache=True)
    logits = output["logits"]
    past_key_values = output["past_key_values"]
    if select_logits:
        next_tokens = logits.argmax(dim=-1)
    else:
        row_indices = torch.arange(len(prompts), device=device)
        positions = torch.tensor(lengths, device=device) - 1
        next_tokens = logits[row_indices, positions].argmax(dim=-1)

    for step in range(max_new_tokens):
        for row, token in enumerate(next_tokens.tolist()):
            if active[row]:
                generated[row].append(token)
        active &= next_tokens.ne(tokenizer.eos_token_id)
        if not bool(active.any()) or step == max_new_tokens - 1:
            break
        next_tokens = torch.where(
            active, next_tokens, torch.full_like(next_tokens, tokenizer.pad_token_id)
        )
        input_ids = torch.cat((input_ids, next_tokens[:, None]), dim=1)
        attention_mask = torch.cat((attention_mask, active.long()[:, None]), dim=1)
        if input_ids.shape[1] > model.config.max_position_embeddings:
            break
        output = model(
            next_tokens[:, None],
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = output["logits"]
        past_key_values = output["past_key_values"]
        next_tokens = logits[:, -1].argmax(dim=-1)
    return generated


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def precision_context(device: torch.device, dtype: str):
    if dtype == "float32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=getattr(torch, dtype))


def load_prompts(path: str | None, fallback: str) -> list[str]:
    if path:
        return [
            line.strip()
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return DEFAULT_PROMPTS if fallback == "The capital of France is" else [fallback]


def main() -> None:
    args = parse_args()
    set_seed(args.seed, deterministic=True)
    device = resolve_device(args.device)
    rows = []
    details = {}
    for norm_backend in args.norm_backends.split(","):
        for linear_backend in args.linear_backends.split(","):
            for backend in args.backends.split(","):
                model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)
                model.set_batch_invariant_backends(
                    attention_backend=backend,
                    rms_norm_backend=norm_backend,
                    linear_backend=linear_backend,
                    attention_fixed_split_size=args.attention_fixed_split_size,
                    linear_tile_m=args.linear_tile_m,
                    linear_tile_n=args.linear_tile_n,
                    linear_k_block_size=args.linear_k_block_size,
                )
                for dtype in args.dtypes.split(","):
                    if device.type == "cpu" and dtype == "float16":
                        rows.append({
                            "prompt": "", "backend": backend, "dtype": dtype,
                            "norm_backend": norm_backend, "linear_backend": linear_backend,
                            "composition": "",
                            "status": "skipped",
                            "reason": "float16 attention is unsupported on CPU",
                        })
                        continue
                    for target in load_prompts(args.prompts_file, args.target):
                        cases = compositions(target)
                        with precision_context(device, dtype):
                            baseline_logits = target_logits(
                                model, tokenizer, cases["A_target_only"], device
                            )
                            baseline_top5 = torch.topk(baseline_logits, 5).indices.tolist()
                            baseline_generated = batch_greedy_generate(
                                model, tokenizer, cases["A_target_only"],
                                args.max_new_tokens, device
                            )[0]
                            for name, prompts in cases.items():
                                logits = target_logits(model, tokenizer, prompts, device)
                                difference = (logits - baseline_logits).abs()
                                top5 = torch.topk(logits, 5).indices.tolist()
                                generated = batch_greedy_generate(
                                    model, tokenizer, prompts, args.max_new_tokens, device
                                )[0]
                                divergence = first_divergence(baseline_generated, generated)
                                rows.append({
                                    "prompt": target,
                                    "backend": backend,
                                    "attention_fixed_split_size": (
                                        args.attention_fixed_split_size
                                    ),
                                    "dtype": dtype,
                                    "norm_backend": norm_backend,
                                    "linear_backend": linear_backend,
                                    "linear_tile_m": args.linear_tile_m,
                                    "linear_tile_n": args.linear_tile_n,
                                    "linear_k_block_size": args.linear_k_block_size,
                                    "composition": name,
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
                                })
                                key = (
                                    f"{norm_backend}/{linear_backend}/{backend}/"
                                    f"{dtype}/{target}/{name}"
                                )
                                details[key] = {
                                    "prompts": prompts,
                                    "top5_token_ids": top5,
                                    "target_generated_token_ids": generated,
                                    "target_generated_text": tokenizer.decode(generated),
                                }

    save_csv(rows, args.output)
    save_json(
        {
            "checkpoint": str(Path(args.checkpoint)),
            "device": str(device),
            "prompts": load_prompts(args.prompts_file, args.target),
            "cases": details,
        },
        args.output.rsplit(".", 1)[0] + ".json",
    )
    print(f"saved batch sensitivity results to {args.output}")


if __name__ == "__main__":
    main()
