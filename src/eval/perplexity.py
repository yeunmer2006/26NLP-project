from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from src.common import resolve_device, save_json
from src.data import PackedTokenDataset
from src.infer.generate import load_model_and_tokenizer


def prepare_wikitext(tokenizer, output: Path, token_budget: int) -> None:
    try:
        import numpy as np
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError("datasets and numpy are required for WikiText evaluation") from error
    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    tokens: list[int] = []
    for text in dataset["text"]:
        if text.strip():
            tokens.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
        if len(tokens) >= token_budget:
            break
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.asarray(tokens[:token_budget], dtype=np.uint16))


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate out-of-domain WikiText-2 PPL.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokens", default="data/processed/wikitext2/test.npy")
    parser.add_argument("--token-budget", type=int, default=500_000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/wikitext_perplexity.json")
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)
    token_path = Path(args.tokens)
    if not token_path.exists():
        prepare_wikitext(tokenizer, token_path, args.token_budget)
    loader = torch.utils.data.DataLoader(
        PackedTokenDataset(token_path, args.seq_len), batch_size=args.batch_size
    )
    losses = []
    for index, (input_ids, labels) in enumerate(loader):
        if index >= args.max_batches:
            break
        loss = model(input_ids.to(device), labels=labels.to(device))["loss"]
        losses.append(float(loss))
    mean_loss = sum(losses) / max(1, len(losses))
    save_json(
        {
            "dataset": "Salesforce/wikitext",
            "dataset_config": "wikitext-2-raw-v1",
            "split": "test",
            "checkpoint": args.checkpoint,
            "batches": len(losses),
            "loss": mean_loss,
            "perplexity": math.exp(min(mean_loss, 20.0)),
        },
        args.output,
    )


if __name__ == "__main__":
    main()
