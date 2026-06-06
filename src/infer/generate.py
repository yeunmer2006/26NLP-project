from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.common import resolve_device, save_json, set_seed
from src.data import load_tokenizer
from src.model import ModelConfig, TinyLlama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Language models")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/generation.json")
    return parser.parse_args()


def load_model(checkpoint_path: str, device: torch.device) -> TinyLlama:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TinyLlama(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def load_model_and_tokenizer(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = TinyLlama(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    tokenizer_path = checkpoint.get("tokenizer_path")
    if tokenizer_path and not Path(tokenizer_path).exists():
        local_tokenizer = Path(checkpoint_path).parent / "tokenizer.model"
        tokenizer_path = str(local_tokenizer) if local_tokenizer.exists() else None
    tokenizer = load_tokenizer(tokenizer_path)
    return model.to(device).eval(), tokenizer


@torch.inference_mode()
def generate(
    model: TinyLlama,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> tuple[str, list[int]]:
    token_ids = tokenizer.encode(prompt, add_bos=True)
    for _ in range(max_new_tokens):
        context = token_ids[-model.config.max_position_embeddings :]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(input_ids)["logits"][0, -1].float()
        if temperature <= 0:
            next_token = int(logits.argmax())
        else:
            logits = logits / temperature
            if top_k > 0:
                threshold = torch.topk(logits, min(top_k, logits.numel())).values[-1]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            next_token = int(torch.multinomial(torch.softmax(logits, dim=-1), 1))
        token_ids.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break
    return tokenizer.decode(token_ids), token_ids


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)
    text, token_ids = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        device,
    )
    result = {
        "checkpoint": str(Path(args.checkpoint)),
        "device": str(device),
        "prompt": args.prompt,
        "generated_text": text,
        "token_ids": token_ids,
        "temperature": args.temperature,
        "top_k": args.top_k,
    }
    save_json(result, args.output)
    print(text)


if __name__ == "__main__":
    main()
