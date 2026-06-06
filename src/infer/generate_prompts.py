from __future__ import annotations

import argparse

from src.common import resolve_device, save_json, set_seed
from src.infer.generate import generate, load_model_and_tokenizer


PROMPTS = [
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed qualitative samples.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="results/generation_samples.json")
    args = parser.parse_args()
    set_seed(42)
    device = resolve_device(args.device)
    model, tokenizer = load_model_and_tokenizer(args.checkpoint, device)
    rows = []
    for prompt in PROMPTS:
        for mode, temperature, top_k in (("greedy", 0.0, 0), ("sample", 0.8, 50)):
            text, token_ids = generate(
                model, tokenizer, prompt, args.max_new_tokens,
                temperature, top_k, device
            )
            rows.append({
                "prompt": prompt,
                "mode": mode,
                "temperature": temperature,
                "top_k": top_k,
                "generated_text": text,
                "token_ids": token_ids,
            })
    save_json(rows, args.output)


if __name__ == "__main__":
    main()
