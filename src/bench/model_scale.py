from __future__ import annotations

import argparse

import torch

from src.common import resolve_device, save_json
from src.model import ModelConfig, TinyLlama


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile model parameter and one-step memory.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", default="results/model_scale.json")
    args = parser.parse_args()
    device = resolve_device(args.device)
    rows = []
    for name in ("30m", "60m", "100m"):
        config = ModelConfig.from_json(f"configs/model_{name}.json")
        model = TinyLlama(config)
        row = {"model": name, "parameter_count": model.parameter_count()}
        try:
            model = model.to(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            input_ids = torch.randint(
                0, config.vocab_size, (args.batch_size, args.seq_len), device=device
            )
            loss = model(input_ids, labels=input_ids)["loss"]
            loss.backward()
            row.update({
                "status": "ok",
                "peak_memory_mb": (
                    torch.cuda.max_memory_allocated(device) / 1024**2
                    if device.type == "cuda" else 0.0
                ),
            })
        except RuntimeError as error:
            row.update({"status": "error", "reason": str(error)})
        rows.append(row)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    save_json({"device": str(device), "models": rows}, args.output)


if __name__ == "__main__":
    main()
