from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from src.common import (
    load_json,
    resolve_device,
    resolve_dtype,
    save_csv,
    save_json,
    set_seed,
    synchronize,
)
from src.data import ByteTokenizer, CausalTextDataset
from src.model import ModelConfig, TinyLlama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small TinyLlama-style causal LM.")
    parser.add_argument("--config", default="configs/train_tiny.json")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output-dir")
    return parser.parse_args()


def autocast_context(device: torch.device, dtype: torch.dtype):
    if dtype == torch.float32:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def infinite_batches(loader: DataLoader):
    while True:
        yield from loader


@torch.no_grad()
def evaluate(
    model: TinyLlama,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int,
) -> float:
    model.eval()
    losses = []
    for batch_index, (input_ids, labels) in enumerate(loader):
        if batch_index >= max_batches:
            break
        input_ids, labels = input_ids.to(device), labels.to(device)
        with autocast_context(device, dtype):
            loss = model(input_ids, labels=labels)["loss"]
        losses.append(float(loss))
    model.train()
    return sum(losses) / max(1, len(losses))


def save_checkpoint(
    model: TinyLlama,
    optimizer: torch.optim.Optimizer,
    step: int,
    output_dir: Path,
) -> None:
    torch.save(
        {
            "model_config": model.config.to_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        output_dir / "checkpoint.pt",
    )


def train(config: dict[str, Any]) -> None:
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(config, output_dir / "resolved_config.json")

    device = resolve_device(config["device"])
    dtype = resolve_dtype(config["dtype"], device)
    set_seed(config["seed"])
    tokenizer = ByteTokenizer()
    model_config = ModelConfig.from_json(config["model_config"])
    train_dataset = CausalTextDataset(config["train_file"], tokenizer, config["seq_len"])
    valid_dataset = CausalTextDataset(config["valid_file"], tokenizer, config["seq_len"])
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        drop_last=False,
        generator=generator,
    )
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"])

    model = TinyLlama(model_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    use_scaler = device.type == "cuda" and dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    metrics: list[dict[str, float | int]] = []
    train_iterator = infinite_batches(train_loader)
    model.train()

    for step in range(1, config["max_steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_tokens = 0
        synchronize(device)
        started = time.perf_counter()
        for _ in range(config["gradient_accumulation_steps"]):
            input_ids, labels = next(train_iterator)
            input_ids, labels = input_ids.to(device), labels.to(device)
            with autocast_context(device, dtype):
                loss = model(input_ids, labels=labels)["loss"]
                scaled_loss = loss / config["gradient_accumulation_steps"]
            scaler.scale(scaled_loss).backward()
            step_loss += float(loss)
            step_tokens += input_ids.numel()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        warmup = min(1.0, step / max(1, config["warmup_steps"]))
        learning_rate = config["learning_rate"] * warmup
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        scaler.step(optimizer)
        scaler.update()
        synchronize(device)
        elapsed = time.perf_counter() - started
        train_loss = step_loss / config["gradient_accumulation_steps"]

        should_evaluate = (
            step % config["eval_interval"] == 0 or step == config["max_steps"]
        )
        should_log = (
            step % config["log_interval"] == 0 or step == 1 or should_evaluate
        )
        if should_log:
            row: dict[str, float | int] = {
                "step": step,
                "train_loss": train_loss,
                "learning_rate": learning_rate,
                "tokens_per_second": step_tokens / elapsed,
                "elapsed_seconds": elapsed,
            }
            if should_evaluate:
                validation_loss = evaluate(
                    model, valid_loader, device, dtype, config["eval_batches"]
                )
                row["validation_loss"] = validation_loss
                row["perplexity"] = math.exp(min(validation_loss, 20.0))
                save_checkpoint(model, optimizer, step, output_dir)
            metrics.append(row)
            save_csv(metrics, output_dir / "training_metrics.csv")
            print(
                f"step={step} loss={train_loss:.4f} "
                f"tokens/s={row['tokens_per_second']:.1f}"
            )

    save_checkpoint(model, optimizer, config["max_steps"], output_dir)
    summary = {
        "device": str(device),
        "dtype": str(dtype),
        "parameter_count": model.parameter_count(),
        "steps": config["max_steps"],
        "final_train_loss": metrics[-1]["train_loss"],
        "final_validation_loss": metrics[-1].get("validation_loss"),
        "final_perplexity": metrics[-1].get("perplexity"),
    }
    save_json(summary, output_dir / "training_summary.json")


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.device is not None:
        config["device"] = args.device
    if args.output_dir is not None:
        config["output_dir"] = args.output_dir
    train(config)


if __name__ == "__main__":
    main()
