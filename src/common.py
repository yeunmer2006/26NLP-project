from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any, Iterable

import torch


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(data: Any, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def save_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> None:
    rows = list(rows)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def set_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]
    if device.type == "cpu" and dtype == torch.float16:
        raise ValueError("float16 training on CPU is unsupported; use float32 or bfloat16")
    return dtype


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
