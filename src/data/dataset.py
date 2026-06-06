from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenizer import ByteTokenizer


class CausalTextDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: ByteTokenizer, seq_len: int) -> None:
        text = Path(path).read_text(encoding="utf-8")
        token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if len(token_ids) < seq_len + 1:
            repeats = (seq_len + 1 + len(token_ids) - 1) // len(token_ids)
            token_ids = token_ids * repeats
        self.tokens = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return max(1, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            missing = self.seq_len + 1 - len(chunk)
            chunk = torch.cat((chunk, self.tokens[:missing]))
        return chunk[:-1], chunk[1:]


class PackedTokenDataset(Dataset):
    """Fixed-length causal examples backed by a memory-mapped NumPy array."""

    def __init__(self, path: str | Path, seq_len: int) -> None:
        self.tokens = np.load(path, mmap_mode="r")
        if self.tokens.ndim != 1:
            raise ValueError("packed token file must contain a one-dimensional array")
        if len(self.tokens) < seq_len + 1:
            raise ValueError("packed token file is shorter than one training example")
        self.seq_len = seq_len

    def __len__(self) -> int:
        return (len(self.tokens) - 1) // self.seq_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.seq_len
        values = np.asarray(self.tokens[start : start + self.seq_len + 1], dtype=np.int64)
        chunk = torch.from_numpy(values.copy())
        return chunk[:-1], chunk[1:]
