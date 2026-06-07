from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 259
    hidden_size: int = 128
    intermediate_size: int = 352
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 256
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    attention_backend: str = "sdpa"
    rms_norm_backend: str = "native"
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if (self.hidden_size // self.num_attention_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.attention_backend not in {"eager", "sdpa"}:
            raise ValueError("attention_backend must be 'eager' or 'sdpa'")
        if self.rms_norm_backend not in {"native", "fixed_tree"}:
            raise ValueError("rms_norm_backend must be 'native' or 'fixed_tree'")

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def to_dict(self) -> dict:
        return asdict(self)
