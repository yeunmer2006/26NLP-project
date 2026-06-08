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
    attention_fixed_split_size: int = 64
    rms_norm_backend: str = "native"
    linear_backend: str = "native"
    linear_tile_m: int = 16
    linear_tile_n: int = 64
    linear_k_block_size: int = 64
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.linear_backend == "fixed_tree":
            self.linear_backend = "fixed_tile"
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if (self.hidden_size // self.num_attention_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        if self.attention_backend not in {"eager", "sdpa", "flash_attn_2_bi"}:
            raise ValueError(
                "attention_backend must be 'eager', 'sdpa', or 'flash_attn_2_bi'"
            )
        if self.attention_fixed_split_size <= 0:
            raise ValueError("attention_fixed_split_size must be positive")
        if self.rms_norm_backend not in {"native", "fixed_tree"}:
            raise ValueError("rms_norm_backend must be 'native' or 'fixed_tree'")
        if self.linear_backend not in {"native", "fixed_tile"}:
            raise ValueError("linear_backend must be 'native' or 'fixed_tile'")
        if self.linear_tile_m <= 0:
            raise ValueError("linear_tile_m must be positive")
        if self.linear_tile_n <= 0:
            raise ValueError("linear_tile_n must be positive")
        if self.linear_k_block_size <= 0:
            raise ValueError("linear_k_block_size must be positive")

    @classmethod
    def from_json(cls, path: str | Path) -> "ModelConfig":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**json.load(handle))

    def to_dict(self) -> dict:
        return asdict(self)
