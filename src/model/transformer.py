from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states.float() * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_positions: int, theta: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_positions, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)
        self.register_buffer("cos", frequencies.cos(), persistent=False)
        self.register_buffer("sin", frequencies.sin(), persistent=False)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos[position_ids].to(dtype=query.dtype)[:, None, :, :]
        sin = self.sin[position_ids].to(dtype=query.dtype)[:, None, :, :]
        return apply_rope(query, cos, sin), apply_rope(key, cos, sin)


def apply_rope(hidden_states: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    first, second = hidden_states.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return hidden_states * torch.cat((cos, cos), dim=-1) + rotated * torch.cat(
        (sin, sin), dim=-1
    )


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    return hidden_states.repeat_interleave(repeats, dim=1)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.kv_repeats = self.num_heads // self.num_kv_heads
        self.backend = config.attention_backend
        self.dropout = config.attention_dropout

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rope = RotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=hidden_states.device)[None, :].expand(
                batch_size, -1
            )
        query = self.q_proj(hidden_states).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch_size, seq_len, self.num_kv_heads, self.head_dim
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self.rope(query, key, position_ids)
        key = repeat_kv(key, self.kv_repeats)
        value = repeat_kv(value, self.kv_repeats)

        if self.backend == "sdpa":
            output = self._sdpa(query, key, value, attention_mask)
        else:
            output = self._eager(query, key, value, attention_mask)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output)

    def _sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        dropout_p = self.dropout if self.training else 0.0
        if attention_mask is None or bool(attention_mask.all()):
            return F.scaled_dot_product_attention(
                query, key, value, dropout_p=dropout_p, is_causal=True
            )
        seq_len = query.shape[-2]
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device).tril()
        allowed = causal[None, None, :, :] & attention_mask[:, None, None, :].bool()
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=allowed, dropout_p=dropout_p
        )

    def _eager(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        seq_len = query.shape[-2]
        allowed = torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device).tril()
        allowed = allowed[None, None, :, :]
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores.float(), dim=-1).to(query.dtype)
        probabilities = F.dropout(probabilities, p=self.dropout, training=self.training)
        return torch.matmul(probabilities, value)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = CausalSelfAttention(config)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = SwiGLU(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            self.input_norm(hidden_states), attention_mask, position_ids
        )
        return hidden_states + self.mlp(self.post_attention_norm(hidden_states))


class TinyLlama(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._initialize_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | None]:
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")
        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids = position_ids.clamp_min(0)
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, position_ids)
        logits = self.lm_head(self.norm(hidden_states))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return {"loss": loss, "logits": logits}

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
