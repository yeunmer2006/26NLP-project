from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


def fixed_tree_sum_last_dim(values: torch.Tensor) -> torch.Tensor:
    width = values.shape[-1]
    padded_width = 1 << (width - 1).bit_length()
    if padded_width != width:
        values = F.pad(values, (0, padded_width - width))
    while values.shape[-1] > 1:
        values = values[..., 0::2] + values[..., 1::2]
    return values


def fixed_tree_max_last_dim(values: torch.Tensor) -> torch.Tensor:
    width = values.shape[-1]
    padded_width = 1 << (width - 1).bit_length()
    if padded_width != width:
        values = F.pad(values, (0, padded_width - width), value=-torch.inf)
    while values.shape[-1] > 1:
        values = torch.maximum(values[..., 0::2], values[..., 1::2])
    return values


def fixed_tile_matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    tile_m: int = 16,
    tile_n: int = 64,
    k_block_size: int = 64,
) -> torch.Tensor:
    """Linear matmul with fixed output tiles and fixed K-block traversal.

    This is a PyTorch reference for the data-parallel matmul strategy described
    in batch-invariant inference work: each output tile owns its K reduction,
    K blocks are visited in a fixed order, and no Split-K/Stream-K style partial
    reductions are introduced. It is intentionally not a fused tensor-core GEMM.
    """
    if input.shape[-1] != weight.shape[-1]:
        raise ValueError("input features must match weight features")
    if tile_m <= 0 or tile_n <= 0 or k_block_size <= 0:
        raise ValueError("tile_m, tile_n, and k_block_size must be positive")

    input_dtype = input.dtype
    flat_input = input.reshape(-1, input.shape[-1]).float()
    flat_weight = weight.float()
    output = torch.empty(
        flat_input.shape[0],
        weight.shape[0],
        dtype=torch.float32,
        device=input.device,
    )
    for m_start in range(0, flat_input.shape[0], tile_m):
        m_end = min(m_start + tile_m, flat_input.shape[0])
        input_tile = flat_input[m_start:m_end]
        for n_start in range(0, weight.shape[0], tile_n):
            n_end = min(n_start + tile_n, weight.shape[0])
            accumulator = torch.zeros(
                m_end - m_start,
                n_end - n_start,
                dtype=torch.float32,
                device=input.device,
            )
            for k_start in range(0, weight.shape[1], k_block_size):
                k_end = min(k_start + k_block_size, weight.shape[1])
                products = (
                    input_tile[:, None, k_start:k_end]
                    * flat_weight[n_start:n_end, k_start:k_end][None, :, :]
                )
                accumulator = accumulator + fixed_tree_sum_last_dim(products).squeeze(-1)
            output[m_start:m_end, n_start:n_end] = accumulator
    if bias is not None:
        output = output + bias.float()
    output = output.reshape(*input.shape[:-1], weight.shape[0])
    return output.to(input_dtype)


class BatchInvariantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        backend: str = "native",
        tile_m: int = 16,
        tile_n: int = 64,
        k_block_size: int = 64,
    ) -> None:
        super().__init__()
        if backend == "fixed_tree":
            backend = "fixed_tile"
        if backend not in {"native", "fixed_tile"}:
            raise ValueError("backend must be 'native' or 'fixed_tile'")
        if tile_m <= 0 or tile_n <= 0 or k_block_size <= 0:
            raise ValueError("tile_m, tile_n, and k_block_size must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.backend = backend
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.k_block_size = k_block_size
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.backend == "fixed_tile":
            return fixed_tile_matmul(
                input,
                self.weight,
                bias=self.bias,
                tile_m=self.tile_m,
                tile_n=self.tile_n,
                k_block_size=self.k_block_size,
            )
        return F.linear(input, self.weight, self.bias)


def flash_attention_2_batch_invariant(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    past_len: int = 0,
    block_size: int = 64,
    causal: bool = True,
) -> torch.Tensor:
    """FlashAttention-2-style online attention with fixed per-row reductions.

    This reference path is intentionally slow: it fixes the traversal order over
    batch, heads, query positions, key blocks, and reduction trees so the result
    for one sequence does not depend on the other sequences in the batch.
    """
    if key.shape[1] != query.shape[1] or value.shape[1] != query.shape[1]:
        raise ValueError("key/value heads must already match query heads")
    if key.shape != value.shape:
        raise ValueError("key and value must have the same shape")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    batch_size, num_heads, query_len, head_dim = query.shape
    key_len = key.shape[-2]
    output = torch.zeros_like(query)
    scale = head_dim ** -0.5
    key_positions = torch.arange(key_len, device=query.device)
    query_positions = torch.arange(
        past_len, past_len + query_len, device=query.device
    )

    for batch_index in range(batch_size):
        valid_keys = torch.ones(key_len, dtype=torch.bool, device=query.device)
        if attention_mask is not None:
            valid_keys = attention_mask[batch_index].bool()
        allowed = valid_keys[None, :].expand(query_len, -1)
        if causal:
            allowed = allowed & (key_positions[None, :] <= query_positions[:, None])

        query_sample = query[batch_index].float()
        running_max = torch.full(
            (num_heads, query_len, 1),
            -torch.inf,
            dtype=torch.float32,
            device=query.device,
        )
        running_sum = torch.zeros_like(running_max)
        accumulator = torch.zeros(
            (num_heads, query_len, head_dim),
            dtype=torch.float32,
            device=query.device,
        )

        for block_start in range(0, key_len, block_size):
            block_end = min(block_start + block_size, key_len)
            block_allowed = allowed[:, block_start:block_end]
            key_block = key[batch_index, :, block_start:block_end].float()
            value_block = value[batch_index, :, block_start:block_end].float()
            products = query_sample[:, :, None, :] * key_block[:, None, :, :]
            scores = fixed_tree_sum_last_dim(products).squeeze(-1) * scale
            scores = scores.masked_fill(~block_allowed[None, :, :], -torch.inf)

            block_max = fixed_tree_max_last_dim(scores)
            new_max = torch.maximum(running_max, block_max)
            old_scale = torch.where(
                torch.isfinite(running_max),
                torch.exp(running_max - new_max),
                torch.zeros_like(running_max),
            )
            probabilities = torch.where(
                block_allowed[None, :, :],
                torch.exp(scores - new_max),
                torch.zeros_like(scores),
            )
            block_sum = fixed_tree_sum_last_dim(probabilities)
            weighted_values = probabilities[..., None] * value_block[:, None, :, :]
            block_accumulator = fixed_tree_sum_last_dim(
                weighted_values.transpose(-1, -2)
            ).squeeze(-1)
            accumulator = accumulator * old_scale + block_accumulator
            running_sum = running_sum * old_scale + block_sum
            running_max = new_max

        normalized = torch.zeros_like(accumulator)
        valid_queries = running_sum.squeeze(-1) > 0
        normalized[valid_queries] = (
            accumulator[valid_queries] / running_sum.expand_as(accumulator)[valid_queries]
        )
        output[batch_index] = normalized.to(query.dtype)
    return output


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float, backend: str = "native") -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        self.backend = backend

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        squared = hidden_states.float().pow(2)
        if self.backend == "fixed_tree":
            variance = fixed_tree_sum_last_dim(squared) / hidden_states.shape[-1]
        else:
            variance = squared.mean(dim=-1, keepdim=True)
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
        self.fixed_split_size = config.attention_fixed_split_size

        self.q_proj = BatchInvariantLinear(
            config.hidden_size,
            self.num_heads * self.head_dim,
            bias=False,
            backend=config.linear_backend,
            tile_m=config.linear_tile_m,
            tile_n=config.linear_tile_n,
            k_block_size=config.linear_k_block_size,
        )
        self.k_proj = BatchInvariantLinear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            backend=config.linear_backend,
            tile_m=config.linear_tile_m,
            tile_n=config.linear_tile_n,
            k_block_size=config.linear_k_block_size,
        )
        self.v_proj = BatchInvariantLinear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=False,
            backend=config.linear_backend,
            tile_m=config.linear_tile_m,
            tile_n=config.linear_tile_n,
            k_block_size=config.linear_k_block_size,
        )
        self.o_proj = BatchInvariantLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            backend=config.linear_backend,
            tile_m=config.linear_tile_m,
            tile_n=config.linear_tile_n,
            k_block_size=config.linear_k_block_size,
        )
        self.rope = RotaryEmbedding(
            self.head_dim, config.max_position_embeddings, config.rope_theta
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.shape
        past_len = 0 if past_key_value is None else past_key_value[0].shape[-2]
        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + seq_len, device=hidden_states.device
            )[None, :].expand(batch_size, -1)
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
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=-2)
            value = torch.cat((past_key_value[1], value), dim=-2)

        if self.backend == "sdpa":
            output = self._sdpa(query, key, value, attention_mask, past_len)
        elif self.backend == "flash_attn_2_bi":
            if self.training and self.dropout:
                raise ValueError("flash_attn_2_bi does not support attention dropout")
            output = flash_attention_2_batch_invariant(
                query,
                key,
                value,
                attention_mask=attention_mask,
                past_len=past_len,
                block_size=self.fixed_split_size,
            )
        else:
            output = self._eager(query, key, value, attention_mask, past_len)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        output = self.o_proj(output)
        if use_cache:
            return output, (key, value)
        return output

    def _sdpa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_len: int,
    ) -> torch.Tensor:
        dropout_p = self.dropout if self.training else 0.0
        if past_len == 0 and (attention_mask is None or bool(attention_mask.all())):
            return F.scaled_dot_product_attention(
                query, key, value, dropout_p=dropout_p, is_causal=True
            )
        query_len, key_len = query.shape[-2], key.shape[-2]
        query_positions = torch.arange(
            past_len, past_len + query_len, device=query.device
        )
        key_positions = torch.arange(key_len, device=query.device)
        allowed = key_positions[None, :] <= query_positions[:, None]
        allowed = allowed[None, None, :, :]
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, :].bool()
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=allowed, dropout_p=dropout_p
        )

    def _eager(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_len: int,
    ) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        query_len, key_len = query.shape[-2], key.shape[-2]
        query_positions = torch.arange(
            past_len, past_len + query_len, device=query.device
        )
        key_positions = torch.arange(key_len, device=query.device)
        allowed = (key_positions[None, :] <= query_positions[:, None])[None, None, :, :]
        if attention_mask is not None:
            allowed = allowed & attention_mask[:, None, None, :].bool()
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probabilities = F.softmax(scores.float(), dim=-1).to(query.dtype)
        probabilities = F.dropout(probabilities, p=self.dropout, training=self.training)
        return torch.matmul(probabilities, value)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = BatchInvariantLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            backend=config.linear_backend,
            tile_m=config.linear_tile_m,
            tile_n=config.linear_tile_n,
            k_block_size=config.linear_k_block_size,
        )
        self.up_proj = BatchInvariantLinear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
            backend=config.linear_backend,
            tile_m=config.linear_tile_m,
            tile_n=config.linear_tile_n,
            k_block_size=config.linear_k_block_size,
        )
        self.down_proj = BatchInvariantLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            backend=config.linear_backend,
            tile_m=config.linear_tile_m,
            tile_n=config.linear_tile_n,
            k_block_size=config.linear_k_block_size,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.input_norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.rms_norm_backend
        )
        self.attention = CausalSelfAttention(config)
        self.post_attention_norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.rms_norm_backend
        )
        self.mlp = SwiGLU(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attention_output = self.attention(
            self.input_norm(hidden_states),
            attention_mask,
            position_ids,
            past_key_value,
            use_cache,
        )
        present = None
        if use_cache:
            attention_output, present = attention_output
        hidden_states = hidden_states + attention_output
        hidden_states = hidden_states + self.mlp(self.post_attention_norm(hidden_states))
        if use_cache:
            return hidden_states, present
        return hidden_states


class TinyLlama(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(
            config.hidden_size, config.rms_norm_eps, config.rms_norm_backend
        )
        self.lm_head = BatchInvariantLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            backend=config.linear_backend,
            tile_m=config.linear_tile_m,
            tile_n=config.linear_tile_n,
            k_block_size=config.linear_k_block_size,
        )
        self.apply(self._initialize_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, (BatchInvariantLinear, nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def set_batch_invariant_backends(
        self,
        *,
        attention_backend: str | None = None,
        rms_norm_backend: str | None = None,
        linear_backend: str | None = None,
        attention_fixed_split_size: int | None = None,
        linear_tile_m: int | None = None,
        linear_tile_n: int | None = None,
        linear_k_block_size: int | None = None,
    ) -> None:
        if attention_backend is not None:
            self.config.attention_backend = attention_backend
        if rms_norm_backend is not None:
            self.config.rms_norm_backend = rms_norm_backend
        if linear_backend is not None:
            self.config.linear_backend = (
                "fixed_tile" if linear_backend == "fixed_tree" else linear_backend
            )
        if attention_fixed_split_size is not None:
            self.config.attention_fixed_split_size = attention_fixed_split_size
        if linear_tile_m is not None:
            self.config.linear_tile_m = linear_tile_m
        if linear_tile_n is not None:
            self.config.linear_tile_n = linear_tile_n
        if linear_k_block_size is not None:
            self.config.linear_k_block_size = linear_k_block_size
        for module in self.modules():
            if isinstance(module, CausalSelfAttention):
                if attention_backend is not None:
                    module.backend = attention_backend
                if attention_fixed_split_size is not None:
                    module.fixed_split_size = attention_fixed_split_size
            elif isinstance(module, RMSNorm):
                if rms_norm_backend is not None:
                    module.backend = rms_norm_backend
            elif isinstance(module, BatchInvariantLinear):
                if linear_backend is not None:
                    module.backend = (
                        "fixed_tile" if linear_backend == "fixed_tree" else linear_backend
                    )
                if linear_tile_m is not None:
                    module.tile_m = linear_tile_m
                if linear_tile_n is not None:
                    module.tile_n = linear_tile_n
                if linear_k_block_size is not None:
                    module.k_block_size = linear_k_block_size

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        past_len = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
        if input_ids.shape[1] + past_len > self.config.max_position_embeddings:
            raise ValueError("sequence length exceeds max_position_embeddings")
        position_ids = None
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids = position_ids.clamp_min(0)
            position_ids = position_ids[:, -input_ids.shape[1]:]
        hidden_states = self.embed_tokens(input_ids)
        presents = []
        for layer_index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[layer_index]
            layer_output = layer(
                hidden_states, attention_mask, position_ids, past, use_cache
            )
            if use_cache:
                hidden_states, present = layer_output
                presents.append(present)
            else:
                hidden_states = layer_output
        logits = self.lm_head(self.norm(hidden_states))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": presents if use_cache else None,
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
