import dataclasses

import torch

from src.model import BatchInvariantLinear, ModelConfig, RMSNorm, TinyLlama, fixed_tile_matmul


def test_model_forward_and_loss() -> None:
    config = ModelConfig(
        vocab_size=259,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    model = TinyLlama(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    output = model(input_ids, labels=input_ids)
    assert output["logits"].shape == (2, 8, config.vocab_size)
    assert output["loss"].ndim == 0


def test_padding_does_not_shift_target_positions() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        attention_backend="eager",
    )
    model = TinyLlama(config).eval()
    target = torch.tensor([[1, 10, 11]])
    single = model(target)["logits"][0, -1]
    batch = torch.tensor([[1, 10, 11, 0, 0], [1, 20, 21, 22, 23]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    batched = model(batch, attention_mask=mask)["logits"][0, 2]
    torch.testing.assert_close(single, batched, rtol=1e-5, atol=1e-6)


def test_flash_attention_2_bi_backend_is_batch_invariant() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        attention_backend="flash_attn_2_bi",
        rms_norm_backend="fixed_tree",
        linear_backend="fixed_tile",
        linear_tile_m=2,
        linear_tile_n=8,
        linear_k_block_size=4,
    )
    model = TinyLlama(config).eval()
    target = torch.tensor([[1, 10, 11]])
    mixed = torch.tensor([[1, 10, 11], [1, 20, 21], [1, 22, 23]])
    single_logits = model(target)["logits"][0, -1]
    mixed_logits = model(mixed)["logits"][0, -1]
    assert torch.equal(single_logits, mixed_logits)


def test_vocab_size_can_be_overridden_from_tokenizer() -> None:
    config = ModelConfig(vocab_size=8000, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=1, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=16)
    model = TinyLlama(config)
    assert model.embed_tokens.num_embeddings == 8000


def test_old_checkpoint_config_defaults_to_native_rmsnorm() -> None:
    fields = {
        field.name: field.default
        for field in dataclasses.fields(ModelConfig)
        if field.name != "rms_norm_backend"
    }
    config = ModelConfig(**fields)
    assert config.rms_norm_backend == "native"
    assert config.linear_backend == "native"


def test_fixed_tree_rmsnorm_is_close_to_native() -> None:
    torch.manual_seed(0)
    values = torch.randn(2, 3, 480, dtype=torch.float32)
    native = RMSNorm(480, 1e-5, "native")
    fixed = RMSNorm(480, 1e-5, "fixed_tree")
    fixed.load_state_dict(native.state_dict())
    torch.testing.assert_close(
        fixed(values), native(values), rtol=1e-5, atol=1e-6
    )


def test_fixed_tree_rmsnorm_is_batch_invariant() -> None:
    torch.manual_seed(0)
    norm = RMSNorm(480, 1e-5, "fixed_tree")
    target = torch.randn(1, 1, 480)
    mixed = torch.cat((target, torch.randn(7, 1, 480)), dim=0)
    assert torch.equal(norm(target)[0, 0], norm(mixed)[0, 0])


def test_fixed_tile_matmul_matches_native() -> None:
    torch.manual_seed(0)
    values = torch.randn(2, 3, 17)
    weight = torch.randn(11, 17)
    candidate = fixed_tile_matmul(
        values,
        weight,
        tile_m=2,
        tile_n=5,
        k_block_size=4,
    )
    reference = torch.nn.functional.linear(values, weight)
    torch.testing.assert_close(candidate, reference, rtol=1e-5, atol=1e-6)


def test_batch_invariant_linear_is_batch_invariant() -> None:
    torch.manual_seed(0)
    layer = BatchInvariantLinear(
        17,
        11,
        backend="fixed_tile",
        tile_m=2,
        tile_n=5,
        k_block_size=4,
    )
    target = torch.randn(1, 1, 17)
    mixed = torch.randn(6, 1, 17)
    mixed[0] = target[0]
    assert torch.equal(layer(target)[0, 0], layer(mixed)[0, 0])


def test_kv_cache_matches_full_forward() -> None:
    torch.manual_seed(0)
    config = ModelConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
        attention_backend="eager",
    )
    model = TinyLlama(config).eval()
    prefix = torch.tensor([[1, 10, 11]])
    next_token = torch.tensor([[12]])
    full = model(torch.cat((prefix, next_token), dim=1))["logits"][:, -1]
    cached = model(prefix, use_cache=True)
    incremental = model(
        next_token,
        past_key_values=cached["past_key_values"],
        use_cache=True,
    )["logits"][:, -1]
    torch.testing.assert_close(incremental, full, rtol=1e-5, atol=1e-6)
