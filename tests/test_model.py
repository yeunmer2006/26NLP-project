import torch

from src.model import ModelConfig, TinyLlama


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

