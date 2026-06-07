import pytest
import torch

from src.bench.attention_benchmark import (
    bottom_right_causal_mask,
    eager_attention,
    sdpa_attention,
    validate_head_counts,
)


def test_bottom_right_causal_mask_matches_flash_attention_2_semantics() -> None:
    expected = torch.tensor(
        [
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )
    assert torch.equal(bottom_right_causal_mask(2, 5), expected)


def test_bottom_right_mask_zeroes_leading_rows_when_query_is_longer() -> None:
    expected = torch.tensor(
        [
            [False, False],
            [False, False],
            [False, False],
            [True, False],
            [True, True],
        ]
    )
    assert torch.equal(bottom_right_causal_mask(5, 2), expected)


def test_decode_causal_mask_can_attend_to_the_full_cache() -> None:
    assert bottom_right_causal_mask(1, 5).all()


def test_gqa_eager_and_sdpa_match() -> None:
    torch.manual_seed(0)
    query = torch.randn(2, 4, 3, 8)
    key = torch.randn(2, 2, 5, 8)
    value = torch.randn(2, 2, 5, 8)
    eager = eager_attention(query, key, value, causal=True)
    sdpa = sdpa_attention(query, key, value, causal=True)
    torch.testing.assert_close(eager, sdpa, rtol=1e-5, atol=1e-6)


def test_fully_masked_causal_rows_match_sdpa_zero_output() -> None:
    torch.manual_seed(0)
    query = torch.randn(1, 2, 5, 8)
    key = torch.randn(1, 2, 2, 8)
    value = torch.randn(1, 2, 2, 8)
    eager = eager_attention(query, key, value, causal=True)
    sdpa = sdpa_attention(query, key, value, causal=True)
    assert torch.equal(eager[:, :, :3], torch.zeros_like(eager[:, :, :3]))
    torch.testing.assert_close(eager, sdpa, rtol=1e-5, atol=1e-6)


def test_invalid_gqa_head_counts_are_rejected() -> None:
    with pytest.raises(ValueError, match="divisible"):
        validate_head_counts(6, 4)
