import random

import torch

from src.toy.reduction_order import (
    FP16_SAFE_SUM,
    blocked_sum,
    fixed_tree_sum,
    overflow_safe_scale,
    quantized_input_reference,
    sequential_sum,
)


def test_fixed_tree_is_repeatable() -> None:
    values = torch.tensor([1e8, 1.0, -1e8, 3.0], dtype=torch.float32)
    first = fixed_tree_sum(values)
    second = fixed_tree_sum(values)
    assert first.item() == second.item()


def test_overflow_safe_scale_is_one_for_small_inputs() -> None:
    values = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
    assert overflow_safe_scale(values) == 1.0


def test_overflow_safe_scale_bounds_absolute_sum() -> None:
    values = torch.full((4096,), 1e4, dtype=torch.float32)
    scale = overflow_safe_scale(values)
    scaled_absolute_sum = float((values * scale).double().abs().sum())

    assert 0.0 < scale < 1.0
    assert scaled_absolute_sum <= FP16_SAFE_SUM * (1.0 + 1e-6)


def test_scaled_fp16_reductions_are_finite_and_share_scale() -> None:
    generator = torch.Generator().manual_seed(42)
    exponents = torch.randint(-4, 5, (4096,), generator=generator)
    signs = torch.randint(0, 2, (4096,), generator=generator) * 2 - 1
    base = signs.float() * torch.pow(10.0, exponents.float())
    scale = overflow_safe_scale(base)
    values = (base * scale).to(torch.float16)
    permutation = list(range(values.numel()))
    random.Random(42).shuffle(permutation)

    results = [
        sequential_sum(values),
        sequential_sum(values.flip(0)),
        blocked_sum(values, 128),
        sequential_sum(values[permutation]),
        fixed_tree_sum(values),
    ]

    assert scale == overflow_safe_scale(base)
    assert all(torch.isfinite(result).item() for result in results)


def test_rescaled_fp64_sum_matches_unscaled_reference() -> None:
    values = torch.tensor([1e4, -3.0, 2e4, 7.0], dtype=torch.float64)
    scale = overflow_safe_scale(values)
    reference = values.sum()
    restored = (values * scale).sum() / scale

    assert torch.allclose(restored, reference, rtol=1e-12, atol=1e-12)


def test_reference_uses_dtype_quantized_input() -> None:
    original = torch.tensor([1.0001, 2.0001, -0.5001], dtype=torch.float32)
    scale = overflow_safe_scale(original)
    quantized = (original * scale).to(torch.float16)
    reference = quantized_input_reference(quantized, scale)

    assert reference == sum(quantized.tolist()) / scale
    assert reference != float(original.double().sum())
