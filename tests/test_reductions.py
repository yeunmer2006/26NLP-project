import torch

from src.toy.reduction_order import fixed_tree_sum


def test_fixed_tree_is_repeatable() -> None:
    values = torch.tensor([1e8, 1.0, -1e8, 3.0], dtype=torch.float32)
    first = fixed_tree_sum(values)
    second = fixed_tree_sum(values)
    assert first.item() == second.item()

