import numpy as np
import torch

from src.common import save_csv
from src.data import PackedTokenDataset


def test_packed_dataset_returns_shifted_examples(tmp_path) -> None:
    path = tmp_path / "tokens.npy"
    np.save(path, np.arange(10, dtype=np.uint16))
    dataset = PackedTokenDataset(path, seq_len=4)
    inputs, labels = dataset[0]
    torch.testing.assert_close(inputs, torch.tensor([0, 1, 2, 3]))
    torch.testing.assert_close(labels, torch.tensor([1, 2, 3, 4]))


def test_save_csv_supports_optional_metric_columns(tmp_path) -> None:
    path = tmp_path / "metrics.csv"
    save_csv([{"step": 1, "loss": 2.0}, {"step": 2, "loss": 1.0, "ppl": 2.7}], path)
    text = path.read_text(encoding="utf-8")
    assert "ppl" in text
