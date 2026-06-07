from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from src.common import save_json


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def first_existing(root: Path, *relative_paths: str) -> Path:
    for relative_path in relative_paths:
        path = root / relative_path
        if path.exists():
            return path
    return root / relative_paths[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures from experiment CSVs.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--output-dir", default="results/figures")
    parser.add_argument("--training-metrics")
    args = parser.parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to generate figures") from error

    root = Path(args.results_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generated = []

    training_path = (
        Path(args.training_metrics)
        if args.training_metrics
        else root / "train_30m" / "training_metrics.csv"
    )
    training = read_csv(training_path)
    if training:
        steps = [int(row["step"]) for row in training]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(steps, [float(row["train_loss"]) for row in training], label="train")
        valid = [(int(row["step"]), float(row["validation_loss"])) for row in training
                 if row.get("validation_loss")]
        if valid:
            axes[0].plot([x for x, _ in valid], [y for _, y in valid], label="validation")
        axes[0].set(xlabel="step", ylabel="loss", title="Training loss")
        axes[0].legend()
        axes[1].plot(steps, [float(row["tokens_per_second"]) for row in training])
        axes[1].set(xlabel="step", ylabel="tokens/s", title="Training throughput")
        fig.tight_layout()
        path = output / "training.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(str(path))

    attention_path = first_existing(
        root, "benchmarks/attention_benchmark.csv", "attention_benchmark.csv"
    )
    attention = [row for row in read_csv(attention_path)
                 if row.get("status") == "ok"]
    if attention:
        groups: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for row in attention:
            if row["workload"] == "prefill" and row["batch_size"] == "1":
                groups[row["backend"]].append(
                    (int(row["seq_len"]), float(row["mean_latency_ms"]))
                )
        fig, axis = plt.subplots(figsize=(6, 4))
        for backend, values in groups.items():
            unique = sorted(set(values))
            axis.plot([x for x, _ in unique], [y for _, y in unique], marker="o", label=backend)
        axis.set(xlabel="sequence length", ylabel="latency (ms)",
                 title="Attention prefill latency, batch=1")
        axis.legend()
        fig.tight_layout()
        path = output / "attention_latency.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(str(path))

    sensitivity_path = first_existing(
        root, "determinism/batch_sensitivity.csv", "batch_sensitivity.csv"
    )
    sensitivity = [row for row in read_csv(sensitivity_path)
                   if row.get("status") == "ok"]
    if sensitivity:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in sensitivity:
            grouped[row["composition"]].append(float(row["max_abs_diff"]))
        labels = list(grouped)
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.bar(labels, [max(grouped[label]) for label in labels])
        axis.tick_params(axis="x", rotation=20)
        axis.set(ylabel="max absolute logits difference", title="Batch sensitivity")
        fig.tight_layout()
        path = output / "batch_sensitivity.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(str(path))

    reductions_path = first_existing(
        root, "toy/reduction_order.csv", "reduction_order.csv"
    )
    reductions = [row for row in read_csv(reductions_path)
                  if row.get("status") == "ok" and row.get("repeat") == "1"]
    if reductions:
        labels = [f"{row['dtype']}:{row['method']}" for row in reductions]
        errors = [max(float(row["absolute_error"]), 1e-12) for row in reductions]
        fig, axis = plt.subplots(figsize=(10, 4))
        axis.bar(labels, errors)
        axis.set_yscale("log")
        axis.tick_params(axis="x", rotation=45)
        axis.set(ylabel="absolute error (log)", title="Reduction-order error")
        fig.tight_layout()
        path = output / "reduction_error.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(str(path))

    invariant_path = first_existing(
        root,
        "toy/batch_invariant_reduction.csv",
        "batch_invariant_reduction.csv",
    )
    invariant = read_csv(invariant_path)
    if invariant:
        fig, axis = plt.subplots(figsize=(7, 4))
        for method in ("block_dependent", "fixed_tree"):
            rows = [row for row in invariant if row["method"] == method]
            axis.plot(
                [int(row["block_size"]) for row in rows],
                [float(row["elapsed_ms"]) for row in rows],
                marker="o",
                label=method,
            )
        axis.set(xlabel="block size", ylabel="elapsed (ms)", title="Reduction runtime")
        axis.legend()
        fig.tight_layout()
        path = output / "fixed_tree_runtime.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        generated.append(str(path))

    save_json({"generated": generated}, output / "manifest.json")
    print(f"generated {len(generated)} figures in {output}")


if __name__ == "__main__":
    main()
