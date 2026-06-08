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
    artifacts = []

    def record_figure(path: Path, sources: list[Path]) -> None:
        generated.append(str(path))
        artifacts.append(
            {
                "figure": str(path),
                "sources": [str(source) for source in sources],
            }
        )

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
        record_figure(path, [training_path])

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
        record_figure(path, [attention_path])

    attention_invariance_path = first_existing(
        root, "determinism/attention_invariance.csv", "attention_invariance.csv"
    )
    attention_invariance = [
        row for row in read_csv(attention_invariance_path)
        if row.get("status") == "ok"
    ]
    if attention_invariance:
        grouped = defaultdict(list)
        equality = defaultdict(list)
        for row in attention_invariance:
            grouped[row["backend"]].append(float(row["max_abs_diff"]))
            equality[row["backend"]].append(row.get("bitwise_equal") == "True")
        labels = list(grouped)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        pass_rates = [
            sum(equality[label]) / len(equality[label]) * 100.0
            for label in labels
        ]
        axes[0].bar(labels, pass_rates)
        axes[0].set(
            xlabel="attention backend",
            ylabel="bitwise equal cases (%)",
            title="Attention invariance pass rate",
            ylim=(0, 105),
        )
        max_diffs = [max(grouped[label]) for label in labels]
        plotted_diffs = [max(value, 1e-12) for value in max_diffs]
        axes[1].bar(labels, plotted_diffs)
        axes[1].set_yscale("log")
        axes[1].set(
            xlabel="attention backend",
            ylabel="max absolute difference",
            title="Attention drift, prefill + decode",
        )
        for axis in axes:
            axis.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        path = output / "attention_invariance.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        record_figure(path, [attention_invariance_path])

    rmsnorm_path = first_existing(
        root, "benchmarks/rmsnorm_benchmark.csv", "rmsnorm_benchmark.csv"
    )
    rmsnorm = [
        row for row in read_csv(rmsnorm_path)
        if row.get("status") == "ok" and row.get("batch_size") == "1"
    ]
    if rmsnorm:
        groups = defaultdict(list)
        for row in rmsnorm:
            groups[row["backend"]].append(
                (int(row["seq_len"]), float(row["mean_latency_ms"]))
            )
        fig, axis = plt.subplots(figsize=(6, 4))
        for backend, values in groups.items():
            unique = sorted(set(values))
            axis.plot(
                [x for x, _ in unique],
                [y for _, y in unique],
                marker="o",
                label=backend,
            )
        axis.set(
            xlabel="sequence length",
            ylabel="latency (ms)",
            title="RMSNorm latency, batch=1",
        )
        axis.legend()
        fig.tight_layout()
        path = output / "rmsnorm_latency.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        record_figure(path, [rmsnorm_path])

    matmul_path = first_existing(
        root, "benchmarks/matmul_benchmark.csv", "matmul_benchmark.csv"
    )
    matmul = [
        row for row in read_csv(matmul_path)
        if row.get("status") == "ok" and row.get("batch_size") == "1"
    ]
    if matmul:
        shapes = list(dict.fromkeys(row["shape"] for row in matmul))
        backends = list(dict.fromkeys(row["backend"] for row in matmul))
        fig, axes = plt.subplots(1, len(shapes), figsize=(5 * len(shapes), 4), squeeze=False)
        for axis, shape in zip(axes[0], shapes):
            for backend in backends:
                values = [
                    (int(row["seq_len"]), float(row["mean_latency_ms"]))
                    for row in matmul
                    if row["shape"] == shape and row["backend"] == backend
                ]
                unique = sorted(set(values))
                axis.plot(
                    [x for x, _ in unique],
                    [y for _, y in unique],
                    marker="o",
                    label=backend,
                )
            axis.set(
                xlabel="sequence length",
                ylabel="latency (ms)",
                title=f"Matmul latency, {shape}, batch=1",
            )
            axis.legend()
        fig.tight_layout()
        path = output / "matmul_latency.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        record_figure(path, [matmul_path])

    matmul_invariance_path = first_existing(
        root, "determinism/matmul_invariance.csv", "matmul_invariance.csv"
    )
    matmul_invariance = [
        row for row in read_csv(matmul_invariance_path)
        if row.get("status") == "ok"
    ]
    if matmul_invariance:
        grouped = defaultdict(list)
        equality = defaultdict(list)
        for row in matmul_invariance:
            grouped[row["backend"]].append(float(row["max_abs_diff"]))
            equality[row["backend"]].append(row.get("bitwise_equal") == "True")
        labels = list(grouped)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(
            labels,
            [sum(equality[label]) / len(equality[label]) * 100.0 for label in labels],
        )
        axes[0].set(
            xlabel="matmul backend",
            ylabel="bitwise equal cases (%)",
            title="Matmul invariance pass rate",
            ylim=(0, 105),
        )
        axes[1].bar(labels, [max(max(grouped[label]), 1e-12) for label in labels])
        axes[1].set_yscale("log")
        axes[1].set(
            xlabel="matmul backend",
            ylabel="max absolute difference",
            title="Matmul drift",
        )
        fig.tight_layout()
        path = output / "matmul_invariance.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        record_figure(path, [matmul_invariance_path])

    sensitivity_path = first_existing(
        root, "determinism/batch_sensitivity.csv", "batch_sensitivity.csv"
    )
    sensitivity = [row for row in read_csv(sensitivity_path)
                   if row.get("status") == "ok"]
    if sensitivity:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in sensitivity:
            label = "/".join(
                [
                    row.get("norm_backend", "native"),
                    row.get("linear_backend", "native"),
                ]
            )
            grouped[label].append(float(row["max_abs_diff"]))
        labels = list(grouped)
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.bar(labels, [max(grouped[label]) for label in labels])
        axis.set(
            xlabel="RMSNorm backend",
            ylabel="max absolute logits difference",
            title="Batch sensitivity",
        )
        fig.tight_layout()
        path = output / "batch_sensitivity.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        record_figure(path, [sensitivity_path])

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
        record_figure(path, [reductions_path])

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
        record_figure(path, [invariant_path])

    save_json(
        {
            "generated": generated,
            "artifacts": artifacts,
        },
        output / "manifest.json",
    )
    print(f"generated {len(generated)} figures in {output}")


if __name__ == "__main__":
    main()
