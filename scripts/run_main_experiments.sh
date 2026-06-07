#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:-results/train_30m/best_checkpoint.pt}"
output_dir="${2:-results/main_experiments}"

mkdir -p \
  "$output_dir/generation" \
  "$output_dir/evaluation" \
  "$output_dir/benchmarks" \
  "$output_dir/determinism" \
  "$output_dir/toy" \
  "$output_dir/figures"

python -m src.bench.environment_report \
  --output "$output_dir/environment.json"
python -m src.infer.generate_prompts \
  --checkpoint "$checkpoint" \
  --output "$output_dir/generation/generation_samples.json"
python -m src.eval.perplexity \
  --checkpoint "$checkpoint" \
  --output "$output_dir/evaluation/wikitext_perplexity.json"
python -m src.bench.attention_benchmark \
  --batch-sizes 1,4,8 --seq-lens 128,256,512 \
  --warmup 20 --iterations 100 --repeats 3 \
  --output "$output_dir/benchmarks/attention_benchmark.csv"
python -m src.determinism.batch_sensitivity \
  --checkpoint "$checkpoint" \
  --output "$output_dir/determinism/batch_sensitivity.csv"
python -m src.toy.reduction_order \
  --repeats 10 \
  --output "$output_dir/toy/reduction_order.csv"
python -m src.toy.batch_invariant_reduction \
  --repeats 10 \
  --output "$output_dir/toy/batch_invariant_reduction.csv"
python -m src.bench.model_scale \
  --output "$output_dir/benchmarks/model_scale.json"
python -m src.analysis.plot_results \
  --results-dir "$output_dir" \
  --training-metrics results/train_30m/training_metrics.csv \
  --output-dir "$output_dir/figures"
