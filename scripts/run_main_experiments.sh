#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:-results/train_30m/best_checkpoint.pt}"
output_dir="${2:-results/main_experiments_30m_v2}"
python_bin="${PYTHON_BIN:-python}"

mkdir -p \
  "$output_dir/generation" \
  "$output_dir/evaluation" \
  "$output_dir/benchmarks" \
  "$output_dir/determinism" \
  "$output_dir/toy" \
  "$output_dir/figures"

"$python_bin" -m src.bench.environment_report \
  --output "$output_dir/environment.json"
"$python_bin" -m src.infer.generate_prompts \
  --checkpoint "$checkpoint" \
  --output "$output_dir/generation/generation_samples.json"
"$python_bin" -m src.eval.perplexity \
  --checkpoint "$checkpoint" \
  --output "$output_dir/evaluation/wikitext_perplexity.json"
"$python_bin" -m src.bench.attention_benchmark \
  --batch-sizes 1,4,8 --seq-lens 128,256,512 \
  --num-heads 8 --num-kv-heads 4 --head-dim 60 \
  --fixed-split-size 64 --warmup 20 --iterations 100 --repeats 3 \
  --output "$output_dir/benchmarks/attention_benchmark.csv" \
  --invariance-output "$output_dir/determinism/attention_invariance.csv"
"$python_bin" -m src.bench.rmsnorm_benchmark \
  --batch-sizes 1,4,8 --seq-lens 128,256,512 \
  --hidden-size 480 --warmup 20 --iterations 100 --repeats 3 \
  --output "$output_dir/benchmarks/rmsnorm_benchmark.csv" \
  --invariance-output "$output_dir/determinism/rmsnorm_invariance.csv"
"$python_bin" -m src.bench.matmul_benchmark \
  --batch-sizes 1,4 --seq-lens 1,32,128 \
  --shapes hidden:480:480,mlp_up:480:1280,lm_head:480:259 \
  --tile-m 16 --tile-n 64 --k-block-size 64 \
  --warmup 10 --iterations 20 --repeats 2 \
  --output "$output_dir/benchmarks/matmul_benchmark.csv" \
  --invariance-output "$output_dir/determinism/matmul_invariance.csv"
"$python_bin" -m src.determinism.batch_sensitivity \
  --checkpoint "$checkpoint" \
  --backends eager,sdpa,flash_attn_2_bi \
  --attention-fixed-split-size 64 \
  --norm-backends native,fixed_tree \
  --linear-backends native \
  --output "$output_dir/determinism/batch_sensitivity.csv"
"$python_bin" -m src.determinism.batch_sensitivity \
  --checkpoint "$checkpoint" \
  --target "Once upon a time" \
  --backends flash_attn_2_bi \
  --attention-fixed-split-size 64 \
  --norm-backends fixed_tree \
  --linear-backends fixed_tile \
  --linear-tile-m 16 --linear-tile-n 64 --linear-k-block-size 64 \
  --dtypes float32 \
  --max-new-tokens 1 \
  --output "$output_dir/determinism/batch_invariant_model_smoke.csv"
"$python_bin" -m src.determinism.divergence_search \
  --checkpoint "$checkpoint" \
  --documents 2000 --prefix-lengths 8,16,32,64,128 \
  --keep 100 --max-new-tokens 128 --norm-backends native \
  --candidates-output "$output_dir/determinism/prompt_margin_candidates.csv" \
  --output "$output_dir/determinism/divergence_search.csv"
"$python_bin" -m src.toy.reduction_order \
  --repeats 10 \
  --output "$output_dir/toy/reduction_order.csv"
"$python_bin" -m src.toy.batch_invariant_reduction \
  --repeats 10 \
  --output "$output_dir/toy/batch_invariant_reduction.csv"
"$python_bin" -m src.bench.model_scale \
  --output "$output_dir/benchmarks/model_scale.json"
"$python_bin" -m src.analysis.plot_results \
  --results-dir "$output_dir" \
  --training-metrics results/train_30m/training_metrics.csv \
  --output-dir "$output_dir/figures"
