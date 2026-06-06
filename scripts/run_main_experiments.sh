#!/usr/bin/env bash
set -euo pipefail

checkpoint="${1:-results/train_30m/best_checkpoint.pt}"

python -m src.bench.environment_report
python -m src.infer.generate_prompts --checkpoint "$checkpoint"
python -m src.eval.perplexity --checkpoint "$checkpoint"
python -m src.bench.attention_benchmark \
  --batch-sizes 1,4,8 --seq-lens 128,256,512 \
  --warmup 20 --iterations 100 --repeats 3
python -m src.determinism.batch_sensitivity --checkpoint "$checkpoint"
python -m src.toy.reduction_order --repeats 10
python -m src.toy.batch_invariant_reduction --repeats 10
python -m src.bench.model_scale
python -m src.analysis.plot_results
