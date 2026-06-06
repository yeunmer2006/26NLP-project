#!/usr/bin/env bash
set -euo pipefail

python -m src.infer.generate \
  --checkpoint results/smoke_run/checkpoint.pt \
  --prompt "Language models" \
  "$@"

