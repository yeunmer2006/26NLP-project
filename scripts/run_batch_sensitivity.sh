#!/usr/bin/env bash
set -euo pipefail

python -m src.determinism.batch_sensitivity \
  --checkpoint results/smoke_run/checkpoint.pt \
  "$@"

