#!/usr/bin/env bash
set -euo pipefail

python -m src.toy.reduction_order
python -m src.toy.batch_invariant_reduction
