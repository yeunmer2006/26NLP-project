#!/usr/bin/env bash
set -euo pipefail

python -m src.train.pretrain --config configs/train_tiny.json "$@"

