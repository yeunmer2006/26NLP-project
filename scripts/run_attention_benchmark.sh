#!/usr/bin/env bash
set -euo pipefail

python -m src.bench.attention_benchmark "$@"

