#!/usr/bin/env bash
set -euo pipefail

python -m src.data.prepare_corpus --config configs/data_tinystories.json
