#!/usr/bin/env bash
set -euo pipefail

conda env create -f environment.yml
printf 'Run: conda activate nlp-project\n'
