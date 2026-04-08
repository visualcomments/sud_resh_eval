#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip
python -m pip install -U g4f
python -m pip install -U datasets huggingface_hub tqdm

echo "[OK] Dependencies installed"
