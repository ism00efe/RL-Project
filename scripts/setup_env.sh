#!/usr/bin/env bash
# Step 1.1: create venv + install deps. Run on Linux / WSL2 with NVIDIA driver installed.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "import jax; print(jax.devices())"
