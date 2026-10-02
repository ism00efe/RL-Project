#!/usr/bin/env bash
# Step 1.1: create venv + install deps. Run on Linux / WSL2 with NVIDIA driver installed.
# Pins are verified on Python 3.11; Ubuntu 24.04 ships 3.12, so uv provides 3.11.
# System packages: ffmpeg (videos), libegl1/libgl1 (MUJOCO_GL=egl rendering).
set -euo pipefail
cd "$(dirname "$0")/.."
sudo apt-get install -y git curl ffmpeg libgl1 libegl1
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv -p 3.11 .venv
. .venv/bin/activate
uv pip install -r requirements.txt
python -c "import jax; print(jax.devices())"
