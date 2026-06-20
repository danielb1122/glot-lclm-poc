#!/bin/bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-glot-lclm}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if command -v conda >/dev/null 2>&1; then
  conda create -n "$ENV_NAME" python=3.10 -y
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
else
  "$PYTHON_BIN" -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"

echo "Environment ready. Next:"
echo "  huggingface-cli login"
echo "  wandb login"
