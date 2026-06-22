#!/bin/bash
#SBATCH --job-name=glot-lclm-kvpress
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

# module load cuda/12.1
# source ~/.bashrc
# conda activate glot-lclm

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-glot-lclm-poc}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

# Activate project environment. Prefer repo-local venv; fall back to conda env.
if [ -f "$PWD/.venv/bin/activate" ]; then
  source "$PWD/.venv/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate glot-lclm
else
  echo "No .venv found and conda is unavailable; using current python: $(command -v python)" >&2
fi

if ! python -c "import glot_lclm; print('Using glot_lclm from', glot_lclm.__file__)"; then
  echo "glot_lclm is not installed in this environment; installing editable package now." >&2
  python -m pip install . --user --no-build-isolation
  python -c "import glot_lclm; print('Using glot_lclm from', glot_lclm.__file__)"
fi

python -m glot_lclm.evaluation.evaluate_kvpress \
  --config configs/kvpress_snapkv_squad_qwen05.yaml \
  --split "${SPLIT:-validation}"
