#!/bin/bash
#SBATCH --job-name=glot-lclm-sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-23
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

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

RATIOS=(4 8)
TOPKS=(2 4 7 8)
GNNS=(gat gcn sage)
IDX="${SLURM_ARRAY_TASK_ID}"
RATIO="${RATIOS[$((IDX % ${#RATIOS[@]}))]}"
TOPK="${TOPKS[$(((IDX / ${#RATIOS[@]}) % ${#TOPKS[@]}))]}"
GNN="${GNNS[$(((IDX / (${#RATIOS[@]} * ${#TOPKS[@]})) % ${#GNNS[@]}))]}"

python -m glot_lclm.training.train \
  --config configs/glot_squad_qwen05_8k_r4_sft.yaml \
  compression.ratio="$RATIO" \
  compression.glot.topk="$TOPK" \
  compression.glot.gnn_type="$GNN" \
  experiment.name="glot_squad_qwen05_${GNN}_r${RATIO}_k${TOPK}" \
  experiment.output_dir="outputs/glot_squad_qwen05_${GNN}_r${RATIO}_k${TOPK}" \
  --eval-max-examples "${EVAL_MAX_EXAMPLES:-50}"
