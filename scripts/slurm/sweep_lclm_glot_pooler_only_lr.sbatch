#!/bin/bash
#SBATCH --job-name=glot-pooler-lr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="${WANDB_PROJECT:-glot-lclm-poc}"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

if [ -f "$PWD/.venv/bin/activate" ]; then
  source "$PWD/.venv/bin/activate"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate glot-lclm
else
  echo "No .venv found and conda is unavailable; using current python: $(command -v python)" >&2
fi

if ! python -c "import glot_lclm; print('Using glot_lclm from', glot_lclm.__file__)"; then
  echo "glot_lclm is not installed in this environment; installing package now." >&2
  python -m pip install . --user --no-build-isolation
  python -c "import glot_lclm; print('Using glot_lclm from', glot_lclm.__file__)"
fi

CONFIG="${CONFIG:-configs/glot_lclm_squad_r16_pooler_only.yaml}"
IFS=" " read -r -a LRS <<< "${LRS:-5e-5 1e-4 2e-4 5e-4 1e-3}"

IDX="${SLURM_ARRAY_TASK_ID}"
if [ "$IDX" -ge "${#LRS[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=$IDX is outside LRS length ${#LRS[@]}" >&2
  exit 1
fi

LR="${LRS[$IDX]}"
LR_NAME="${LR//./p}"
LR_NAME="${LR_NAME//e-/em}"
LR_NAME="${LR_NAME//e+/ep}"

python -m glot_lclm.training.train \
  --config "$CONFIG" \
  --eval-max-examples "${EVAL_MAX_EXAMPLES:-200}" \
  training.stages.0.lr="$LR" \
  experiment.name="glot_lclm_squad_r16_pooler_only_lr_${LR_NAME}" \
  experiment.output_dir="outputs/glot_lclm_squad_r16_pooler_only_lr_${LR_NAME}" \
  experiment.tags='["squad","latent_context","glot","pooler_only","mean_init","lr_sweep","qwen4b","r16"]'
