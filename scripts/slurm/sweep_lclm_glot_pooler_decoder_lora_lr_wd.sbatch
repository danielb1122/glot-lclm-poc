#!/bin/bash
#SBATCH --job-name=glot-pooler-lora-lr-wd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
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

CONFIG="${CONFIG:-configs/glot_lclm_squad_r16_pooler_decoder_lora.yaml}"
IFS=" " read -r -a LRS <<< "${LRS:-3e-5 5e-5 8e-5}"
IFS=" " read -r -a WEIGHT_DECAYS <<< "${WEIGHT_DECAYS:-0 0.01 0.05}"

N_LRS="${#LRS[@]}"
N_WDS="${#WEIGHT_DECAYS[@]}"
TOTAL=$((N_LRS * N_WDS))
IDX="${SLURM_ARRAY_TASK_ID}"
if [ "$IDX" -ge "$TOTAL" ]; then
  echo "SLURM_ARRAY_TASK_ID=$IDX is outside grid size $TOTAL" >&2
  exit 1
fi

LR_IDX=$((IDX / N_WDS))
WD_IDX=$((IDX % N_WDS))
LR="${LRS[$LR_IDX]}"
WD="${WEIGHT_DECAYS[$WD_IDX]}"

slugify() {
  local value="$1"
  value="${value//./p}"
  value="${value//e-/em}"
  value="${value//e+/ep}"
  echo "$value"
}

LR_NAME="$(slugify "$LR")"
WD_NAME="$(slugify "$WD")"
RUN_NAME="glot_lclm_squad_r16_pooler_decoder_lora_lr_${LR_NAME}_wd_${WD_NAME}"

python -m glot_lclm.training.train \
  --config "$CONFIG" \
  --eval-max-examples "${EVAL_MAX_EXAMPLES:-100}" \
  training.eval_every_steps="${EVAL_EVERY_STEPS:-50}" \
  training.stages.0.lr="$LR" \
  training.stages.0.weight_decay="$WD" \
  experiment.name="$RUN_NAME" \
  experiment.output_dir="outputs/$RUN_NAME" \
  experiment.tags='["squad","latent_context","glot","pooler_decoder_lora","adapter_frozen","mean_init","lr_wd_sweep","qwen4b","r16"]'
