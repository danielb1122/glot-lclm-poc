#!/bin/bash
#SBATCH --job-name=lclm-2epoch-lr-wd
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
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

CONFIG="${CONFIG:?Set CONFIG, for example CONFIG=configs/glot_lclm_squad_r16_pooler_decoder_lora.yaml}"
RUN_PREFIX="${RUN_PREFIX:-$(basename "$CONFIG" .yaml)}"
IFS=" " read -r -a HPARAMS <<< "${HPARAMS:-2e-5:0.02 3e-5:0.02 3e-5:0.05}"

IDX="${SLURM_ARRAY_TASK_ID}"
if [ "$IDX" -ge "${#HPARAMS[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=$IDX is outside HPARAMS length ${#HPARAMS[@]}" >&2
  exit 1
fi

PAIR="${HPARAMS[$IDX]}"
LR="${PAIR%%:*}"
WD="${PAIR##*:}"

slugify() {
  local value="$1"
  value="${value//./p}"
  value="${value//e-/em}"
  value="${value//e+/ep}"
  echo "$value"
}

LR_NAME="$(slugify "$LR")"
WD_NAME="$(slugify "$WD")"
RUN_NAME="${RUN_PREFIX}_2epoch_lr_${LR_NAME}_wd_${WD_NAME}"

python -m glot_lclm.training.train \
  --config "$CONFIG" \
  --eval-max-examples "${EVAL_MAX_EXAMPLES:-1000}" \
  training.batch_size="${BATCH_SIZE:-4}" \
  training.gradient_accumulation_steps="${GRAD_ACCUM:-4}" \
  training.num_workers="${NUM_WORKERS:-8}" \
  training.eval_every_steps="${EVAL_EVERY_STEPS:-2750}" \
  training.save_every_steps="${SAVE_EVERY_STEPS:-2750}" \
  training.stages.0.steps="${TOTAL_STEPS:-11000}" \
  training.stages.0.lr="$LR" \
  training.stages.0.weight_decay="$WD" \
  training.stages.0.scheduler="${SCHEDULER:-cosine}" \
  training.stages.0.warmup_ratio="${WARMUP_RATIO:-0.04}" \
  experiment.name="$RUN_NAME" \
  experiment.output_dir="outputs/$RUN_NAME"
