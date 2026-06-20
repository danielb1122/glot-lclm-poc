#!/bin/bash
#SBATCH --job-name=glot-lclm-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
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

python -m glot_lclm.training.train \
  --config configs/full_context_qwen05.yaml \
  --eval-max-examples "${EVAL_MAX_EXAMPLES:-50}"

