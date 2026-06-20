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

RATIOS=(4 8)
TOPKS=(2 4 7 8)
GNNS=(gat gcn sage)
IDX="${SLURM_ARRAY_TASK_ID}"
RATIO="${RATIOS[$((IDX % ${#RATIOS[@]}))]}"
TOPK="${TOPKS[$(((IDX / ${#RATIOS[@]}) % ${#TOPKS[@]}))]}"
GNN="${GNNS[$(((IDX / (${#RATIOS[@]} * ${#TOPKS[@]})) % ${#GNNS[@]}))]}"

python -m glot_lclm.training.train \
  --config configs/glot_musique_qwen05.yaml \
  compression.ratio="$RATIO" \
  compression.glot.topk="$TOPK" \
  compression.glot.gnn_type="$GNN" \
  experiment.name="glot_musique_qwen05_${GNN}_r${RATIO}_k${TOPK}" \
  experiment.output_dir="outputs/glot_musique_qwen05_${GNN}_r${RATIO}_k${TOPK}" \
  --eval-max-examples "${EVAL_MAX_EXAMPLES:-50}"
