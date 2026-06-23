# GLOT-LCLM PoC

Task-specific latent context compression for long-context QA.

This repo adapts the LCLM flow from *End-to-End Context Compression at Scale* to a single-task, single-GPU setting and replaces mean pooling with GLOT-style token graph pooling:

```text
context tokens -> encoder (+ optional LoRA) -> pooler -> adapter -> decoder LLM (+ optional LoRA) -> answer
```

The default target task is SQuAD QA. The default comparisons are:

- full-context decoder baseline
- truncation baseline
- mean-pooling LCLM baseline
- learned attention-pooling LCLM baseline
- GLOT-style graph-pooling LCLM
- optional KVPress/SnapKV inference baseline

Metrics logged to Weights & Biases:

- answer exact match
- answer token-level F1
- time to first token (TTFT)
- peak GPU memory
- compression ratio
- train/eval loss

## Install

Python 3.10+ is recommended.

```bash
cd /Users/danielboharon/glot-lclm-poc
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e .
```

For 4-bit QLoRA on NVIDIA GPUs:

```bash
pip install bitsandbytes
```

For the optional KV cache baseline:

```bash
pip install kvpress
```

Login once on the machine that will train:

```bash
huggingface-cli login
wandb login
```

## Smoke Test

```bash
pytest -q
```

Cluster preflight before submitting SLURM:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python scripts/check_cluster_env.py
```

Deeper check with a tiny model forward pass:

```bash
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python scripts/check_cluster_env.py --deep --device cuda
```

CPU end-to-end smoke run with tiny random local models:

```bash
python -m glot_lclm.training.train \
  --config configs/smoke_tiny_glot.yaml \
  --device cpu \
  --eval-max-examples 2
```

This checks the data loader, encoder, graph pooler, adapter, decoder, loss, generation, metrics, and checkpoint path. It is not a quality test.

## Train Mean Pooling Baseline

```bash
python -m glot_lclm.training.train \
  --config configs/mean_squad_qwen05_8k_r4_sft.yaml
```

## Train GLOT Pooling

```bash
python -m glot_lclm.training.train \
  --config configs/glot_squad_qwen05_8k_r4_sft.yaml
```

## Train Learned Attention Pooling

```bash
python -m glot_lclm.training.train \
  --config configs/attention_squad_qwen05_8k_r4_sft.yaml
```

## Evaluate A Checkpoint

```bash
python -m glot_lclm.evaluation.evaluate \
  --config configs/glot_squad_qwen05_8k_r4_sft.yaml \
  --checkpoint outputs/glot_squad_qwen05_8k_r4_sft/best.pt \
  --split validation
```

## Full Context Baseline

```bash
python -m glot_lclm.evaluation.evaluate_full_context \
  --config configs/full_context_squad_qwen05.yaml \
  --split validation
```

## KVPress/SnapKV Baseline

```bash
python -m glot_lclm.evaluation.evaluate_kvpress \
  --config configs/kvpress_snapkv_squad_qwen05.yaml \
  --split validation
```

KVPress is evaluated through its Hugging Face pipeline. The logged timing is measured around the full pipeline call; use it as a baseline latency number unless you later replace it with a lower-level first-token hook.

## SLURM

From `/home/bohadan/glot-lclm-poc` on the BGU cluster:

```bash
git pull
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
python scripts/check_cluster_env.py
```

Submit SQuAD jobs on RTX 6000:

```bash
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_full_context.sbatch
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_mean.sbatch
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_glot.sbatch
```

The scripts default to SQuAD. The MuSiQue configs are still in `configs/` only as legacy options.

Monitor:

```bash
squeue -u "$USER"
tail -f "$(ls -t logs/*.out | head -1)"
tail -n 120 "$(ls -t logs/*.err | head -1)"
```

## Latent-Context Repeat Experiments

These use the authors' released checkpoint `latent-context/0.6b-4b-LCLM-4x`:

- frozen encoder from `encoder/`
- decoder from `decoder/`
- pretrained adapter from `adapter/adapter.safetensors`
- LCLM memory markers `<|memory_start|>` and `<|memory_end|>`
- repeat-level synthetic dataset `d4nieldev/glotcond-cola-repeat-levels`, `level_5`

Run the released mean-pooling checkpoint as a baseline:

```bash
CONFIG=configs/mean_lclm_repeat_level5_qwen4b_r4.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch
```

Then run GLOT fine-tuning regimes:

```bash
CONFIG=configs/glot_lclm_repeat_level5_pooler_only.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch

CONFIG=configs/glot_lclm_repeat_level5_pooler_adapter.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch

CONFIG=configs/glot_lclm_repeat_level5_pooler_adapter_decoder_lora.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch
```

The GLOT configs use `residual_mean: true` and `zero_init_output: true`, so the graph pooler starts as exact mean pooling and learns a residual correction.

For SQuAD with the authors' 4x and 8x checkpoints, first run mean baselines:

```bash
CONFIG=configs/mean_lclm_squad_qwen4b_r4.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch

CONFIG=configs/mean_lclm_squad_qwen4b_r8.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch
```

Then run GLOT 4x:

```bash
CONFIG=configs/glot_lclm_squad_r4_pooler_only.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch

CONFIG=configs/glot_lclm_squad_r4_pooler_adapter.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch

CONFIG=configs/glot_lclm_squad_r4_pooler_adapter_decoder_lora.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch
```

And GLOT 8x:

```bash
CONFIG=configs/glot_lclm_squad_r8_pooler_only.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch

CONFIG=configs/glot_lclm_squad_r8_pooler_adapter.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch

CONFIG=configs/glot_lclm_squad_r8_pooler_adapter_decoder_lora.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch
```

For the focused 16x comparison, run mean pooling with only decoder LoRA:

```bash
CONFIG=configs/mean_lclm_squad_qwen4b_r16_decoder_lora.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch
```

Then run GLOT with pooler, adapter, and decoder LoRA:

```bash
CONFIG=configs/glot_lclm_squad_r16_pooler_adapter_decoder_lora.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch
```

## Notes

- The GLOT implementation here is block-local. A compression ratio of 8 means each block of 8 encoder token states becomes one latent token.
- Main configs follow the paper's encoder-window setup: `dataset.max_context_tokens` is total context length `T`, `compression.encoder_window_tokens` is encoder window size `W`, and `compression.ratio` is compression ratio `N`.
- The graph is rebuilt from hidden-state cosine similarity at every forward pass.
- The pooler is trained from the first compressed stage together with the adapter.
- Main experiment configs use the paper-style LCLM adapter: `RMSNorm -> Linear(input_dim, decoder_dim) -> GELU -> Linear(decoder_dim, decoder_dim)`.
- The default config is intentionally small for 24GB GPUs. Increase max context, batch size, LoRA rank, or decoder size only after the smoke runs are stable.
