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

For exact GLOT pooling, install PyTorch Geometric. On the BGU cluster setup we have been using (`torch==2.8.0+cu128`), use:

```bash
pip install torch-geometric
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
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

The 16x GLOT configs use `init_as_mean: true`, so the graph pooler weights start as exact mean pooling.

Check repeat split overlap:

```bash
python scripts/check_repeat_split_overlap.py --levels level_5,level_20
```

The repeat configs use `train_split: train` and `eval_split: test`. The exact output strings do not overlap, but the underlying source sentence IDs overlap heavily between synthetic train and test.

For level-20 repeat, run the released mean baseline:

```bash
CONFIG=configs/mean_lclm_repeat_level20_qwen4b_r4.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch
```

Then run mean pooling with only decoder LoRA:

```bash
CONFIG=configs/mean_lclm_repeat_level20_qwen4b_r4_decoder_lora.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch
```

And compare with GLOT plus adapter plus decoder LoRA:

```bash
CONFIG=configs/glot_lclm_repeat_level20_pooler_adapter_decoder_lora.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch
```

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

Run the clean GLOT-pooler ablation with mean-pooling initialization. This freezes the encoder, adapter, and decoder, and trains only the GLOT pooler:

```bash
CONFIG=configs/glot_lclm_squad_r16_pooler_only.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch
```

Run the fairer comparison against mean pooling plus decoder LoRA. This freezes the encoder and adapter, trains GLOT from mean initialization, and trains decoder LoRA:

```bash
CONFIG=configs/glot_lclm_squad_r16_pooler_decoder_lora.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch
```

Then run GLOT with pooler, adapter, and decoder LoRA:

```bash
CONFIG=configs/glot_lclm_squad_r16_pooler_adapter_decoder_lora.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/train_config.sbatch
```

Sweep the pooler-only learning rate. By default this launches five jobs with learning rates `5e-5`, `1e-4`, `2e-4`, `5e-4`, and `1e-3`:

```bash
CONFIG=configs/glot_lclm_squad_r16_pooler_only.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/sweep_lclm_glot_pooler_only_lr.sbatch
```

Sweep the GLOT plus decoder-LoRA learning rate. By default this launches five jobs with learning rates `2e-5`, `5e-5`, `1e-4`, `2e-4`, and `5e-4`:

```bash
CONFIG=configs/glot_lclm_squad_r16_pooler_decoder_lora.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/sweep_lclm_glot_pooler_decoder_lora_lr.sbatch
```

Sweep GLOT plus decoder-LoRA around the best learning-rate region with explicit AdamW weight decay. By default this launches a 3x3 grid: learning rates `3e-5`, `5e-5`, `8e-5` crossed with weight decay values `0`, `0.01`, and `0.05`. It evaluates every 50 steps on 100 validation examples:

```bash
CONFIG=configs/glot_lclm_squad_r16_pooler_decoder_lora.yaml \
EVAL_MAX_EXAMPLES=100 \
EVAL_EVERY_STEPS=50 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/sweep_lclm_glot_pooler_decoder_lora_lr_wd.sbatch
```

Run the longer two-epoch sweep with cosine decay and warmup. By default this launches three jobs with `(lr, weight_decay)` pairs `(2e-5, 0.02)`, `(3e-5, 0.02)`, and `(3e-5, 0.05)`, uses `batch_size=4`, `gradient_accumulation_steps=4`, evaluates every 2750 steps on 1000 validation examples, and saves the best checkpoint by `eval/f1`:

```bash
CONFIG=configs/glot_lclm_squad_r16_pooler_decoder_lora.yaml \
RUN_PREFIX=glot_lclm_squad_r16_pooler_decoder_lora \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/sweep_lclm_two_epoch_lr_wd.sbatch
```

```bash
CONFIG=configs/mean_lclm_squad_qwen4b_r16_decoder_lora.yaml \
RUN_PREFIX=mean_lclm_squad_r16_decoder_lora \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/sweep_lclm_two_epoch_lr_wd.sbatch
```

Sweep the GLOT threshold `tau`. By default this launches four jobs with `tau` values `0.3`, `0.5`, `0.6`, and `0.8`:

```bash
CONFIG=configs/glot_lclm_squad_r16_pooler_decoder_lora.yaml \
EVAL_MAX_EXAMPLES=200 \
sbatch -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/sweep_lclm_glot_pooler_decoder_lora_tau.sbatch
```

To sweep a smaller set:

```bash
CONFIG=configs/glot_lclm_squad_r16_pooler_only.yaml \
LRS="1e-4 2e-4 5e-4" \
EVAL_MAX_EXAMPLES=200 \
sbatch --array=0-2 -p rtx6000 --gres=gpu:rtx_6000:1 scripts/slurm/sweep_lclm_glot_pooler_only_lr.sbatch
```

## Notes

- The GLOT implementation here is block-local. A compression ratio of 8 means each block of 8 encoder token states becomes one latent token.
- Main configs follow the paper's encoder-window setup: `dataset.max_context_tokens` is total context length `T`, `compression.encoder_window_tokens` is encoder window size `W`, and `compression.ratio` is compression ratio `N`.
- The 16x GLOT configs use a GLOT-style threshold graph: edges are rebuilt from hidden-state cosine similarity inside each compression block. The default threshold is `tau: 0.6`.
- The stage flags choose what is trainable: pooler only, pooler plus decoder LoRA, or pooler plus adapter plus decoder LoRA.
- In `*_pooler_only` configs, only the pooler is trainable; the pretrained adapter and decoder are frozen.
- In `*_pooler_decoder_lora` configs, the pooler and decoder LoRA are trainable; the pretrained adapter is frozen.
- `implementation: pyg` uses the PyTorch Geometric GLOT implementation: `Data/Batch`, threshold graph, `GATConv`, `ReLU`, Jumping-Knowledge concat, and learned readout.
- `init_as_mean: true` initializes GLOT itself as mean pooling: the readout scores start uniform and the output projection selects the original token features from the Jumping-Knowledge representation.
- Main experiment configs use the paper-style LCLM adapter: `RMSNorm -> Linear(input_dim, decoder_dim) -> GELU -> Linear(decoder_dim, decoder_dim)`.
- The default config is intentionally small for 24GB GPUs. Increase max context, batch size, LoRA rank, or decoder size only after the smoke runs are stable.
