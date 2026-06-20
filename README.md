# GLOT-LCLM PoC

Task-specific latent context compression for long-context QA.

This repo adapts the LCLM flow from *End-to-End Context Compression at Scale* to a single-task, single-GPU setting and replaces mean pooling with GLOT-style token graph pooling:

```text
context tokens -> encoder (+ optional LoRA) -> pooler -> adapter -> decoder LLM (+ optional LoRA) -> answer
```

The default target task is MuSiQue answerable multi-hop QA. The default comparisons are:

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
  --config configs/mean_musique_qwen05.yaml
```

## Train GLOT Pooling

```bash
python -m glot_lclm.training.train \
  --config configs/glot_musique_qwen05.yaml
```

## Train Learned Attention Pooling

```bash
python -m glot_lclm.training.train \
  --config configs/attention_musique_qwen05.yaml
```

## Evaluate A Checkpoint

```bash
python -m glot_lclm.evaluation.evaluate \
  --config configs/glot_musique_qwen05.yaml \
  --checkpoint outputs/glot_musique_qwen05/best.pt \
  --split validation
```

## Full Context Baseline

```bash
python -m glot_lclm.evaluation.evaluate_full_context \
  --config configs/full_context_qwen05.yaml \
  --split validation
```

## KVPress/SnapKV Baseline

```bash
python -m glot_lclm.evaluation.evaluate_kvpress \
  --config configs/kvpress_snapkv_qwen05.yaml \
  --split validation
```

KVPress is evaluated through its Hugging Face pipeline. The logged timing is measured around the full pipeline call; use it as a baseline latency number unless you later replace it with a lower-level first-token hook.

## SLURM

Edit account/partition/module lines in `scripts/slurm/*.sbatch`, then:

```bash
sbatch scripts/slurm/train_mean.sbatch
sbatch scripts/slurm/train_glot.sbatch
sbatch scripts/slurm/sweep_glot.sbatch
```

## Notes

- The GLOT implementation here is block-local. A compression ratio of 8 means each block of 8 encoder token states becomes one latent token.
- The graph is rebuilt from hidden-state cosine similarity at every forward pass.
- The pooler is trained from the first compressed stage together with the adapter.
- The default config is intentionally small for 24GB GPUs. Increase max context, batch size, LoRA rank, or decoder size only after the smoke runs are stable.
