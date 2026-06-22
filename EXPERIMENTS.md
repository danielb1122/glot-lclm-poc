# Experiment Protocol

## Main Question

Can GLOT-style learned token graph pooling preserve task-relevant information better than mean pooling at the same compression ratio and similar inference cost?

## Main Task

Default: MuSiQue answerable multi-hop QA.

Input:

```text
context paragraphs + question
```

Output:

```text
short answer
```

Metrics:

- exact match
- answer F1
- TTFT
- peak GPU memory
- compression ratio
- encoder window size `W`

## Runs

### Full Context

Purpose: uncompressed task reference.

```bash
python -m glot_lclm.training.train --config configs/full_context_qwen05.yaml
```

### Truncation

Purpose: cheap short-context baseline.

```bash
python -m glot_lclm.evaluation.evaluate_full_context --config configs/truncation_qwen05.yaml
```

### Mean Pooling LCLM

Purpose: closest naive-pooling baseline.

```bash
python -m glot_lclm.training.train --config configs/mean_musique_qwen05.yaml
```

### Learned Attention Pooling

Purpose: separates learned weighting from graph structure.

```bash
python -m glot_lclm.training.train --config configs/attention_musique_qwen05.yaml
```

### GLOT Pooling LCLM

Purpose: proposed method.

```bash
python -m glot_lclm.training.train --config configs/glot_musique_qwen05.yaml
```

### KVPress/SnapKV

Purpose: inference-time KV compression baseline.

```bash
python -m glot_lclm.evaluation.evaluate_kvpress --config configs/kvpress_snapkv_qwen05.yaml
```

## Fairness Rules

- Compare mean pooling and GLOT pooling with identical encoder, decoder, LoRA ranks, train examples, train steps, compression ratio, and evaluation split.
- Keep `dataset.max_context_tokens` fixed when sweeping `compression.encoder_window_tokens`, so the run changes encoder granularity rather than total available context.
- Report all runs, including failed or weak compression ratios.
- Use the full-context run as a reference, not as a direct efficiency competitor.
- Use truncation to show the task is not solved by simply dropping most context.
- Use KVPress/SnapKV to compare against training-free cache compression.

## Suggested First Table

| Method | Compression | EM | F1 | TTFT ms | Peak GB |
|---|---:|---:|---:|---:|---:|
| Full context | 1x | | | | |
| Truncation 1k | varies | | | | |
| Mean pooling | 4x | | | | |
| Attention pooling | 4x | | | | |
| GLOT pooling | 4x | | | | |
| Mean pooling | 8x | | | | |
| Attention pooling | 8x | | | | |
| GLOT pooling | 8x | | | | |
| SnapKV/KVPress | 8x keep | | | | |

## Strong Claim Threshold

A meaningful positive result is:

```text
GLOT F1 > mean-pooling F1 at the same compression ratio
```

with TTFT and memory close enough that the graph pooler cost is justified.
