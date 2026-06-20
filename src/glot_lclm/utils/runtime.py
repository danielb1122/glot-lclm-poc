from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def cuda_peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**2)


def cuda_sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def distributed_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_main_process() -> bool:
    return distributed_rank() == 0


def count_trainable_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def detach_to_cpu(metrics: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.detach().float().cpu().item()
        else:
            out[key] = value
    return out

