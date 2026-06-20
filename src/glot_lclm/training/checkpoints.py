from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from glot_lclm.models.compressor_qa import trainable_state_dict
from glot_lclm.utils.config import save_config


def save_checkpoint(
    model,
    output_dir: str | Path,
    *,
    name: str,
    config: dict[str, Any],
    step: int,
    metrics: dict[str, Any] | None = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.pt"
    torch.save(
        {
            "step": step,
            "model": trainable_state_dict(model),
            "metrics": metrics or {},
        },
        path,
    )
    save_config(config, out_dir / "config.yaml")
    return path


def load_checkpoint(model, path: str | Path) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    ckpt["missing_keys"] = missing
    ckpt["unexpected_keys"] = unexpected
    return ckpt

