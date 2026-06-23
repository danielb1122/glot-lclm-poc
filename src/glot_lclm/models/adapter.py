from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn


class LinearAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.proj(x)


class MLPAdapter(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_multiplier: int = 2):
        super().__init__()
        hidden_dim = max(input_dim, output_dim) * hidden_multiplier
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class PaperMLPAdapter(nn.Module):
    """LCLM paper adapter: RMSNorm, Linear, GELU, Linear.

    Hidden size is the decoder embedding size, so this maps
    input_dim -> output_dim -> output_dim independently for each latent token.
    """

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.RMSNorm(input_dim),
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_adapter(input_dim: int, output_dim: int, cfg: dict) -> nn.Module:
    name = cfg.get("adapter", "mlp")
    if name == "linear":
        return LinearAdapter(input_dim, output_dim)
    if name == "mlp":
        return MLPAdapter(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_multiplier=int(cfg.get("adapter_hidden_multiplier", 2)),
        )
    if name in {"paper_mlp", "lclm_mlp"}:
        return PaperMLPAdapter(input_dim=input_dim, output_dim=output_dim)
    raise ValueError(f"Unknown adapter: {name}")


def _resolve_checkpoint_path(spec: str | dict[str, Any]) -> str:
    if isinstance(spec, str):
        return spec

    path = spec.get("path")
    if path:
        return str(path)

    repo_id = spec.get("repo_id")
    filename = spec.get("filename")
    if not repo_id or not filename:
        raise ValueError("Adapter checkpoint needs either path or repo_id + filename")

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=spec.get("revision"),
    )


def _map_lclm_adapter_state(state: dict) -> dict:
    if "fc1.weight" not in state:
        return state
    return {
        "net.0.weight": state["norm.weight"],
        "net.1.weight": state["fc1.weight"],
        "net.1.bias": state["fc1.bias"],
        "net.3.weight": state["fc2.weight"],
        "net.3.bias": state["fc2.bias"],
    }


def load_pretrained_adapter(adapter: nn.Module, spec: str | dict[str, Any]):
    path = _resolve_checkpoint_path(spec)
    suffix = Path(path).suffix
    if suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(path, device="cpu")
    else:
        import torch

        state = torch.load(path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]

    if isinstance(spec, dict) and spec.get("format", "auto") in {"auto", "lclm"}:
        state = _map_lclm_adapter_state(state)

    strict = bool(spec.get("strict", True)) if isinstance(spec, dict) else True
    return adapter.load_state_dict(state, strict=strict)
