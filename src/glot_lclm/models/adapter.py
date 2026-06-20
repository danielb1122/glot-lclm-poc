from __future__ import annotations

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
    raise ValueError(f"Unknown adapter: {name}")

