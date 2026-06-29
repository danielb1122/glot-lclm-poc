from types import SimpleNamespace

import torch
import pytest

from glot_lclm.models.compressor_qa import CompressedQAModel
from glot_lclm.models.poolers import (
    AttentionBlockPooler,
    GLOTBlockPooler,
    MeanBlockPooler,
    PyGGLOTBlockPooler,
    build_pooler,
)


def test_mean_pooler_shapes():
    hidden = torch.randn(2, 9, 16)
    mask = torch.ones(2, 9, dtype=torch.long)
    pooler = MeanBlockPooler(input_dim=16, compression_ratio=4)
    out = pooler(hidden, mask)
    assert out.latents.shape == (2, 3, 16)
    assert out.latent_mask.shape == (2, 3)
    assert out.latent_mask.sum().item() == 6


def test_attention_pooler_shapes_with_padding():
    hidden = torch.randn(1, 7, 8)
    mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]])
    pooler = AttentionBlockPooler(input_dim=8, compression_ratio=4, hidden_dim=16)
    out = pooler(hidden, mask)
    assert out.latents.shape == (1, 2, 8)
    assert out.latent_mask.tolist() == [[1, 1]]


def test_glot_pooler_shapes():
    hidden = torch.randn(2, 8, 12)
    mask = torch.ones(2, 8, dtype=torch.long)
    pooler = GLOTBlockPooler(
        input_dim=12,
        compression_ratio=4,
        hidden_dim=16,
        num_layers=2,
        heads=4,
        graph="topk",
        topk=2,
        jk="cat",
    )
    out = pooler(hidden, mask)
    assert out.latents.shape == (2, 2, 44)
    assert out.latent_mask.shape == (2, 2)
    assert "edge_density" in out.aux
    assert "pool_entropy" in out.aux


def test_glot_residual_mean_starts_as_mean():
    hidden = torch.randn(1, 8, 12)
    mask = torch.ones(1, 8, dtype=torch.long)
    mean = MeanBlockPooler(input_dim=12, compression_ratio=4)(hidden, mask).latents
    pooler = GLOTBlockPooler(
        input_dim=12,
        compression_ratio=4,
        hidden_dim=16,
        output_dim=12,
        num_layers=1,
        heads=4,
        graph="topk",
        topk=2,
        jk="cat",
        residual_mean=True,
        zero_init_output=True,
    )
    out = pooler(hidden, mask)
    assert out.latents.shape == (1, 2, 12)
    assert torch.allclose(out.latents, mean, atol=1e-6)


def test_glot_weight_initialization_expresses_mean_pooling():
    hidden = torch.randn(1, 7, 12)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0]])
    mean = MeanBlockPooler(input_dim=12, compression_ratio=4)(hidden, mask).latents
    pooler = GLOTBlockPooler(
        input_dim=12,
        compression_ratio=4,
        hidden_dim=16,
        output_dim=12,
        num_layers=1,
        heads=4,
        graph="threshold",
        tau=0.6,
        jk="cat",
        residual_mean=False,
        zero_init_output=False,
        init_as_mean=True,
        layer_style="repo",
    )

    out = pooler(hidden, mask)

    assert out.latents.shape == (1, 2, 12)
    assert torch.allclose(out.latents, mean, atol=1e-6)


def test_glot_pooler_bfloat16_forward():
    hidden = torch.randn(1, 8, 12, dtype=torch.bfloat16)
    mask = torch.ones(1, 8, dtype=torch.long)
    pooler = GLOTBlockPooler(
        input_dim=12,
        compression_ratio=4,
        hidden_dim=16,
        output_dim=12,
        num_layers=1,
        heads=4,
        graph="topk",
        topk=2,
        jk="cat",
        init_as_mean=True,
        layer_style="repo",
    ).to(dtype=torch.bfloat16)

    out = pooler(hidden, mask)

    assert out.latents.dtype == torch.bfloat16
    assert out.latents.shape == (1, 2, 12)


def test_pyg_glot_weight_initialization_expresses_mean_pooling():
    pytest.importorskip("torch_geometric")
    pytest.importorskip("torch_scatter")
    hidden = torch.randn(1, 7, 12)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1, 0]])
    mean = MeanBlockPooler(input_dim=12, compression_ratio=4)(hidden, mask).latents
    pooler = PyGGLOTBlockPooler(
        input_dim=12,
        compression_ratio=4,
        hidden_dim=16,
        output_dim=12,
        num_layers=1,
        conv="gat",
        adjacency="threshold",
        tau=0.6,
        jk_mode="cat",
        init_as_mean=True,
    )

    out = pooler(hidden, mask)

    assert out.latents.shape == (1, 2, 12)
    assert torch.allclose(out.latents, mean, atol=1e-6)


def test_build_pooler_selects_exact_pyg_glot():
    pytest.importorskip("torch_geometric")
    pytest.importorskip("torch_scatter")
    pooler = build_pooler(
        12,
        {
            "pooler": "glot",
            "ratio": 4,
            "glot": {
                "implementation": "pyg",
                "hidden_dim": 16,
                "output_dim": 12,
                "num_layers": 1,
                "conv": "gat",
                "graph": "threshold",
                "tau": 0.6,
                "jk": "cat",
                "init_as_mean": True,
            },
        },
    )

    assert isinstance(pooler, PyGGLOTBlockPooler)


def test_runtime_mean_initialization_check_passes_for_mean_initialized_glot():
    model = CompressedQAModel.__new__(CompressedQAModel)
    torch.nn.Module.__init__(model)
    model.cfg = {
        "model": {"dtype": "float32"},
        "compression": {
            "pooler": "glot",
            "ratio": 4,
            "glot": {"init_as_mean": True},
        },
    }
    model.encoder_backbone = SimpleNamespace(hidden_size=12)
    model.pooler = GLOTBlockPooler(
        input_dim=12,
        compression_ratio=4,
        hidden_dim=16,
        output_dim=12,
        num_layers=1,
        heads=4,
        graph="threshold",
        tau=0.6,
        jk="cat",
        init_as_mean=True,
    )

    metrics = model.check_pooler_mean_initialization(device="cpu")

    assert metrics["glot_mean_init_max_abs_diff"] <= metrics["glot_mean_init_tolerance"]
