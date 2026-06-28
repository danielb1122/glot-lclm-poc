import torch

from glot_lclm.models.poolers import (
    AttentionBlockPooler,
    DenseRepoGraphAttentionLayer,
    GLOTBlockPooler,
    MeanBlockPooler,
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


def test_glot_repo_style_uses_repo_like_gat_layer():
    pooler = GLOTBlockPooler(
        input_dim=12,
        compression_ratio=4,
        hidden_dim=16,
        output_dim=12,
        num_layers=1,
        heads=1,
        graph="threshold",
        tau=0.6,
        jk="cat",
        init_as_mean=True,
        layer_style="repo",
    )

    assert isinstance(pooler.layers[0], DenseRepoGraphAttentionLayer)
    assert not hasattr(pooler.layers[0], "norm")
    assert not hasattr(pooler.layers[0], "residual")


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
