import torch

from glot_lclm.models.poolers import AttentionBlockPooler, GLOTBlockPooler, MeanBlockPooler


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
