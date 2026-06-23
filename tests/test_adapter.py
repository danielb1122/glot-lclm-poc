import torch
from torch import nn
from safetensors.torch import save_file

from glot_lclm.models.adapter import PaperMLPAdapter, build_adapter, load_pretrained_adapter


def test_paper_mlp_adapter_shape_and_layers():
    adapter = PaperMLPAdapter(input_dim=12, output_dim=8)
    x = torch.randn(2, 3, 12)
    y = adapter(x)
    assert y.shape == (2, 3, 8)
    assert isinstance(adapter.net[0], nn.RMSNorm)
    assert isinstance(adapter.net[1], nn.Linear)
    assert adapter.net[1].in_features == 12
    assert adapter.net[1].out_features == 8
    assert isinstance(adapter.net[3], nn.Linear)
    assert adapter.net[3].in_features == 8
    assert adapter.net[3].out_features == 8


def test_build_paper_mlp_aliases():
    assert isinstance(build_adapter(12, 8, {"adapter": "paper_mlp"}), PaperMLPAdapter)
    assert isinstance(build_adapter(12, 8, {"adapter": "lclm_mlp"}), PaperMLPAdapter)


def test_load_lclm_adapter_checkpoint(tmp_path):
    path = tmp_path / "adapter.safetensors"
    state = {
        "norm.weight": torch.ones(12),
        "fc1.weight": torch.randn(8, 12),
        "fc1.bias": torch.randn(8),
        "fc2.weight": torch.randn(8, 8),
        "fc2.bias": torch.randn(8),
    }
    save_file(state, str(path))
    adapter = PaperMLPAdapter(input_dim=12, output_dim=8)

    load_pretrained_adapter(adapter, {"path": str(path), "format": "lclm", "strict": True})

    assert torch.allclose(adapter.net[0].weight, state["norm.weight"])
    assert torch.allclose(adapter.net[1].weight, state["fc1.weight"])
    assert torch.allclose(adapter.net[3].bias, state["fc2.bias"])
