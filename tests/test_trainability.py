from torch import nn

from glot_lclm.models.compressor_qa import set_trainability


class DummyCompressedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cfg = {"model": {"encoder_name": "encoder", "decoder_name": "decoder"}}
        self.encoder = nn.Linear(3, 3)
        self.pooler = nn.Linear(3, 3)
        self.adapter = nn.Linear(3, 3)
        self.decoder = nn.Linear(3, 3)


def _trainable_names(model: nn.Module) -> set[str]:
    return {name for name, param in model.named_parameters() if param.requires_grad}


def test_train_decoder_full_trains_decoder_without_lora():
    model = DummyCompressedModel()

    set_trainability(
        model,
        train_pooler=True,
        train_adapter=True,
        train_encoder_lora=False,
        train_decoder_lora=False,
        train_decoder_full=True,
    )

    trainable = _trainable_names(model)
    assert "pooler.weight" in trainable
    assert "adapter.weight" in trainable
    assert "decoder.weight" in trainable
    assert "encoder.weight" not in trainable


def test_decoder_stays_frozen_without_lora_or_full_flag():
    model = DummyCompressedModel()

    set_trainability(
        model,
        train_pooler=True,
        train_adapter=True,
        train_encoder_lora=False,
        train_decoder_lora=False,
        train_decoder_full=False,
    )

    trainable = _trainable_names(model)
    assert "decoder.weight" not in trainable
