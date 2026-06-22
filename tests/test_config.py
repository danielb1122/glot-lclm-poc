from glot_lclm.utils.config import apply_overrides


def test_apply_overrides():
    cfg = {"compression": {"ratio": 8}, "experiment": {"name": "x"}}
    out = apply_overrides(cfg, ["compression.ratio=4", "experiment.name=y"])
    assert out["compression"]["ratio"] == 4
    assert out["experiment"]["name"] == "y"
    assert cfg["compression"]["ratio"] == 8


def test_apply_overrides_list_index():
    cfg = {"training": {"stages": [{"steps": 10}, {"steps": 20}]}}
    out = apply_overrides(cfg, ["training.stages.0.steps=1", "training.stages.1.steps=2"])
    assert out["training"]["stages"][0]["steps"] == 1
    assert out["training"]["stages"][1]["steps"] == 2


def test_apply_overrides_encoder_window():
    cfg = {"dataset": {"max_context_tokens": 8192}, "compression": {"encoder_window_tokens": 1024}}
    out = apply_overrides(cfg, ["compression.encoder_window_tokens=512"])
    assert out["dataset"]["max_context_tokens"] == 8192
    assert out["compression"]["encoder_window_tokens"] == 512
