from glot_lclm.utils.config import apply_overrides


def test_apply_overrides():
    cfg = {"compression": {"ratio": 8}, "experiment": {"name": "x"}}
    out = apply_overrides(cfg, ["compression.ratio=4", "experiment.name=y"])
    assert out["compression"]["ratio"] == 4
    assert out["experiment"]["name"] == "y"
    assert cfg["compression"]["ratio"] == 8

