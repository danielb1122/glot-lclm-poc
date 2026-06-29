from datasets import Dataset

from glot_lclm.data.qa_examples import select_range


def test_select_range_skips_dev_examples():
    ds = Dataset.from_dict({"id": list(range(10))})

    out = select_range(ds, start_index=3, limit=4)

    assert out["id"] == [3, 4, 5, 6]


def test_select_range_can_take_rest():
    ds = Dataset.from_dict({"id": list(range(5))})

    out = select_range(ds, start_index=2, limit=None)

    assert out["id"] == [2, 3, 4]
