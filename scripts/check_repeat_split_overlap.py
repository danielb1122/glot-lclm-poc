from __future__ import annotations

import argparse

from datasets import load_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="d4nieldev/glotcond-cola-repeat-levels")
    parser.add_argument("--levels", default="level_5,level_20")
    args = parser.parse_args()

    for level in [part.strip() for part in args.levels.split(",") if part.strip()]:
        ds = load_dataset(args.dataset, level)
        train = ds["train"]
        test = ds["test"]

        train_outputs = set(train["expected_output"])
        test_outputs = set(test["expected_output"])
        train_sources = {idx for row in train["source_global_indices"] for idx in row}
        test_sources = {idx for row in test["source_global_indices"] for idx in row}

        output_overlap = len(train_outputs & test_outputs)
        source_overlap = len(train_sources & test_sources)
        source_overlap_fraction = source_overlap / max(len(test_sources), 1)

        print(f"\n== {level} ==")
        print(f"train rows: {len(train)}")
        print(f"test rows: {len(test)}")
        print(f"exact expected_output overlap: {output_overlap}")
        print(f"unique train source sentences: {len(train_sources)}")
        print(f"unique test source sentences: {len(test_sources)}")
        print(f"source sentence overlap: {source_overlap}")
        print(f"test source sentence overlap fraction: {source_overlap_fraction:.4f}")


if __name__ == "__main__":
    main()

