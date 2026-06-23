from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from glot_lclm.data.qa_examples import QACollator
from glot_lclm.evaluation.generation import greedy_generate
from glot_lclm.utils.text import best_em_f1


def evaluate_qa(
    model,
    dataset,
    cfg: dict[str, Any],
    mode: str,
    max_examples: int | None = None,
    show_progress: bool = True,
) -> dict[str, Any]:
    model.eval()
    collator = QACollator(
        include_titles=bool(cfg["dataset"].get("include_titles", True)),
        prompt_style=str(cfg["dataset"].get("prompt_style", "default")),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)
    max_new_tokens = int(cfg.get("generation", {}).get("max_new_tokens", 32))

    sums = defaultdict(float)
    count = 0
    examples_out = []
    iterator = tqdm(loader, desc=f"eval/{mode}", disable=not show_progress)
    with torch.no_grad():
        for batch in iterator:
            result = greedy_generate(model, batch, mode=mode, max_new_tokens=max_new_tokens)
            pred = result.texts[0]
            golds = batch[0].answers
            em, f1 = best_em_f1(pred, golds)
            sums["em"] += em
            sums["f1"] += f1
            sums["ttft_ms"] += result.ttft_ms
            sums["peak_memory_mb"] += result.peak_memory_mb
            sums["generated_token_count"] += result.generated_token_count
            count += 1
            if len(examples_out) < 20:
                examples_out.append(
                    {
                        "qid": batch[0].qid,
                        "question": batch[0].question,
                        "prediction": pred,
                        "answers": golds,
                        "em": em,
                        "f1": f1,
                        "ttft_ms": result.ttft_ms,
                        "peak_memory_mb": result.peak_memory_mb,
                    }
                )
            if max_examples is not None and count >= max_examples:
                break

    if count == 0:
        raise ValueError("No evaluation examples")

    metrics = {key: value / count for key, value in sums.items()}
    metrics["n_examples"] = count
    metrics["examples"] = examples_out
    return metrics
