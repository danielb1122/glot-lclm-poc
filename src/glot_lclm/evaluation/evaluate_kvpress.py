from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from glot_lclm.data.qa_examples import QACollator, load_qa_dataset, maybe_limit
from glot_lclm.training.train import _split_or_fallback
from glot_lclm.utils.config import apply_overrides, flatten_dict, load_config
from glot_lclm.utils.runtime import cuda_peak_memory_mb, cuda_sync, ensure_dir, get_dtype, set_seed
from glot_lclm.utils.text import best_em_f1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default=None)
    parser.add_argument("--max-examples", type=int, default=None)
    args, unknown = parser.parse_known_args()
    args.overrides = [x[2:] if x.startswith("--") else x for x in unknown]
    return args


def _load_press(cfg: dict[str, Any]):
    try:
        import kvpress
    except ImportError as exc:
        raise ImportError("Install optional KV baseline dependency with `pip install kvpress`.") from exc

    press_name = cfg["kvpress"].get("press", "SnapKVPress")
    if not hasattr(kvpress, press_name):
        raise AttributeError(f"kvpress has no press named {press_name}")
    press_cls = getattr(kvpress, press_name)
    kwargs = {
        k: v
        for k, v in cfg["kvpress"].items()
        if k not in {"press", "pipeline_task"}
    }
    return press_cls(**kwargs)


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    set_seed(int(cfg["experiment"].get("seed", 42)))
    raw = load_qa_dataset(cfg["dataset"])
    split = args.split or cfg["dataset"]["eval_split"]
    ds = maybe_limit(_split_or_fallback(raw, split), cfg["dataset"].get("eval_limit"))
    if args.max_examples:
        ds = maybe_limit(ds, args.max_examples)

    model_name = cfg["model"]["decoder_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=cfg["model"].get("trust_remote_code", True))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"].get("trust_remote_code", True),
        torch_dtype=get_dtype(cfg["model"].get("dtype")),
        device_map="auto",
    )
    press = _load_press(cfg)
    pipe = pipeline(
        cfg["kvpress"].get("pipeline_task", "kv-press-text-generation"),
        model=model,
        tokenizer=tokenizer,
    )

    collator = QACollator(include_titles=bool(cfg["dataset"].get("include_titles", True)))
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collator)
    max_new_tokens = int(cfg.get("generation", {}).get("max_new_tokens", 32))

    sums = {"em": 0.0, "f1": 0.0, "ttft_ms": 0.0, "peak_memory_mb": 0.0}
    examples_out = []
    count = 0
    for batch in tqdm(loader, desc="eval/kvpress"):
        context = batch[0].context
        question = f"\nQuestion: {batch[0].question}\nAnswer:"
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        cuda_sync()
        start = time.perf_counter()
        outputs = pipe(
            context,
            question=question,
            press=press,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        cuda_sync()
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        if isinstance(outputs, dict) and "answer" in outputs:
            pred = str(outputs["answer"]).strip()
        elif isinstance(outputs, list) and outputs and "generated_text" in outputs[0]:
            pred = str(outputs[0]["generated_text"]).strip()
        else:
            pred = str(outputs).strip()
        em, f1 = best_em_f1(pred, batch[0].answers)
        sums["em"] += em
        sums["f1"] += f1
        sums["ttft_ms"] += elapsed_ms
        sums["peak_memory_mb"] += cuda_peak_memory_mb()
        count += 1
        if len(examples_out) < 20:
            examples_out.append(
                {
                    "qid": batch[0].qid,
                    "question": batch[0].question,
                    "prediction": pred,
                    "answers": batch[0].answers,
                    "em": em,
                    "f1": f1,
                    "ttft_ms": elapsed_ms,
                    "peak_memory_mb": cuda_peak_memory_mb(),
                }
            )

    metrics = {k: v / max(count, 1) for k, v in sums.items()}
    metrics["n_examples"] = count
    metrics["examples"] = examples_out
    out_dir = ensure_dir(Path(cfg["experiment"]["output_dir"]) / "eval")
    out_path = out_dir / f"kvpress_{split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if wandb is not None and cfg["experiment"].get("wandb_mode", "online") != "disabled":
        run = wandb.init(
            project=cfg["experiment"].get("wandb_project"),
            entity=cfg["experiment"].get("wandb_entity"),
            name=f"{cfg['experiment']['name']}-eval-{split}",
            config=flatten_dict(cfg),
            mode=cfg["experiment"].get("wandb_mode", "online"),
            tags=cfg["experiment"].get("tags", []) + ["eval"],
        )
        log_metrics = {f"eval/{k}": v for k, v in metrics.items() if k != "examples"}
        if metrics.get("examples"):
            table = wandb.Table(
                columns=["qid", "question", "prediction", "answers", "em", "f1", "ttft_ms", "peak_memory_mb"]
            )
            for ex in metrics["examples"]:
                table.add_data(
                    ex.get("qid"),
                    ex.get("question"),
                    ex.get("prediction"),
                    ex.get("answers"),
                    ex.get("em"),
                    ex.get("f1"),
                    ex.get("ttft_ms"),
                    ex.get("peak_memory_mb"),
                )
            log_metrics["eval/examples"] = table
        wandb.log(log_metrics)
        run.finish()

    print(json.dumps({k: v for k, v in metrics.items() if k != "examples"}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
