from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from glot_lclm.data.qa_examples import load_qa_dataset, maybe_limit, select_range
from glot_lclm.evaluation.qa_eval import evaluate_qa
from glot_lclm.models.compressor_qa import FullContextQAModel
from glot_lclm.training.checkpoints import load_checkpoint
from glot_lclm.training.train import _move_model_if_needed, _split_or_fallback
from glot_lclm.utils.config import apply_overrides, flatten_dict, load_config
from glot_lclm.utils.runtime import ensure_dir, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", type=int, default=None)
    args, unknown = parser.parse_known_args()
    args.overrides = [x[2:] if x.startswith("--") else x for x in unknown]
    return args


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.overrides)
    set_seed(int(cfg["experiment"].get("seed", 42)))
    raw = load_qa_dataset(cfg["dataset"])
    split = args.split or cfg["dataset"]["eval_split"]
    ds = maybe_limit(_split_or_fallback(raw, split), cfg["dataset"].get("eval_limit"))
    ds = select_range(ds, start_index=args.start_index, limit=args.max_examples)

    model = FullContextQAModel(cfg)
    model = _move_model_if_needed(model, args.device)
    step = 0
    if args.checkpoint:
        ckpt = load_checkpoint(model, args.checkpoint)
        step = int(ckpt.get("step", 0))

    metrics = evaluate_qa(model, ds, cfg, mode="full_context")
    metrics["start_index"] = args.start_index
    out_dir = ensure_dir(Path(cfg["experiment"]["output_dir"]) / "eval")
    stem = Path(args.checkpoint).stem if args.checkpoint else "base"
    end_label = "end" if args.max_examples is None else str(args.start_index + args.max_examples)
    out_path = out_dir / f"{stem}_{split}_{args.start_index}_{end_label}.json"
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
        wandb.log(log_metrics, step=step)
        run.finish()

    print(json.dumps({k: v for k, v in metrics.items() if k != "examples"}, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
