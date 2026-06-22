from __future__ import annotations

import argparse
import itertools
import math
import time
from typing import Any

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from glot_lclm.data.qa_examples import QACollator, load_qa_dataset, maybe_limit
from glot_lclm.evaluation.qa_eval import evaluate_qa
from glot_lclm.models.compressor_qa import CompressedQAModel, FullContextQAModel, set_trainability
from glot_lclm.training.checkpoints import save_checkpoint
from glot_lclm.utils.config import apply_overrides, flatten_dict, load_config, save_config
from glot_lclm.utils.runtime import (
    count_trainable_parameters,
    detach_to_cpu,
    ensure_dir,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eval-max-examples", type=int, default=0)
    args, unknown = parser.parse_known_args()
    cleaned = []
    for item in unknown:
        cleaned.append(item[2:] if item.startswith("--") else item)
    args.overrides = cleaned
    return args


def _move_model_if_needed(model: torch.nn.Module, device: str) -> torch.nn.Module:
    quantized = any(
        getattr(module, "is_loaded_in_4bit", False) or getattr(module, "is_loaded_in_8bit", False)
        for module in model.modules()
    )
    if not quantized:
        model.to(device)
    return model


def _split_or_fallback(raw, requested: str):
    if requested in raw:
        return raw[requested]
    aliases = {
        "validation": ["validation", "dev", "eval", "test"],
        "train": ["train"],
        "test": ["test", "validation", "dev"],
    }
    for name in aliases.get(requested, []):
        if name in raw:
            return raw[name]
    available = ", ".join(raw.keys())
    raise KeyError(f"Split {requested!r} not found. Available splits: {available}")


def _wandb_init(cfg: dict[str, Any]):
    if wandb is None or cfg["experiment"].get("wandb_mode", "online") == "disabled":
        return None
    return wandb.init(
        project=cfg["experiment"].get("wandb_project"),
        entity=cfg["experiment"].get("wandb_entity"),
        name=cfg["experiment"]["name"],
        config=flatten_dict(cfg),
        mode=cfg["experiment"].get("wandb_mode", "online"),
        tags=cfg["experiment"].get("tags", []),
    )


def _log(metrics: dict[str, Any], step: int):
    if wandb is not None and wandb.run is not None:
        wandb.log(metrics, step=step)


def _sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _wandb_examples_table(examples: list[dict[str, Any]]):
    if wandb is None or wandb.run is None or not examples:
        return None
    columns = ["qid", "question", "prediction", "answers", "em", "f1", "ttft_ms", "peak_memory_mb"]
    table = wandb.Table(columns=columns)
    for ex in examples:
        table.add_data(*(ex.get(col) for col in columns))
    return table


def _trainable_parameters(model: torch.nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def _stage_trainability(stage: dict[str, Any]) -> dict[str, bool]:
    return {
        "train_pooler": bool(stage.get("train_pooler", False)),
        "train_adapter": bool(stage.get("train_adapter", False)),
        "train_encoder_lora": bool(stage.get("train_encoder_lora", False)),
        "train_decoder_lora": bool(stage.get("train_decoder_lora", False)),
    }


def _log_eval_metrics(prefix: str, eval_metrics: dict[str, Any], step: int) -> dict[str, Any]:
    scalar_log_metrics = {
        f"{prefix}/{k}": v
        for k, v in eval_metrics.items()
        if k != "examples"
    }
    log_metrics = dict(scalar_log_metrics)
    table = _wandb_examples_table(eval_metrics.get("examples", []))
    if table is not None:
        log_metrics[f"{prefix}/examples"] = table
    _log(log_metrics, step=step)
    return scalar_log_metrics


def main() -> None:
    args = parse_args()
    eval_max_examples = None if args.eval_max_examples <= 0 else args.eval_max_examples
    cfg = apply_overrides(load_config(args.config), args.overrides)
    set_seed(int(cfg["experiment"].get("seed", 42)))

    out_dir = ensure_dir(cfg["experiment"]["output_dir"])
    save_config(cfg, out_dir / "resolved_config.yaml")

    raw = load_qa_dataset(cfg["dataset"])
    train_ds = maybe_limit(_split_or_fallback(raw, cfg["dataset"]["train_split"]), cfg["dataset"].get("train_limit"))
    eval_ds = maybe_limit(_split_or_fallback(raw, cfg["dataset"]["eval_split"]), cfg["dataset"].get("eval_limit"))

    collator = QACollator(include_titles=bool(cfg["dataset"].get("include_titles", True)))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["training"].get("num_workers", 0)),
        collate_fn=collator,
    )

    has_compressor = bool(cfg["model"].get("encoder_name"))
    model = CompressedQAModel(cfg) if has_compressor else FullContextQAModel(cfg)
    model = _move_model_if_needed(model, args.device)
    run = _wandb_init(cfg)

    trainable, total = count_trainable_parameters(model)
    _log({"params/trainable_initial": trainable, "params/total": total}, step=0)

    global_step = 0
    best_f1 = -math.inf
    grad_accum = int(cfg["training"].get("gradient_accumulation_steps", 1))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 1.0))
    log_every = int(cfg["training"].get("log_every_steps", 10))
    eval_every = int(cfg["training"].get("eval_every_steps", 250))
    save_every = int(cfg["training"].get("save_every_steps", 500))

    train_start = _sync_time()

    eval_start = _sync_time()
    pretrained_metrics = evaluate_qa(
        model,
        eval_ds,
        cfg,
        mode="full_context",
        max_examples=eval_max_examples,
        show_progress=False,
    )
    pretrained_metrics["wall_time_s"] = _sync_time() - eval_start
    _log_eval_metrics("pretrained_full_context", pretrained_metrics, step=global_step)

    for stage in cfg["training"]["stages"]:
        stage_name = stage["name"]
        stage_start = _sync_time()
        set_trainability(model, **_stage_trainability(stage))
        trainable_params = _trainable_parameters(model)
        if not trainable_params:
            raise ValueError(f"Stage has no trainable parameters: {stage_name}")
        optimizer = AdamW(trainable_params, lr=float(stage["lr"]))
        trainable, total = count_trainable_parameters(model)
        _log(
            {
                "stage/index": cfg["training"]["stages"].index(stage),
                "stage/name": stage_name,
                "stage/trainable_params": trainable,
                "stage/total_params": total,
            },
            step=global_step,
        )

        model.train()
        iterator = itertools.cycle(train_loader)
        total_micro_steps = int(stage["steps"]) * grad_accum
        pbar = tqdm(range(total_micro_steps), desc=f"train/{stage_name}")
        optimizer.zero_grad(set_to_none=True)
        step_start = _sync_time()
        for local_micro_step in pbar:
            batch = next(iterator)
            if stage.get("mode") == "full_context":
                out = model.forward_full_context(batch)
            elif stage.get("mode") == "compressed":
                if not has_compressor:
                    raise ValueError("Compressed stage requested but model.encoder_name is null")
                out = model.forward_compressed(batch)
            else:
                raise ValueError(f"Unknown stage mode: {stage.get('mode')}")

            loss = out.loss / grad_accum
            loss.backward()

            if (local_micro_step + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                now = _sync_time()
                step_time_s = now - step_start
                step_start = now

                metrics = {
                    "train/loss": float(loss.detach().cpu()) * grad_accum,
                    "train/stage": stage_name,
                    "time/step_s": step_time_s,
                    f"time/{stage_name}_step_s": step_time_s,
                    "time/elapsed_s": now - train_start,
                }
                metrics.update({k: float(v.detach().cpu()) for k, v in out.metrics.items()})
                if global_step % log_every == 0:
                    _log(metrics, step=global_step)
                pbar.set_postfix(loss=f"{metrics['train/loss']:.4f}", step=global_step)

                if global_step % save_every == 0:
                    save_checkpoint(
                        model,
                        out_dir,
                        name=f"step_{global_step}",
                        config=cfg,
                        step=global_step,
                    )

                if global_step % eval_every == 0:
                    eval_mode = "compressed" if stage.get("mode") == "compressed" else "full_context"
                    eval_start = _sync_time()
                    eval_metrics = evaluate_qa(
                        model,
                        eval_ds,
                        cfg,
                        mode=eval_mode,
                        max_examples=eval_max_examples,
                        show_progress=False,
                    )
                    eval_metrics["wall_time_s"] = _sync_time() - eval_start
                    eval_metrics["stage"] = stage_name
                    scalar_log_metrics = _log_eval_metrics("eval", eval_metrics, step=global_step)
                    if eval_metrics["f1"] > best_f1:
                        best_f1 = float(eval_metrics["f1"])
                        save_checkpoint(
                            model,
                            out_dir,
                            name="best",
                            config=cfg,
                            step=global_step,
                            metrics=detach_to_cpu(scalar_log_metrics),
                        )
                    model.train()

        stage_time_s = _sync_time() - stage_start
        _log(
            {
                f"time/stage_{stage_name}_s": stage_time_s,
                "time/last_stage_s": stage_time_s,
                "stage/completed": stage_name,
            },
            step=global_step,
        )

        if stage.get("mode") == "full_context":
            eval_start = _sync_time()
            warmup_metrics = evaluate_qa(
                model,
                eval_ds,
                cfg,
                mode="full_context",
                max_examples=eval_max_examples,
                show_progress=False,
            )
            warmup_metrics["wall_time_s"] = _sync_time() - eval_start
            _log_eval_metrics(f"after_{stage['name']}", warmup_metrics, step=global_step)
            model.train()

    final_mode = "compressed" if has_compressor else "full_context"
    eval_start = _sync_time()
    final_metrics = evaluate_qa(
        model,
        eval_ds,
        cfg,
        mode=final_mode,
        max_examples=eval_max_examples,
        show_progress=True,
    )
    final_metrics["wall_time_s"] = _sync_time() - eval_start
    _log_eval_metrics("final", final_metrics, step=global_step)
    _log({"time/total_s": _sync_time() - train_start}, step=global_step)
    save_checkpoint(model, out_dir, name="last", config=cfg, step=global_step, metrics=final_metrics)

    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
