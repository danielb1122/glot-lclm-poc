from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from glot_lclm.utils.runtime import get_dtype


@dataclass
class LoadedBackbone:
    model: torch.nn.Module
    tokenizer: Any
    hidden_size: int


def _quantization_config(load_in_4bit: bool):
    if not load_in_4bit:
        return None
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:
        raise ImportError("4-bit loading requires a recent transformers install") from exc
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "right"


def load_encoder(name: str, cfg: dict) -> LoadedBackbone:
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=cfg.get("trust_remote_code", True))
    _ensure_pad_token(tokenizer)
    model = AutoModel.from_pretrained(
        name,
        trust_remote_code=cfg.get("trust_remote_code", True),
        torch_dtype=get_dtype(cfg.get("dtype")),
        quantization_config=_quantization_config(bool(cfg.get("load_in_4bit", False))),
        device_map="auto" if bool(cfg.get("load_in_4bit", False)) else None,
    )
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    hidden_size = int(getattr(model.config, "hidden_size"))
    return LoadedBackbone(model=model, tokenizer=tokenizer, hidden_size=hidden_size)


def load_decoder(name: str, cfg: dict) -> LoadedBackbone:
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=cfg.get("trust_remote_code", True))
    _ensure_pad_token(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(
        name,
        trust_remote_code=cfg.get("trust_remote_code", True),
        torch_dtype=get_dtype(cfg.get("dtype")),
        quantization_config=_quantization_config(bool(cfg.get("load_in_4bit", False))),
        device_map="auto" if bool(cfg.get("load_in_4bit", False)) else None,
    )
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    hidden_size = int(getattr(model.config, "hidden_size"))
    return LoadedBackbone(model=model, tokenizer=tokenizer, hidden_size=hidden_size)


def maybe_apply_lora(model: torch.nn.Module, lora_cfg: dict, task_type: str) -> torch.nn.Module:
    if not lora_cfg or not lora_cfg.get("enabled", False):
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise ImportError("LoRA requires peft") from exc

    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model)

    peft_task_type = {
        "causal_lm": TaskType.CAUSAL_LM,
        "feature_extraction": TaskType.FEATURE_EXTRACTION,
    }[task_type]
    config = LoraConfig(
        task_type=peft_task_type,
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", [])),
        bias="none",
    )
    return get_peft_model(model, config)

