from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from glot_lclm.data.prompts import compressed_prompt_parts, full_context_prompt_parts
from glot_lclm.data.qa_examples import QAExample
from glot_lclm.models.adapter import build_adapter, load_pretrained_adapter
from glot_lclm.models.loaders import load_decoder, load_encoder, maybe_apply_lora
from glot_lclm.models.poolers import MeanBlockPooler, build_pooler


@dataclass
class ForwardOutput:
    loss: torch.Tensor
    logits: torch.Tensor | None
    metrics: dict[str, torch.Tensor]


def _last_hidden(outputs: Any) -> torch.Tensor:
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
        return outputs.hidden_states[-1]
    if isinstance(outputs, tuple):
        return outputs[0]
    raise ValueError("Could not find hidden states in encoder output")


class CompressedQAModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg["model"]
        self.encoder_backbone = load_encoder(model_cfg["encoder_name"], model_cfg)
        self.decoder_backbone = load_decoder(model_cfg["decoder_name"], model_cfg)

        self.encoder = maybe_apply_lora(
            self.encoder_backbone.model,
            cfg.get("lora", {}).get("encoder", {}),
            task_type="feature_extraction",
        )
        self.decoder = maybe_apply_lora(
            self.decoder_backbone.model,
            cfg.get("lora", {}).get("decoder", {}),
            task_type="causal_lm",
        )
        if model_cfg.get("gradient_checkpointing", False):
            if hasattr(self.encoder, "gradient_checkpointing_enable"):
                self.encoder.gradient_checkpointing_enable()
            if hasattr(self.decoder, "gradient_checkpointing_enable"):
                self.decoder.gradient_checkpointing_enable()
            if hasattr(self.decoder.config, "use_cache"):
                self.decoder.config.use_cache = False

        self.encoder_tokenizer = self.encoder_backbone.tokenizer
        self.decoder_tokenizer = self.decoder_backbone.tokenizer
        self.pooler = build_pooler(self.encoder_backbone.hidden_size, cfg["compression"])
        self.adapter = build_adapter(
            input_dim=int(self.pooler.out_dim),
            output_dim=self.decoder_backbone.hidden_size,
            cfg=cfg["compression"],
        )
        adapter_checkpoint = cfg["compression"].get("adapter_checkpoint")
        if adapter_checkpoint:
            load_pretrained_adapter(self.adapter, adapter_checkpoint)

    @torch.no_grad()
    def check_pooler_mean_initialization(
        self,
        device: str | torch.device | None = None,
        atol: float | None = None,
    ) -> dict[str, float]:
        compression_cfg = self.cfg.get("compression", {})
        glot_cfg = compression_cfg.get("glot", {})
        should_check = (
            compression_cfg.get("pooler") == "glot"
            and bool(glot_cfg.get("init_as_mean", False))
        )
        if not should_check:
            return {}

        target_device = torch.device(device) if device is not None else self.device
        if target_device.type == "cuda" and not torch.cuda.is_available():
            target_device = torch.device("cpu")

        check_dtype = torch.float32

        ratio = int(compression_cfg["ratio"])
        input_dim = int(self.encoder_backbone.hidden_size)
        seq_len = max(ratio * 2 - 3, ratio)
        hidden = torch.randn(
            2,
            seq_len,
            input_dim,
            device=target_device,
            dtype=check_dtype,
        )
        attention_mask = torch.ones(2, seq_len, device=target_device, dtype=torch.long)

        was_training = self.pooler.training
        self.pooler.eval()
        self.pooler.to(device=target_device, dtype=check_dtype)
        glot_latents = self.pooler(hidden, attention_mask).latents
        mean_latents = MeanBlockPooler(input_dim=input_dim, compression_ratio=ratio)(
            hidden,
            attention_mask,
        ).latents
        if was_training:
            self.pooler.train()

        max_abs_diff = float((glot_latents - mean_latents).abs().max().detach().cpu())
        tolerance = (
            atol
            if atol is not None
            else 1e-5
        )
        ok = max_abs_diff <= tolerance
        if not ok:
            raise RuntimeError(
                "GLOT init_as_mean check failed: "
                f"max_abs_diff={max_abs_diff:.6g}, tolerance={tolerance:.6g}. "
                "The GLOT pooler is not initialized equivalently to mean pooling."
            )

        print(
            "[init-check] GLOT equals mean pooling at initialization: "
            f"max_abs_diff={max_abs_diff:.6g}, tolerance={tolerance:.6g}, "
            f"dtype={check_dtype}, device={target_device}"
        )
        return {
            "glot_mean_init_max_abs_diff": max_abs_diff,
            "glot_mean_init_tolerance": float(tolerance),
        }

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def decoder_embedding_device(self) -> torch.device:
        return self.decoder.get_input_embeddings().weight.device

    def _encoder_window_tokens(self, max_context_tokens: int) -> int:
        window_tokens = self.cfg.get("compression", {}).get("encoder_window_tokens")
        if window_tokens is None or int(window_tokens) <= 0:
            return max_context_tokens
        return min(int(window_tokens), max_context_tokens)

    def encode_context(self, examples: list[QAExample]) -> tuple[torch.Tensor, torch.Tensor, dict]:
        contexts = [ex.context for ex in examples]
        max_context_tokens = int(self.cfg["dataset"]["max_context_tokens"])
        window_tokens = self._encoder_window_tokens(max_context_tokens)
        encoded = self.encoder_tokenizer(
            contexts,
            truncation=True,
            max_length=max_context_tokens,
            padding=False,
            return_attention_mask=True,
        )

        windows: list[dict[str, list[int]]] = []
        owners: list[int] = []
        for example_idx, input_ids in enumerate(encoded["input_ids"]):
            attention_mask = encoded["attention_mask"][example_idx]
            if not input_ids:
                continue
            for start in range(0, len(input_ids), window_tokens):
                end = min(start + window_tokens, len(input_ids))
                windows.append(
                    {
                        "input_ids": input_ids[start:end],
                        "attention_mask": attention_mask[start:end],
                    }
                )
                owners.append(example_idx)

        if not windows:
            pad_id = self.encoder_tokenizer.pad_token_id or 0
            windows = [{"input_ids": [pad_id], "attention_mask": [0]}]
            owners = [0]

        batch = self.encoder_tokenizer.pad(
            windows,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.encoder(**batch, output_hidden_states=True, return_dict=True)
        hidden = _last_hidden(outputs)
        self.pooler.to(device=hidden.device, dtype=hidden.dtype)
        self.adapter.to(device=hidden.device, dtype=hidden.dtype)
        pooled = self.pooler(hidden, batch["attention_mask"])

        per_example_latents: list[list[torch.Tensor]] = [[] for _ in examples]
        per_example_masks: list[list[torch.Tensor]] = [[] for _ in examples]
        for window_idx, example_idx in enumerate(owners):
            per_example_latents[example_idx].append(pooled.latents[window_idx])
            per_example_masks[example_idx].append(pooled.latent_mask[window_idx])

        latent_sequences: list[torch.Tensor] = []
        mask_sequences: list[torch.Tensor] = []
        for example_idx in range(len(examples)):
            if per_example_latents[example_idx]:
                latent_sequences.append(torch.cat(per_example_latents[example_idx], dim=0))
                mask_sequences.append(torch.cat(per_example_masks[example_idx], dim=0))
            else:
                latent_sequences.append(pooled.latents.new_zeros(1, pooled.latents.size(-1)))
                mask_sequences.append(pooled.latent_mask.new_zeros(1))

        max_latents = max(seq.size(0) for seq in latent_sequences)
        latent_tensor = pooled.latents.new_zeros(len(examples), max_latents, pooled.latents.size(-1))
        latent_mask = pooled.latent_mask.new_zeros(len(examples), max_latents)
        for example_idx, seq in enumerate(latent_sequences):
            length = seq.size(0)
            latent_tensor[example_idx, :length] = seq
            latent_mask[example_idx, :length] = mask_sequences[example_idx]

        adapted = self.adapter(latent_tensor).to(self.decoder_embedding_device)
        latent_mask = latent_mask.to(self.decoder_embedding_device)
        aux = dict(pooled.aux)
        aux["encoder_window_tokens"] = torch.tensor(
            float(window_tokens),
            device=hidden.device,
        )
        aux["encoder_windows_per_example"] = torch.tensor(
            float(len(windows) / max(len(examples), 1)),
            device=hidden.device,
        )
        aux["latent_tokens_per_example"] = latent_mask.float().sum(dim=1).mean().to(hidden.device)
        return adapted, latent_mask, aux

    def _token_ids(self, text: str, add_special_tokens: bool = False) -> torch.Tensor:
        ids = self.decoder_tokenizer(
            text,
            add_special_tokens=add_special_tokens,
            return_tensors="pt",
        )["input_ids"][0]
        return ids.to(self.device)

    def _text_embeds(self, ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.get_input_embeddings()(ids.unsqueeze(0)).squeeze(0)

    def truncate_context(self, text: str) -> str:
        max_context_tokens = int(self.cfg["dataset"]["max_context_tokens"])
        ids = self.decoder_tokenizer(
            text,
            truncation=True,
            max_length=max_context_tokens,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        return self.decoder_tokenizer.decode(ids, skip_special_tokens=True)

    def _build_compressed_inputs(
        self,
        examples: list[QAExample],
        include_answer: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor]]:
        latent_embeds, latent_mask, aux = self.encode_context(examples)
        bsz = len(examples)
        seq_embeds: list[torch.Tensor] = []
        seq_labels: list[torch.Tensor] = []
        seq_masks: list[torch.Tensor] = []

        eos = self.decoder_tokenizer.eos_token_id
        for i, ex in enumerate(examples):
            parts = compressed_prompt_parts(ex)
            prefix_ids = self._token_ids(parts.prefix, add_special_tokens=True)
            suffix_ids = self._token_ids(parts.suffix, add_special_tokens=False)

            pieces = [self._text_embeds(prefix_ids)]
            labels = [
                torch.full_like(prefix_ids, -100),
            ]

            valid_latents = latent_embeds[i, latent_mask[i].bool()]
            pieces.append(valid_latents)
            labels.append(torch.full((valid_latents.size(0),), -100, device=self.device, dtype=torch.long))

            pieces.append(self._text_embeds(suffix_ids))
            labels.append(torch.full_like(suffix_ids, -100))

            if include_answer:
                answer_ids = self._token_ids(parts.answer, add_special_tokens=False)
                if eos is not None:
                    answer_ids = torch.cat([answer_ids, torch.tensor([eos], device=self.device)])
                pieces.append(self._text_embeds(answer_ids))
                labels.append(answer_ids)

            embeds = torch.cat(pieces, dim=0)
            label = torch.cat(labels, dim=0) if include_answer else None
            seq_embeds.append(embeds)
            if include_answer and label is not None:
                seq_labels.append(label)
            seq_masks.append(torch.ones(embeds.size(0), device=self.device, dtype=torch.long))

        max_len = max(x.size(0) for x in seq_embeds)
        hidden_size = seq_embeds[0].size(-1)
        inputs_embeds = torch.zeros(bsz, max_len, hidden_size, device=self.device, dtype=seq_embeds[0].dtype)
        attention_mask = torch.zeros(bsz, max_len, device=self.device, dtype=torch.long)
        labels_out = (
            torch.full((bsz, max_len), -100, device=self.device, dtype=torch.long)
            if include_answer
            else None
        )
        for i, embeds in enumerate(seq_embeds):
            length = embeds.size(0)
            inputs_embeds[i, :length] = embeds
            attention_mask[i, :length] = 1
            if include_answer and labels_out is not None:
                labels_out[i, :length] = seq_labels[i]

        return inputs_embeds, attention_mask, labels_out, aux

    def forward_compressed(self, examples: list[QAExample]) -> ForwardOutput:
        inputs_embeds, attention_mask, labels, aux = self._build_compressed_inputs(
            examples,
            include_answer=True,
        )
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        metrics = {f"pool/{k}": v for k, v in aux.items() if torch.is_tensor(v)}
        return ForwardOutput(loss=outputs.loss, logits=outputs.logits, metrics=metrics)

    def _build_full_inputs(
        self,
        examples: list[QAExample],
        include_answer: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        input_ids: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        eos = self.decoder_tokenizer.eos_token_id
        for ex in examples:
            context = self.truncate_context(ex.context)
            parts = full_context_prompt_parts(ex, context=context)
            prefix_ids = self.decoder_tokenizer(
                parts.prefix,
                add_special_tokens=True,
                return_tensors="pt",
            )["input_ids"][0].to(self.device)
            pieces = [prefix_ids]
            label_pieces = [torch.full_like(prefix_ids, -100)]
            if include_answer:
                answer_ids = self.decoder_tokenizer(
                    parts.answer,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"][0].to(self.device)
                if eos is not None:
                    answer_ids = torch.cat([answer_ids, torch.tensor([eos], device=self.device)])
                pieces.append(answer_ids)
                label_pieces.append(answer_ids)
            ids = torch.cat(pieces)
            input_ids.append(ids)
            if include_answer:
                labels.append(torch.cat(label_pieces))

        max_len = max(x.size(0) for x in input_ids)
        pad_id = self.decoder_tokenizer.pad_token_id
        batch_ids = torch.full((len(examples), max_len), pad_id, device=self.device, dtype=torch.long)
        attention_mask = torch.zeros(len(examples), max_len, device=self.device, dtype=torch.long)
        labels_out = (
            torch.full((len(examples), max_len), -100, device=self.device, dtype=torch.long)
            if include_answer
            else None
        )
        for i, ids in enumerate(input_ids):
            length = ids.size(0)
            batch_ids[i, :length] = ids
            attention_mask[i, :length] = 1
            if include_answer and labels_out is not None:
                labels_out[i, :length] = labels[i]
        return batch_ids, attention_mask, labels_out

    def forward_full_context(self, examples: list[QAExample]) -> ForwardOutput:
        input_ids, attention_mask, labels = self._build_full_inputs(examples, include_answer=True)
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return ForwardOutput(loss=outputs.loss, logits=outputs.logits, metrics={})

    @torch.no_grad()
    def compressed_prefill(self, examples: list[QAExample]):
        inputs_embeds, attention_mask, _, _ = self._build_compressed_inputs(
            examples,
            include_answer=False,
        )
        return self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        ), attention_mask

    @torch.no_grad()
    def full_context_prefill(self, examples: list[QAExample]):
        input_ids, attention_mask, _ = self._build_full_inputs(examples, include_answer=False)
        return self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        ), attention_mask


class FullContextQAModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg["model"]
        self.decoder_backbone = load_decoder(model_cfg["decoder_name"], model_cfg)
        self.decoder = maybe_apply_lora(
            self.decoder_backbone.model,
            cfg.get("lora", {}).get("decoder", {}),
            task_type="causal_lm",
        )
        if model_cfg.get("gradient_checkpointing", False) and hasattr(
            self.decoder, "gradient_checkpointing_enable"
        ):
            self.decoder.gradient_checkpointing_enable()
            if hasattr(self.decoder.config, "use_cache"):
                self.decoder.config.use_cache = False
        self.decoder_tokenizer = self.decoder_backbone.tokenizer

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def truncate_context(self, text: str) -> str:
        max_context_tokens = int(self.cfg["dataset"]["max_context_tokens"])
        ids = self.decoder_tokenizer(
            text,
            truncation=True,
            max_length=max_context_tokens,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"][0]
        return self.decoder_tokenizer.decode(ids, skip_special_tokens=True)

    def _build_full_inputs(
        self,
        examples: list[QAExample],
        include_answer: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        input_ids: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        eos = self.decoder_tokenizer.eos_token_id
        for ex in examples:
            context = self.truncate_context(ex.context)
            parts = full_context_prompt_parts(ex, context=context)
            prefix_ids = self.decoder_tokenizer(
                parts.prefix,
                add_special_tokens=True,
                return_tensors="pt",
            )["input_ids"][0].to(self.device)
            pieces = [prefix_ids]
            label_pieces = [torch.full_like(prefix_ids, -100)]
            if include_answer:
                answer_ids = self.decoder_tokenizer(
                    parts.answer,
                    add_special_tokens=False,
                    return_tensors="pt",
                )["input_ids"][0].to(self.device)
                if eos is not None:
                    answer_ids = torch.cat([answer_ids, torch.tensor([eos], device=self.device)])
                pieces.append(answer_ids)
                label_pieces.append(answer_ids)
            ids = torch.cat(pieces)
            input_ids.append(ids)
            if include_answer:
                labels.append(torch.cat(label_pieces))

        max_len = max(x.size(0) for x in input_ids)
        pad_id = self.decoder_tokenizer.pad_token_id
        batch_ids = torch.full((len(examples), max_len), pad_id, device=self.device, dtype=torch.long)
        attention_mask = torch.zeros(len(examples), max_len, device=self.device, dtype=torch.long)
        labels_out = (
            torch.full((len(examples), max_len), -100, device=self.device, dtype=torch.long)
            if include_answer
            else None
        )
        for i, ids in enumerate(input_ids):
            length = ids.size(0)
            batch_ids[i, :length] = ids
            attention_mask[i, :length] = 1
            if include_answer and labels_out is not None:
                labels_out[i, :length] = labels[i]
        return batch_ids, attention_mask, labels_out

    def forward_full_context(self, examples: list[QAExample]) -> ForwardOutput:
        input_ids, attention_mask, labels = self._build_full_inputs(examples, include_answer=True)
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )
        return ForwardOutput(loss=outputs.loss, logits=outputs.logits, metrics={})

    @torch.no_grad()
    def full_context_prefill(self, examples: list[QAExample]):
        input_ids, attention_mask, _ = self._build_full_inputs(examples, include_answer=False)
        return self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        ), attention_mask


def set_trainability(
    model: nn.Module,
    *,
    train_pooler: bool,
    train_adapter: bool,
    train_encoder_lora: bool,
    train_decoder_lora: bool,
) -> None:
    train_full_tiny_encoder = (
        train_encoder_lora
        and getattr(model, "cfg", {}).get("model", {}).get("encoder_name") == "tiny-random"
    )
    train_full_tiny_decoder = (
        train_decoder_lora
        and getattr(model, "cfg", {}).get("model", {}).get("decoder_name") == "tiny-random"
    )
    for name, param in model.named_parameters():
        train = False
        if ".pooler." in f".{name}." or name.startswith("pooler."):
            train = train_pooler
        elif ".adapter." in f".{name}." or name.startswith("adapter."):
            train = train_adapter
        elif train_full_tiny_encoder and name.startswith("encoder."):
            train = True
        elif train_full_tiny_decoder and name.startswith("decoder."):
            train = True
        elif name.startswith("encoder.") and "lora_" in name:
            train = train_encoder_lora
        elif name.startswith("decoder.") and "lora_" in name:
            train = train_decoder_lora
        elif isinstance(model, FullContextQAModel) and name.startswith("decoder.") and "lora_" in name:
            train = train_decoder_lora
        param.requires_grad = train


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names or "lora_" in name or name.startswith("adapter.") or name.startswith("pooler.")
    }
