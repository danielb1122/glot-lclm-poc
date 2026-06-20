from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from glot_lclm.data.qa_examples import QAExample
from glot_lclm.models.compressor_qa import CompressedQAModel, FullContextQAModel
from glot_lclm.utils.runtime import cuda_peak_memory_mb, cuda_sync


@dataclass
class GenerationResult:
    texts: list[str]
    ttft_ms: float
    peak_memory_mb: float
    generated_token_count: int


@torch.no_grad()
def greedy_generate(
    model: CompressedQAModel | FullContextQAModel,
    examples: list[QAExample],
    mode: str,
    max_new_tokens: int,
) -> GenerationResult:
    decoder = model.decoder
    tokenizer = model.decoder_tokenizer
    decoder.eval()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    cuda_sync()
    start = time.perf_counter()

    if mode == "compressed":
        outputs, attention_mask = model.compressed_prefill(examples)
    elif mode == "full_context":
        outputs, attention_mask = model.full_context_prefill(examples)
    else:
        raise ValueError(f"Unknown generation mode: {mode}")

    next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    cuda_sync()
    ttft_ms = (time.perf_counter() - start) * 1000.0

    generated = [next_token]
    past_key_values = outputs.past_key_values
    eos = tokenizer.eos_token_id
    finished = torch.zeros(next_token.size(0), dtype=torch.bool, device=next_token.device)
    if eos is not None:
        finished |= next_token.squeeze(-1).eq(eos)

    for _ in range(max_new_tokens - 1):
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)],
            dim=1,
        )
        outputs = decoder(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated.append(next_token)
        if eos is not None:
            finished |= next_token.squeeze(-1).eq(eos)
            if bool(finished.all()):
                break

    token_ids = torch.cat(generated, dim=1)
    texts = tokenizer.batch_decode(token_ids, skip_special_tokens=True)
    return GenerationResult(
        texts=[text.strip() for text in texts],
        ttft_ms=ttft_ms,
        peak_memory_mb=cuda_peak_memory_mb(),
        generated_token_count=int(token_ids.numel()),
    )

