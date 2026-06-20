from __future__ import annotations

from dataclasses import dataclass

from glot_lclm.data.qa_examples import QAExample


SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "Return the shortest answer span or phrase."
)


@dataclass
class PromptParts:
    prefix: str
    suffix: str
    answer: str


def compressed_prompt_parts(example: QAExample) -> PromptParts:
    prefix = f"{SYSTEM_PROMPT}\n\nCompressed context:\n"
    suffix = f"\n\nQuestion: {example.question}\nAnswer:"
    answer = " " + (example.answers[0] if example.answers else "")
    return PromptParts(prefix=prefix, suffix=suffix, answer=answer)


def full_context_prompt_parts(example: QAExample, context: str | None = None) -> PromptParts:
    ctx = example.context if context is None else context
    prefix = f"{SYSTEM_PROMPT}\n\nContext:\n{ctx}\n\nQuestion: {example.question}\nAnswer:"
    answer = " " + (example.answers[0] if example.answers else "")
    return PromptParts(prefix=prefix, suffix="", answer=answer)


def full_context_generation_prompt(example: QAExample, context: str | None = None) -> str:
    return full_context_prompt_parts(example, context=context).prefix

