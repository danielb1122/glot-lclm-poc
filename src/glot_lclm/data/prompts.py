from __future__ import annotations

from dataclasses import dataclass

from glot_lclm.data.qa_examples import QAExample


SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "Return the shortest answer span or phrase."
)

REPEAT_SYSTEM_PROMPT = "Repeat the provided context exactly."
MEMORY_START = "<|memory_start|>"
MEMORY_END = "<|memory_end|>"


@dataclass
class PromptParts:
    prefix: str
    suffix: str
    answer: str


def compressed_prompt_parts(example: QAExample) -> PromptParts:
    if example.task_type == "repeat":
        if example.prompt_style == "lclm_memory":
            prefix = f"{REPEAT_SYSTEM_PROMPT}\n\n{MEMORY_START}"
            suffix = f"{MEMORY_END}\n\n{example.question}Output:"
            answer = "\n" + (example.answers[0] if example.answers else "")
            return PromptParts(prefix=prefix, suffix=suffix, answer=answer)
        prefix = f"{REPEAT_SYSTEM_PROMPT}\n\nCompressed context:\n"
        suffix = f"\n\n{example.question}Output:"
        answer = "\n" + (example.answers[0] if example.answers else "")
        return PromptParts(prefix=prefix, suffix=suffix, answer=answer)

    prefix = f"{SYSTEM_PROMPT}\n\nCompressed context:\n"
    suffix = f"\n\nQuestion: {example.question}\nAnswer:"
    answer = " " + (example.answers[0] if example.answers else "")
    return PromptParts(prefix=prefix, suffix=suffix, answer=answer)


def full_context_prompt_parts(example: QAExample, context: str | None = None) -> PromptParts:
    ctx = example.context if context is None else context
    if example.task_type == "repeat":
        if example.prompt_style == "lclm_memory":
            prefix = (
                f"{REPEAT_SYSTEM_PROMPT}\n\n"
                f"{MEMORY_START}{ctx}{MEMORY_END}\n\n"
                f"{example.question}Output:"
            )
            answer = "\n" + (example.answers[0] if example.answers else "")
            return PromptParts(prefix=prefix, suffix="", answer=answer)
        prefix = f"{REPEAT_SYSTEM_PROMPT}\n\nContext:\n{ctx}\n\n{example.question}Output:"
        answer = "\n" + (example.answers[0] if example.answers else "")
        return PromptParts(prefix=prefix, suffix="", answer=answer)

    prefix = f"{SYSTEM_PROMPT}\n\nContext:\n{ctx}\n\nQuestion: {example.question}\nAnswer:"
    answer = " " + (example.answers[0] if example.answers else "")
    return PromptParts(prefix=prefix, suffix="", answer=answer)


def full_context_generation_prompt(example: QAExample, context: str | None = None) -> str:
    return full_context_prompt_parts(example, context=context).prefix
