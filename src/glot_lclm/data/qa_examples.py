from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


@dataclass
class QAExample:
    qid: str
    question: str
    context: str
    answers: list[str]
    support_indices: list[int]
    task_type: str = "qa"
    prompt_style: str = "default"


def _first_present(row: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    for key in keys:
        if key in row and row[key] is not None:
            return row[key]
    return default


def _normalize_answers(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        for key in ["text", "answer", "answers", "aliases"]:
            if key in value:
                return _normalize_answers(value[key])
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                out.extend(_normalize_answers(item))
        return list(dict.fromkeys(x for x in out if x))
    return [str(value)]


def _paragraph_text(paragraph: Any, include_titles: bool) -> tuple[str, bool]:
    if isinstance(paragraph, str):
        return paragraph, False
    if not isinstance(paragraph, dict):
        return str(paragraph), False

    title = _first_present(paragraph, ["title", "heading", "document_title"], "")
    text = _first_present(
        paragraph,
        ["paragraph_text", "text", "content", "passage", "paragraph"],
        "",
    )
    support = bool(
        _first_present(
            paragraph,
            ["is_supporting", "supporting", "is_support", "support"],
            False,
        )
    )
    if include_titles and title:
        return f"Title: {title}\n{text}", support
    return str(text), support


def _supporting_titles_from_row(row: dict[str, Any]) -> set[str]:
    supporting_facts = row.get("supporting_facts")
    if supporting_facts is None:
        return set()
    if isinstance(supporting_facts, dict):
        titles = supporting_facts.get("title", [])
        return {str(title) for title in titles}
    if isinstance(supporting_facts, list):
        titles = []
        for item in supporting_facts:
            if isinstance(item, (list, tuple)) and item:
                titles.append(item[0])
            elif isinstance(item, dict) and "title" in item:
                titles.append(item["title"])
        return {str(title) for title in titles}
    return set()


def _hotpot_context_from_dict(
    context: dict[str, Any],
    row: dict[str, Any],
    include_titles: bool,
) -> tuple[str, list[int]] | None:
    titles = context.get("title")
    sentences = context.get("sentences")
    if not isinstance(titles, list) or not isinstance(sentences, list):
        return None

    support_titles = _supporting_titles_from_row(row)
    parts: list[str] = []
    support_indices: list[int] = []
    for idx, (title, paragraph_sentences) in enumerate(zip(titles, sentences)):
        if isinstance(paragraph_sentences, list):
            text = " ".join(str(sentence) for sentence in paragraph_sentences)
        else:
            text = str(paragraph_sentences)
        if not text:
            continue
        paragraph = f"Title: {title}\n{text}" if include_titles and title else text
        parts.append(f"[{idx}] {paragraph}")
        if str(title) in support_titles:
            support_indices.append(idx)

    return "\n\n".join(parts), support_indices


def _context_from_row(row: dict[str, Any], include_titles: bool) -> tuple[str, list[int]]:
    hotpot_context = row.get("context")
    if isinstance(hotpot_context, dict):
        parsed = _hotpot_context_from_dict(hotpot_context, row, include_titles)
        if parsed is not None:
            return parsed

    paragraphs = _first_present(row, ["paragraphs", "context", "contexts", "documents"], None)
    support_indices: list[int] = []

    if isinstance(paragraphs, list):
        parts = []
        for idx, paragraph in enumerate(paragraphs):
            text, is_support = _paragraph_text(paragraph, include_titles)
            if text:
                parts.append(f"[{idx}] {text}")
            if is_support:
                support_indices.append(idx)
        return "\n\n".join(parts), support_indices

    context = _first_present(row, ["context", "article", "input", "long_context"], "")
    if isinstance(context, dict):
        parsed = _hotpot_context_from_dict(context, row, include_titles)
        if parsed is not None:
            return parsed
        return _context_from_row({"paragraphs": list(context.values())}, include_titles)
    return str(context), support_indices


def row_to_qa_example(
    row: dict[str, Any],
    include_titles: bool = True,
    prompt_style: str = "default",
) -> QAExample:
    if "sentences" in row and "expected_output" in row:
        sentences = row["sentences"]
        context = "\n".join(str(sentence) for sentence in sentences)
        question = str(row.get("suffix") or "Repeat the above sentences:")
        qid = f"repeat-{row.get('level', '')}-{row.get('synthetic_row_id', '')}"
        return QAExample(
            qid=qid,
            question=question,
            context=context,
            answers=[str(row["expected_output"])],
            support_indices=[],
            task_type="repeat",
            prompt_style=prompt_style,
        )

    qid = str(_first_present(row, ["id", "qid", "_id"], ""))
    question = str(_first_present(row, ["question", "query"], ""))
    context, support_indices = _context_from_row(row, include_titles)
    answers = _normalize_answers(
        _first_present(row, ["answer", "answers", "answer_aliases", "aliases"], None)
    )
    return QAExample(
        qid=qid,
        question=question,
        context=context,
        answers=answers,
        support_indices=support_indices,
        task_type="qa",
        prompt_style=prompt_style,
    )


def load_qa_dataset(cfg: dict[str, Any]) -> DatasetDict:
    local_data_files = cfg.get("local_data_files")
    name = cfg.get("name")
    config_name = cfg.get("config_name")

    if local_data_files:
        raw = load_dataset("json", data_files=local_data_files)
    elif config_name:
        raw = load_dataset(name, config_name)
    else:
        raw = load_dataset(name)

    if isinstance(raw, Dataset):
        raw = DatasetDict({"train": raw})
    return raw


class QACollator:
    def __init__(self, include_titles: bool = True, prompt_style: str = "default"):
        self.include_titles = include_titles
        self.prompt_style = prompt_style

    def __call__(self, rows: list[dict[str, Any]]) -> list[QAExample]:
        return [
            row_to_qa_example(
                dict(row),
                include_titles=self.include_titles,
                prompt_style=self.prompt_style,
            )
            for row in rows
        ]


def maybe_limit(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit <= 0:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def select_range(dataset: Dataset, start_index: int = 0, limit: int | None = None) -> Dataset:
    start = max(0, int(start_index))
    end = len(dataset) if limit is None or limit <= 0 else min(len(dataset), start + int(limit))
    if start >= len(dataset):
        return dataset.select([])
    return dataset.select(range(start, end))
