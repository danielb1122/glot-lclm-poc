from glot_lclm.data.prompts import compressed_prompt_parts, full_context_prompt_parts
from glot_lclm.data.qa_examples import row_to_qa_example


def test_repeat_row_mapping():
    row = {
        "sentences": ["A first sentence.", "A second sentence."],
        "suffix": "Repeat the above sentences:\n",
        "expected_output": "A first sentence.\nA second sentence.",
        "level": 2,
        "synthetic_row_id": 7,
    }

    example = row_to_qa_example(row)

    assert example.task_type == "repeat"
    assert example.context == "A first sentence.\nA second sentence."
    assert example.question == "Repeat the above sentences:\n"
    assert example.answers == ["A first sentence.\nA second sentence."]


def test_repeat_prompt_does_not_use_short_answer_instruction():
    example = row_to_qa_example(
        {
            "sentences": ["A first sentence."],
            "suffix": "Repeat the above sentences:\n",
            "expected_output": "A first sentence.",
        }
    )

    parts = full_context_prompt_parts(example)

    assert "Repeat the provided context exactly." in parts.prefix
    assert "shortest answer" not in parts.prefix
    assert parts.answer == "\nA first sentence."


def test_lclm_memory_repeat_prompt():
    example = row_to_qa_example(
        {
            "sentences": ["A first sentence."],
            "suffix": "Repeat the above sentences:\n",
            "expected_output": "A first sentence.",
        },
        prompt_style="lclm_memory",
    )

    compressed = compressed_prompt_parts(example)
    full = full_context_prompt_parts(example)

    assert compressed.prefix.endswith("<|memory_start|>")
    assert compressed.suffix.startswith("<|memory_end|>")
    assert "<|memory_start|>A first sentence.<|memory_end|>" in full.prefix
