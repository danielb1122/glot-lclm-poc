from glot_lclm.data.qa_examples import row_to_qa_example


def test_hotpotqa_context_mapping_preserves_titles_and_sentences():
    row = {
        "id": "hotpot-1",
        "question": "Which person was born first?",
        "answer": "Ada Lovelace",
        "supporting_facts": {
            "title": ["Ada Lovelace", "Grace Hopper"],
            "sent_id": [0, 0],
        },
        "context": {
            "title": ["Ada Lovelace", "Distractor", "Grace Hopper"],
            "sentences": [
                ["Ada Lovelace was born in 1815.", "She worked on early computing."],
                ["This paragraph is not relevant."],
                ["Grace Hopper was born in 1906."],
            ],
        },
    }

    example = row_to_qa_example(row, prompt_style="lclm_memory")

    assert example.qid == "hotpot-1"
    assert example.question == "Which person was born first?"
    assert example.answers == ["Ada Lovelace"]
    assert "[0] Title: Ada Lovelace" in example.context
    assert "Ada Lovelace was born in 1815. She worked on early computing." in example.context
    assert "[1] Title: Distractor" in example.context
    assert "[2] Title: Grace Hopper" in example.context
    assert example.support_indices == [0, 2]
