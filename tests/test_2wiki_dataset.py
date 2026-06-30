from glot_lclm.data.qa_examples import row_to_qa_example


def test_2wiki_context_mapping_uses_hotpot_style_schema():
    row = {
        "id": "2wiki-1",
        "question": "Are the two directors from the same country?",
        "answer": "no",
        "type": "bridge_comparison",
        "evidences": [
            ["Move (1970 film)", "director", "Stuart Rosenberg"],
            ["Méditerranée (1963 film)", "director", "Jean-Daniel Pollet"],
        ],
        "supporting_facts": {
            "title": ["Move (1970 film)", "Stuart Rosenberg"],
            "sent_id": [0, 0],
        },
        "context": {
            "title": ["Stuart Rosenberg", "Distractor", "Move (1970 film)"],
            "sentences": [
                ["Stuart Rosenberg was an American director."],
                ["This paragraph is not useful."],
                ["Move is a 1970 film directed by Stuart Rosenberg."],
            ],
        },
    }

    example = row_to_qa_example(row, prompt_style="lclm_memory")

    assert example.qid == "2wiki-1"
    assert example.question == "Are the two directors from the same country?"
    assert example.answers == ["no"]
    assert "[0] Title: Stuart Rosenberg" in example.context
    assert "[1] Title: Distractor" in example.context
    assert "[2] Title: Move (1970 film)" in example.context
    assert example.support_indices == [0, 2]
