from __future__ import annotations

from reproduce.formal_pairwise_eval import (
    PairwiseExample,
    build_examples,
    build_user_prompt,
    map_winner_to_variant,
    summarize_judgments,
)


def test_build_examples_alternates_answer_order():
    baseline_rows = [
        {"query_id": 1, "query": "q1", "result": "baseline-1"},
        {"query_id": 2, "query": "q2", "result": "baseline-2"},
    ]
    noisefilter_rows = [
        {"query_id": 1, "query": "q1", "result": "noise-1"},
        {"query_id": 2, "query": "q2", "result": "noise-2"},
    ]

    examples = build_examples(baseline_rows, noisefilter_rows)

    assert examples[0].label_a == "baseline"
    assert examples[0].label_b == "noisefilter"
    assert examples[1].label_a == "noisefilter"
    assert examples[1].label_b == "baseline"


def test_build_user_prompt_falls_back_for_none_answers():
    example = PairwiseExample(
        query_id=1,
        query="What happened?",
        answer_a=None,
        answer_b="Some answer",
        label_a="baseline",
        label_b="noisefilter",
    )

    prompt = build_user_prompt(example)

    assert "Answer A:\nNone" in prompt
    assert "Answer B:\nSome answer" in prompt


def test_map_winner_to_variant_uses_answer_labels():
    example = PairwiseExample(
        query_id=2,
        query="q",
        answer_a="a",
        answer_b="b",
        label_a="noisefilter",
        label_b="baseline",
    )

    assert map_winner_to_variant("answer_a", example) == "noisefilter"
    assert map_winner_to_variant("answer_b", example) == "baseline"
    assert map_winner_to_variant("tie", example) == "tie"


def test_summarize_judgments_computes_win_rates():
    rows = [
        {
            "query_id": 1,
            "query": "q1",
            "judgment": {
                "Comprehensiveness": {"mapped_winner": "baseline"},
                "Diversity": {"mapped_winner": "noisefilter"},
                "Empowerment": {"mapped_winner": "tie"},
                "Overall Winner": {"mapped_winner": "baseline"},
            },
        },
        {
            "query_id": 2,
            "query": "q2",
            "judgment": {
                "Comprehensiveness": {"mapped_winner": "noisefilter"},
                "Diversity": {"mapped_winner": "noisefilter"},
                "Empowerment": {"mapped_winner": "tie"},
                "Overall Winner": {"mapped_winner": "noisefilter"},
            },
        },
    ]

    summary = summarize_judgments(rows)

    assert summary["total_queries"] == 2
    assert summary["criteria"]["Overall Winner"]["baseline_win_rate"] == 0.5
    assert summary["criteria"]["Overall Winner"]["noisefilter_win_rate"] == 0.5
    assert summary["criteria"]["Empowerment"]["tie_rate"] == 1.0
