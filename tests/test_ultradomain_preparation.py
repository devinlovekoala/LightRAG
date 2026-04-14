from __future__ import annotations

from reproduce.evaluate_formal_results import evaluate_rows, normalize_answer, token_f1_score
from reproduce.prepare_ultradomain import (
    build_questions_text,
    extract_ultradomain_artifacts,
    normalize_domains,
)


def test_normalize_domains_discards_empty_entries():
    assert normalize_domains("mix, agriculture, ,cs") == ["mix", "agriculture", "cs"]


def test_extract_ultradomain_artifacts_builds_unique_contexts_and_qa():
    rows = [
        {
            "input": "Question A",
            "answers": ["Answer A"],
            "context": "Context One",
        },
        {
            "input": "Question B",
            "answers": ["Answer B1", "Answer B2"],
            "context": "Context One",
        },
        {
            "input": "Question C",
            "answers": ["Answer C"],
            "context": "Context Two",
        },
    ]

    unique_contexts, qa_records = extract_ultradomain_artifacts(
        rows,
        source_name="mix.jsonl",
    )

    assert unique_contexts == ["Context One", "Context Two"]
    assert [record["query_id"] for record in qa_records] == [1, 2, 3]
    assert qa_records[1]["answers"] == ["Answer B1", "Answer B2"]
    assert qa_records[2]["metadata"]["source_file"] == "mix.jsonl"


def test_build_questions_text_uses_expected_format():
    text = build_questions_text(
        [
            {"query_id": 1, "question": "Question A"},
            {"query_id": 2, "question": "Question B"},
        ]
    )

    assert text == "- Question 1: Question A\n- Question 2: Question B\n"


def test_normalize_answer_and_token_f1_are_stable():
    assert normalize_answer("The LightRAG, system!") == "lightrag system"
    assert token_f1_score("LightRAG system", "the LightRAG system") == 1.0


def test_evaluate_rows_scores_best_gold_answer():
    result_rows = [
        {
            "query_id": 1,
            "query": "What is LightRAG?",
            "result": "A graph-based RAG framework",
            "ground_truth_answers": [
                "A graph-based RAG framework",
                "A lightweight graph RAG framework",
            ],
        }
    ]
    error_rows = []

    per_query, summary = evaluate_rows(result_rows, error_rows)

    assert len(per_query) == 1
    assert per_query[0]["exact_match"] == 1.0
    assert per_query[0]["token_f1"] == 1.0
    assert summary["successful_queries"] == 1
    assert summary["avg_exact_match"] == 1.0
