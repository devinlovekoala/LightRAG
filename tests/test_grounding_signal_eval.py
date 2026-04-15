from __future__ import annotations

import json
from pathlib import Path

import pytest

from reproduce.evaluate_grounding_signals import (
    build_hypotheses,
    build_comparison_summary,
    compute_anchor_scores,
    compute_average_precision,
    compute_roc_auc,
    evaluate_variant,
    load_edge_rows,
    resolve_local_model_path,
    resolve_source_texts,
)


def test_load_edge_rows_parses_json_fields(tmp_path: Path) -> None:
    csv_path = tmp_path / "edges.csv"
    csv_path.write_text(
        "\n".join(
            [
                "variant,src,dst,description,chunk_ids,source_chunks,manual_label",
                'noisefilter,Alice,Acme,"Alice works at Acme","[""c1"", ""c2""]","[""chunk 1""]",correct',
            ]
        ),
        encoding="utf-8",
    )

    rows = load_edge_rows(csv_path)

    assert rows[0]["chunk_ids"] == ["c1", "c2"]
    assert rows[0]["source_chunks"] == ["chunk 1"]
    assert rows[0]["manual_label"] == "correct"


def test_resolve_source_texts_prefers_text_chunk_lookup() -> None:
    row = {
        "chunk_ids": ["c1", "c2"],
        "source_chunks": ["excerpt only"],
    }
    text_chunks = {
        "c1": "full chunk 1",
        "c2": "full chunk 2",
    }

    assert resolve_source_texts(row, text_chunks) == ["full chunk 1", "full chunk 2"]


def test_compute_anchor_scores_detects_entity_mentions() -> None:
    scores = compute_anchor_scores(
        "Alice Johnson",
        "Acme Corp",
        [
            "Alice Johnson joined Acme Corp in 2020.",
            "Completely unrelated text.",
        ],
    )

    assert scores["src_anchor_score"] == pytest.approx(0.5)
    assert scores["dst_anchor_score"] == pytest.approx(0.5)
    assert scores["cooccurrence_score"] == pytest.approx(0.5)


def test_auc_helpers_handle_perfect_ranking() -> None:
    labels = [1, 1, 0, 0]
    scores = [0.9, 0.8, 0.3, 0.1]

    assert compute_roc_auc(labels, scores) == pytest.approx(1.0)
    assert compute_average_precision(labels, scores) == pytest.approx(1.0)


def test_resolve_local_model_path_prefers_existing_directory(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    assert resolve_local_model_path(str(model_dir)) == str(model_dir)


def test_build_hypotheses_splits_sep_and_adds_keyword_templates() -> None:
    hypotheses = build_hypotheses(
        {
            "src": "Rumbi Katedza",
            "dst": "CKUT Radio",
            "keywords": "radio production,presentation",
            "description": (
                "Rumbi Katedza produced radio shows for CKUT.<SEP>"
                "Rumbi Katedza presented radio shows for CKUT."
            ),
        }
    )

    assert "Rumbi Katedza produced radio shows for CKUT." in hypotheses
    assert "Rumbi Katedza presented radio shows for CKUT." in hypotheses
    assert any("radio production" in hypothesis.lower() for hypothesis in hypotheses)


def test_evaluate_variant_summarizes_signals() -> None:
    rows = [
        {
            "variant": "noisefilter",
            "src": "Alice",
            "dst": "Acme",
            "description": "Alice works at Acme.",
            "manual_label": "correct",
            "chunk_ids": ["c1"],
            "source_chunks": ["Alice works at Acme."],
        },
        {
            "variant": "noisefilter",
            "src": "Alice",
            "dst": "Beta",
            "description": "Alice founded Beta.",
            "manual_label": "wrong",
            "chunk_ids": ["c2"],
            "source_chunks": ["This passage is about a sports event."],
        },
        {
            "variant": "noisefilter",
            "src": "Alice",
            "dst": "Gamma",
            "description": "Alice advised Gamma.",
            "manual_label": "ambiguous",
            "chunk_ids": ["c3"],
            "source_chunks": ["Alice mentioned Gamma in passing."],
        },
    ]

    def fake_nli(premise: str, hypothesis: str) -> float:
        if "works at" in premise.lower() and "works at" in hypothesis.lower():
            return 0.95
        if "worked on radio shows" in premise.lower() and "worked on radio shows" in hypothesis.lower():
            return 0.75
        if "sports event" in premise.lower():
            return 0.05
        return 0.4

    summary = evaluate_variant("noisefilter", rows, nli_scorer=fake_nli)

    assert summary["total_edges"] == 3
    assert summary["label_counts"] == {"correct": 1, "wrong": 1, "ambiguous": 1}
    assert summary["signal_means_by_label"]["nli_support_score"]["correct"] > summary["signal_means_by_label"]["nli_support_score"]["wrong"]
    assert summary["binary_ranking"]["nli_support_score"]["roc_auc"] > 0.9


def test_evaluate_variant_uses_best_hypothesis_template() -> None:
    rows = [
        {
            "variant": "noisefilter",
            "src": "Rumbi Katedza",
            "dst": "CKUT Radio",
            "keywords": "radio shows",
            "description": "Rumbi Katedza produced and presented radio shows for CKUT from 1994 to 2000.",
            "manual_label": "correct",
            "chunk_ids": ["c1"],
            "source_chunks": [
                "From 1994 to 2000, she worked on radio shows for CKUT in Montreal."
            ],
        },
        {
            "variant": "noisefilter",
            "src": "Alice",
            "dst": "Beta",
            "keywords": "founded",
            "description": "Alice founded Beta.",
            "manual_label": "wrong",
            "chunk_ids": ["c2"],
            "source_chunks": ["Completely unrelated sports event."],
        },
    ]

    def fake_nli(premise: str, hypothesis: str) -> float:
        if "radio shows" in premise.lower() and "radio shows" in hypothesis.lower():
            return 0.8
        if "produced and presented" in hypothesis.lower():
            return 0.05
        return 0.01

    summary = evaluate_variant("noisefilter", rows, nli_scorer=fake_nli)

    correct_row = next(
        row for row in summary["evaluated_rows"] if row["manual_label"] == "correct"
    )
    assert correct_row["nli_support_score"] == pytest.approx(0.8)


def test_evaluate_variant_uses_batch_nli_when_available() -> None:
    rows = [
        {
            "variant": "noisefilter",
            "src": "Alice",
            "dst": "Acme",
            "keywords": "employment",
            "description": "Alice works at Acme.",
            "manual_label": "correct",
            "chunk_ids": ["c1"],
            "source_chunks": ["Alice works at Acme."],
        },
        {
            "variant": "noisefilter",
            "src": "Alice",
            "dst": "Beta",
            "keywords": "founded",
            "description": "Alice founded Beta.",
            "manual_label": "wrong",
            "chunk_ids": ["c2"],
            "source_chunks": ["Sports event only."],
        },
    ]

    class FakeBatchScorer:
        def score_many(self, pairs):
            return [0.9 if "works at" in hypothesis.lower() else 0.1 for _, hypothesis in pairs]

    summary = evaluate_variant("noisefilter", rows, nli_scorer=FakeBatchScorer())

    assert summary["binary_ranking"]["nli_support_score"]["roc_auc"] == pytest.approx(1.0)


def test_build_comparison_summary_uses_signal_deltas() -> None:
    baseline = {
        "strict_precision": 0.15,
        "lenient_precision": 0.50,
        "binary_ranking": {
            "nli_support_score": {"roc_auc": 0.55, "average_precision": 0.30}
        },
    }
    noisefilter = {
        "strict_precision": 0.22,
        "lenient_precision": 0.57,
        "binary_ranking": {
            "nli_support_score": {"roc_auc": 0.82, "average_precision": 0.61}
        },
    }

    comparison = build_comparison_summary(
        {"baseline": baseline, "noisefilter": noisefilter}
    )

    assert comparison["precision_delta"]["strict"] == pytest.approx(0.07)
    assert comparison["ranking_delta"]["nli_support_score"]["roc_auc"] == pytest.approx(
        0.27
    )
