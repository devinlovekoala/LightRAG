from __future__ import annotations

from pathlib import Path

import pytest

from reproduce.analyze_labeled_edge_samples import (
    build_comparison_summary,
    load_labeled_rows,
    save_canonical_csv,
    save_markdown,
    summarize_variant,
)


def test_load_labeled_rows_normalizes_metrics_and_labels(tmp_path: Path) -> None:
    csv_path = tmp_path / "labeled.csv"
    csv_path.write_text(
        "\n".join(
            [
                "variant,src,dst,conf_score,conf_freq_score,conf_consistency_score,conf_semantic_score,manual_label,manual_notes",
                "baseline,A,B,0.1,0.2,0.3,0.4,Correct,good edge",
                "baseline,C,D,,0.0,,0.2, wrong ,bad edge",
            ]
        ),
        encoding="utf-8",
    )

    rows = load_labeled_rows(csv_path)

    assert len(rows) == 2
    assert rows[0]["manual_label"] == "correct"
    assert rows[0]["conf_score"] == 0.1
    assert rows[1]["manual_label"] == "wrong"
    assert rows[1]["conf_score"] == 0.0
    assert rows[1]["conf_consistency_score"] == 0.0


def test_load_labeled_rows_accepts_annotation_alias_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "annotated.csv"
    csv_path.write_text(
        "\n".join(
            [
                "#,src（源实体）,dst（目标实体）,relation description（关系描述）,conf_score,variant,manual taxonomy,manual_label,notes / 备注",
                "1,Alice,Acme,Alice works at Acme.,0.42,,A. unsupported_relation,Wrong,not supported",
            ]
        ),
        encoding="utf-8",
    )

    rows = load_labeled_rows(csv_path, default_variant="baseline")

    assert rows == [
        {
            "variant": "baseline",
            "src": "Alice",
            "dst": "Acme",
            "description": "Alice works at Acme.",
            "keywords": "",
            "weight": 0.0,
            "conf_score": 0.42,
            "conf_freq_score": 0.0,
            "conf_consistency_score": 0.0,
            "conf_semantic_score": 0.0,
            "conf_support": 0.0,
            "chunk_ids": "",
            "source_chunks": "",
            "taxonomy": "A. unsupported_relation",
            "manual_label": "wrong",
            "manual_notes": "not supported",
        }
    ]


def test_summarize_variant_reports_precision_and_signal_direction() -> None:
    rows = [
        {
            "variant": "noisefilter",
            "conf_score": 0.8,
            "conf_freq_score": 0.7,
            "conf_consistency_score": 0.9,
            "conf_semantic_score": 0.6,
            "manual_label": "correct",
        },
        {
            "variant": "noisefilter",
            "conf_score": 0.6,
            "conf_freq_score": 0.7,
            "conf_consistency_score": 0.8,
            "conf_semantic_score": 0.4,
            "manual_label": "ambiguous",
        },
        {
            "variant": "noisefilter",
            "conf_score": 0.2,
            "conf_freq_score": 0.1,
            "conf_consistency_score": 0.3,
            "conf_semantic_score": 0.1,
            "manual_label": "wrong",
        },
    ]

    summary = summarize_variant("noisefilter", rows)

    assert summary["total_edges"] == 3
    assert summary["label_counts"] == {"correct": 1, "wrong": 1, "ambiguous": 1}
    assert summary["strict_precision"] == 1 / 3
    assert summary["lenient_precision"] == 2 / 3
    assert summary["wrong_rate"] == 1 / 3
    assert summary["signal_diagnostics"]["conf_score"]["status"] == "discriminative"
    assert summary["signal_diagnostics"]["conf_score"][
        "correct_minus_wrong_mean"
    ] == pytest.approx(0.6)


def test_build_comparison_summary_computes_precision_delta() -> None:
    baseline = summarize_variant(
        "baseline",
        [
            {
                "variant": "baseline",
                "conf_score": 0.0,
                "conf_freq_score": 0.0,
                "conf_consistency_score": 0.0,
                "conf_semantic_score": 0.0,
                "manual_label": "correct",
            },
            {
                "variant": "baseline",
                "conf_score": 0.0,
                "conf_freq_score": 0.0,
                "conf_consistency_score": 0.0,
                "conf_semantic_score": 0.0,
                "manual_label": "wrong",
            },
        ],
    )
    noisefilter = summarize_variant(
        "noisefilter",
        [
            {
                "variant": "noisefilter",
                "conf_score": 0.7,
                "conf_freq_score": 0.5,
                "conf_consistency_score": 1.0,
                "conf_semantic_score": 0.2,
                "manual_label": "correct",
            },
            {
                "variant": "noisefilter",
                "conf_score": 0.3,
                "conf_freq_score": 0.5,
                "conf_consistency_score": 1.0,
                "conf_semantic_score": 0.1,
                "manual_label": "wrong",
            },
        ],
    )

    comparison = build_comparison_summary(
        {
            "baseline": baseline,
            "noisefilter": noisefilter,
        }
    )

    assert comparison["precision_delta"]["strict"] == 0.0
    assert comparison["precision_delta"]["lenient"] == 0.0
    assert comparison["precision_delta"]["wrong_rate"] == 0.0
    assert comparison["signal_delta"]["conf_score"] == pytest.approx(0.4)


def test_save_markdown_includes_comparison_table(tmp_path: Path) -> None:
    report = {
        "variants": {
            "baseline": {
                "total_edges": 100,
                "label_counts": {"correct": 15, "wrong": 50, "ambiguous": 35},
                "strict_precision": 0.15,
                "lenient_precision": 0.5,
                "wrong_rate": 0.5,
                "taxonomy_counts": {"A. unsupported_relation": 1},
                "metric_means_by_label": {},
                "signal_diagnostics": {},
            },
            "noisefilter": {
                "total_edges": 100,
                "label_counts": {"correct": 22, "wrong": 43, "ambiguous": 35},
                "strict_precision": 0.22,
                "lenient_precision": 0.57,
                "wrong_rate": 0.43,
                "taxonomy_counts": {"E. cross_chunk_hallucination": 1},
                "metric_means_by_label": {},
                "signal_diagnostics": {},
            },
        },
        "comparison": {
            "precision_delta": {"strict": 0.07, "lenient": 0.07, "wrong_rate": -0.07},
            "signal_delta": {"conf_score": -0.004},
        },
    }

    output_path = tmp_path / "report.md"
    save_markdown(output_path, report)

    content = output_path.read_text(encoding="utf-8")
    assert "# Labeled Edge Sample Analysis" in content
    assert "| baseline | 100 | 15 | 50 | 35 | 0.1500 | 0.5000 |" in content
    assert "| noisefilter | 100 | 22 | 43 | 35 | 0.2200 | 0.5700 |" in content
    assert "Strict precision delta (noisefilter - baseline): 0.0700" in content
    assert "Wrong-rate delta (noisefilter - baseline): -0.0700" in content
    assert "| E. cross_chunk_hallucination | 1 |" in content


def test_save_canonical_csv_writes_stable_columns(tmp_path: Path) -> None:
    output_path = tmp_path / "canonical.csv"

    save_canonical_csv(
        output_path,
        [
            {
                "variant": "baseline",
                "src": "Alice",
                "dst": "Acme",
                "description": "Alice works at Acme.",
                "manual_label": "correct",
            }
        ],
    )

    content = output_path.read_text(encoding="utf-8")
    assert content.startswith("variant,src,dst,description,keywords")
    assert "baseline,Alice,Acme,Alice works at Acme." in content
