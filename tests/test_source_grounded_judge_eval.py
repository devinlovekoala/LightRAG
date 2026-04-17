from __future__ import annotations

import pytest

from reproduce.evaluate_source_grounded_judge import (
    JUDGE_VERDICTS,
    SOURCE_MODES,
    build_claim,
    build_comparison_summary,
    calibrate_judge_result,
    evaluate_variant_async,
    prepare_source_texts,
    resolve_judge_source_texts,
    score_verdict,
    summarize_variant_rows,
)


def test_build_claim_prefers_description_then_keywords() -> None:
    assert (
        build_claim(
            {
                "src": "Alice",
                "dst": "Acme",
                "description": "Alice works at Acme.",
                "keywords": "employment",
            }
        )
        == "Alice works at Acme."
    )
    assert (
        build_claim({"src": "Alice", "dst": "Acme", "description": "", "keywords": "employment"})
        == "Alice employment Acme."
    )


def test_score_verdict_orders_supported_above_partial_and_rejected() -> None:
    assert score_verdict("supported") > score_verdict("partially_supported")
    assert score_verdict("partially_supported") > score_verdict("not_supported")
    assert score_verdict("contradicted") == 0.0


def test_source_modes_enum_is_stable() -> None:
    assert SOURCE_MODES == ("auto", "exported", "resolved")


def test_prepare_source_texts_limits_chunk_count_and_char_budget() -> None:
    prepared = prepare_source_texts(
        [
            "Alice joined Acme in 2020. " * 10,
            "Bob founded Beta in 2021. " * 10,
            "Carol advised Gamma in 2022. " * 10,
        ],
        max_source_chunks=2,
        max_source_chars=120,
    )

    assert len(prepared) == 1
    assert len(prepared[0]) <= 120
    assert "[...truncated]" in prepared[0]


def test_resolve_judge_source_texts_respects_source_mode() -> None:
    row = {
        "src": "Alice",
        "dst": "Acme",
        "chunk_ids": ["c1"],
        "source_chunks": ["exported preview"],
    }
    lookup = {"c1": "resolved full chunk"}

    assert resolve_judge_source_texts(row, text_chunk_lookup=lookup, source_mode="exported") == [
        "exported preview"
    ]
    assert resolve_judge_source_texts(row, text_chunk_lookup=lookup, source_mode="resolved") == [
        "resolved full chunk"
    ]
    assert resolve_judge_source_texts(row, text_chunk_lookup=lookup, source_mode="auto") == [
        "resolved full chunk"
    ]
def test_calibrate_judge_result_downgrades_unsupported_supported_verdict() -> None:
    calibrated = calibrate_judge_result(
        "supported",
        0.99,
        "The claim seems related.",
        [],
        ["A sports event recap with no relevant relation."],
    )

    assert calibrated["verdict"] == "partially_supported"
    assert calibrated["support_score"] <= 0.6
    assert calibrated["anchored_evidence"] is False


@pytest.mark.asyncio
async def test_evaluate_variant_async_summarizes_judge_scores() -> None:
    rows = [
        {
            "variant": "noisefilter",
            "src": "Alice",
            "dst": "Acme",
            "description": "Alice works at Acme.",
            "manual_label": "correct",
            "source_chunks": ["Alice joined Acme in 2020."],
            "chunk_ids": ["c1"],
        },
        {
            "variant": "noisefilter",
            "src": "Alice",
            "dst": "Beta",
            "description": "Alice founded Beta.",
            "manual_label": "wrong",
            "source_chunks": ["A sports event recap."],
            "chunk_ids": ["c2"],
        },
        {
            "variant": "noisefilter",
            "src": "Alice",
            "dst": "Gamma",
            "description": "Alice advised Gamma.",
            "manual_label": "ambiguous",
            "source_chunks": ["Gamma was mentioned alongside Alice."],
            "chunk_ids": ["c3"],
        },
    ]

    async def fake_judge(claim: str, source_texts: list[str]) -> dict[str, str | float]:
        if "works at" in claim.lower():
            return {
                "verdict": "supported",
                "support_score": 0.95,
                "explanation": "directly supported",
                "evidence": ["Alice joined Acme in 2020."],
                "anchored_evidence": True,
            }
        if "sports event" in source_texts[0].lower():
            return {
                "verdict": "not_supported",
                "support_score": 0.05,
                "explanation": "irrelevant source",
                "evidence": [],
                "anchored_evidence": False,
            }
        return {
            "verdict": "partially_supported",
            "support_score": 0.55,
            "explanation": "partial overlap",
            "evidence": [],
            "anchored_evidence": False,
        }

    summary = await evaluate_variant_async(
        "noisefilter",
        rows,
        judge_func=fake_judge,
        concurrency=2,
    )

    assert summary["label_counts"] == {"correct": 1, "wrong": 1, "ambiguous": 1}
    assert summary["judge_score_means_by_label"]["correct"] > summary["judge_score_means_by_label"]["wrong"]
    assert summary["binary_ranking"]["roc_auc"] == pytest.approx(1.0)
    assert summary["verdict_counts"]["supported"] == 1
    assert summary["rows"][0]["judge_anchored_evidence"] is True


@pytest.mark.asyncio
async def test_evaluate_variant_async_can_force_exported_source_mode() -> None:
    rows = [
        {
            "variant": "baseline",
            "src": "Alice",
            "dst": "Acme",
            "description": "Alice works at Acme.",
            "manual_label": "correct",
            "source_chunks": ["exported preview only"],
            "chunk_ids": ["c1"],
        }
    ]

    async def fake_judge(claim: str, source_texts: list[str]) -> dict[str, str | float]:
        assert source_texts == ["exported preview only"]
        return {
            "verdict": "partially_supported",
            "support_score": 0.55,
            "explanation": "preview path used",
            "evidence": [],
            "anchored_evidence": False,
        }

    summary = await evaluate_variant_async(
        "baseline",
        rows,
        judge_func=fake_judge,
        text_chunk_lookup={"c1": "resolved full chunk"},
        concurrency=1,
        source_mode="exported",
    )

    assert summary["rows"][0]["judge_verdict"] == "partially_supported"


def test_summarize_variant_rows_aggregates_verdicts() -> None:
    summary = summarize_variant_rows(
        "baseline",
        [
            {"manual_label": "correct", "judge_verdict": "supported", "judge_support_score": 0.9},
            {"manual_label": "wrong", "judge_verdict": "not_supported", "judge_support_score": 0.1},
            {"manual_label": "ambiguous", "judge_verdict": "partially_supported", "judge_support_score": 0.5},
        ],
    )

    assert summary["verdict_counts"]["supported"] == 1
    assert summary["judge_score_means_by_label"]["ambiguous"] == pytest.approx(0.5)


def test_build_comparison_summary_reports_auc_delta() -> None:
    comparison = build_comparison_summary(
        {
            "baseline": {"strict_precision": 0.15, "lenient_precision": 0.50, "binary_ranking": {"roc_auc": 0.61, "average_precision": 0.28}},
            "noisefilter": {"strict_precision": 0.22, "lenient_precision": 0.57, "binary_ranking": {"roc_auc": 0.78, "average_precision": 0.52}},
        }
    )

    assert comparison["precision_delta"]["strict"] == pytest.approx(0.07)
    assert comparison["ranking_delta"]["roc_auc"] == pytest.approx(0.17)


def test_supported_verdict_enum_is_stable() -> None:
    assert JUDGE_VERDICTS == (
        "supported",
        "partially_supported",
        "not_supported",
        "contradicted",
    )
