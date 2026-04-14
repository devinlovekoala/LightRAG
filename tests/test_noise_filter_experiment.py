from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_parse_thresholds_sorts_and_deduplicates():
    from lightrag.noisefilter.experiment import parse_thresholds

    assert parse_thresholds("0.3, 0.1,0.3,0.2") == [0.1, 0.2, 0.3]
    assert parse_thresholds([0.4, 0.2, 0.4]) == [0.2, 0.4]


def test_summarize_experiment_rows_picks_best_strategies():
    from lightrag.noisefilter.experiment import summarize_experiment_rows

    rows = [
        {
            "injector": "random",
            "noise_ratio": 0.2,
            "strategy": "hard",
            "threshold": 0.2,
            "filter_f1": 0.5,
            "retrieval_noise_rate_at_k": 0.25,
        },
        {
            "injector": "random",
            "noise_ratio": 0.2,
            "strategy": "hard",
            "threshold": 0.4,
            "filter_f1": 0.8,
            "retrieval_noise_rate_at_k": 0.1,
        },
        {
            "injector": "random",
            "noise_ratio": 0.2,
            "strategy": "soft",
            "threshold": None,
            "filter_f1": 0.0,
            "retrieval_noise_rate_at_k": 0.12,
        },
    ]

    summary = summarize_experiment_rows(rows)

    assert summary["best_hard_by_f1"]["random@0.20"]["threshold"] == 0.4
    assert summary["best_overall_retrieval"]["random@0.20"]["strategy"] == "hard"


def test_write_experiment_bundle_outputs_json_csv_and_markdown(tmp_path):
    from lightrag.noisefilter.experiment import write_experiment_bundle

    rows = [
        {
            "injector": "random",
            "noise_ratio": 0.2,
            "strategy": "hard",
            "threshold": 0.3,
            "filter_f1": 0.8,
            "retrieval_noise_rate_at_k": 0.1,
        }
    ]
    summary = {
        "best_hard_by_f1": {
            "random@0.20": {
                "strategy": "hard",
                "threshold": 0.3,
                "filter_f1": 0.8,
            }
        },
        "best_overall_retrieval": {
            "random@0.20": {
                "strategy": "hard",
                "threshold": 0.3,
                "retrieval_noise_rate_at_k": 0.1,
            }
        },
    }
    config = {"noise_ratios": [0.2], "thresholds": [0.3], "top_k": 5}

    outputs = write_experiment_bundle(tmp_path, rows, summary, config, stem="demo")

    assert outputs["json"].exists()
    assert outputs["csv"].exists()
    assert outputs["markdown"].exists()

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["summary"]["best_hard_by_f1"]["random@0.20"]["threshold"] == 0.3
    assert "retrieval_noise_rate_at_k" in outputs["csv"].read_text(encoding="utf-8")
    assert "NoiseFilter-RAG Experiment Report" in outputs["markdown"].read_text(
        encoding="utf-8"
    )
