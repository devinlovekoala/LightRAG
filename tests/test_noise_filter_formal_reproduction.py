from __future__ import annotations

from pathlib import Path

import pytest


def test_resolve_variant_settings_for_baseline():
    from lightrag.noisefilter.reproduction import resolve_variant_settings

    settings = resolve_variant_settings("baseline")

    assert settings.variant == "baseline"
    assert settings.enable_noise_filter is False
    assert settings.noise_filter_config == {}


def test_resolve_variant_settings_for_noisefilter():
    from lightrag.noisefilter.reproduction import resolve_variant_settings

    settings = resolve_variant_settings(
        "noisefilter",
        conf_threshold=0.4,
        soft_mode=False,
        w_freq=0.6,
        w_cons=0.2,
        w_sem=0.2,
    )

    assert settings.variant == "noisefilter"
    assert settings.enable_noise_filter is True
    assert settings.noise_filter_config["conf_threshold"] == 0.4
    assert settings.noise_filter_config["soft_mode"] is False
    assert settings.noise_filter_config["w_freq"] == 0.6


def test_build_formal_run_paths_uses_stable_layout(tmp_path):
    from lightrag.noisefilter.reproduction import build_formal_run_paths

    paths = build_formal_run_paths(
        dataset="agriculture",
        variant="baseline",
        query_mode="hybrid",
        datasets_root=tmp_path / "datasets",
        working_root=tmp_path / "rag_storage" / "formal_runs",
        results_root=tmp_path / "reproduce" / "results" / "formal",
    )

    assert paths.context_file == tmp_path / "datasets" / "unique_contexts" / "agriculture_unique_contexts.json"
    assert paths.questions_file == tmp_path / "datasets" / "questions" / "agriculture_questions.txt"
    assert paths.working_dir == tmp_path / "rag_storage" / "formal_runs" / "agriculture" / "baseline"
    assert paths.result_file == tmp_path / "reproduce" / "results" / "formal" / "agriculture" / "baseline" / "hybrid_results.json"
    assert paths.error_file == tmp_path / "reproduce" / "results" / "formal" / "agriculture" / "baseline" / "hybrid_errors.json"


def test_result_directory_is_created_when_building_paths(tmp_path):
    from lightrag.noisefilter.reproduction import build_formal_run_paths

    paths = build_formal_run_paths(
        dataset="legal",
        variant="noisefilter",
        query_mode="mix",
        datasets_root=tmp_path / "datasets",
        working_root=tmp_path / "rag_storage",
        results_root=tmp_path / "results",
    )

    assert paths.result_file.parent.exists()


def test_resolve_variant_settings_rejects_unknown_variant():
    from lightrag.noisefilter.reproduction import resolve_variant_settings

    with pytest.raises(ValueError):
        resolve_variant_settings("unknown")
