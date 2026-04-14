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
    assert paths.qa_file == tmp_path / "datasets" / "qa" / "agriculture_qa.json"
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


def test_load_qa_records_accepts_list_payload(tmp_path):
    from lightrag.noisefilter.reproduction import load_qa_records

    qa_file = tmp_path / "qa.json"
    qa_file.write_text(
        """
[
  {
    "query_id": 7,
    "question": "What is LightRAG?",
    "answers": ["A graph-based RAG framework."],
    "metadata": {"domain": "cs"}
  }
]
""".strip(),
        encoding="utf-8",
    )

    records = load_qa_records(qa_file)

    assert len(records) == 1
    assert records[0].query_id == 7
    assert records[0].question == "What is LightRAG?"
    assert records[0].answers == ["A graph-based RAG framework."]
    assert records[0].metadata["domain"] == "cs"


@pytest.mark.asyncio
async def test_insert_contexts_batches_large_input(tmp_path):
    from lightrag.noisefilter.reproduction import insert_contexts

    context_file = tmp_path / "contexts.json"
    context_file.write_text(
        '["ctx1", "ctx2", "ctx3", "ctx4", "ctx5"]',
        encoding="utf-8",
    )

    class FakeRAG:
        def __init__(self):
            self.batches = []

        async def ainsert(self, batch):
            self.batches.append(list(batch))

    rag = FakeRAG()
    inserted = await insert_contexts(rag, context_file, batch_size=2, retries=1)

    assert inserted == 5
    assert rag.batches == [["ctx1", "ctx2"], ["ctx3", "ctx4"], ["ctx5"]]
