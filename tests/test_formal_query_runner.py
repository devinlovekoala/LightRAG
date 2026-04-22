from __future__ import annotations

import sys

import pytest


def test_parse_args_accepts_workspace(monkeypatch):
    from reproduce import run_formal_query

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_formal_query.py",
            "--dataset",
            "mix",
            "--variant",
            "noisefilter",
            "--workspace",
            " p0_query_workspace ",
        ],
    )

    args = run_formal_query.parse_args()

    assert args.workspace == " p0_query_workspace "


@pytest.mark.asyncio
async def test_run_passes_workspace_to_runtime_overrides(monkeypatch, tmp_path):
    from reproduce import run_formal_query

    captured: dict[str, object] = {}

    class FakeRAG:
        pass

    async def fake_create_formal_rag(*, working_dir, variant_settings, runtime_overrides):
        captured["working_dir"] = working_dir
        captured["variant_settings"] = variant_settings
        captured["runtime_overrides"] = runtime_overrides
        return FakeRAG()

    async def fake_finalize_rag(_rag):
        captured["finalized"] = True

    async def fake_run_queries(*args, **kwargs):
        captured["query_kwargs"] = kwargs
        return [], []

    monkeypatch.setattr(
        run_formal_query,
        "build_formal_run_paths",
        lambda **kwargs: type(
            "Paths",
            (),
            {
                "dataset": kwargs["dataset"],
                "variant": kwargs["variant"],
                "query_mode": kwargs["query_mode"],
                "qa_file": tmp_path / "qa.json",
                "questions_file": tmp_path / "questions.txt",
                "working_dir": tmp_path / "workspace",
                "result_file": tmp_path / "result.json",
                "error_file": tmp_path / "errors.json",
            },
        )(),
    )
    monkeypatch.setattr(
        run_formal_query,
        "resolve_variant_settings",
        lambda *args, **kwargs: {"variant": "noisefilter"},
    )
    monkeypatch.setattr(
        run_formal_query,
        "build_runtime_overrides",
        lambda **kwargs: {"storage_workspace": kwargs["workspace"]},
    )
    monkeypatch.setattr(run_formal_query, "create_formal_rag", fake_create_formal_rag)
    monkeypatch.setattr(run_formal_query, "finalize_rag", fake_finalize_rag)
    monkeypatch.setattr(run_formal_query, "run_queries", fake_run_queries)
    monkeypatch.setattr(run_formal_query, "save_json_records", lambda *args, **kwargs: None)

    namespace = type(
        "Args",
        (),
        {
            "dataset": "mix",
            "variant": "noisefilter",
            "mode": "hybrid",
            "datasets_root": "datasets",
            "working_root": "rag_storage/formal_runs",
            "results_root": "reproduce/results/formal",
            "workspace": "query_workspace",
            "disable_qa": True,
            "conf_threshold": 0.3,
            "hard_filter": False,
            "w_freq": 0.5,
            "w_cons": 0.3,
            "w_sem": 0.2,
            "chunk_size": None,
            "chunk_overlap_size": None,
            "max_async": None,
            "embedding_max_async": None,
            "max_parallel_insert": None,
            "max_gleaning": None,
            "max_extract_input_tokens": None,
            "llm_timeout": None,
            "embedding_timeout": None,
            "query_concurrency": 2,
        },
    )()

    await run_formal_query._run(namespace)

    assert captured["runtime_overrides"] == {"storage_workspace": "query_workspace"}
    assert captured["query_kwargs"]["query_concurrency"] == 2
    assert captured["finalized"] is True
