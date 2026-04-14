from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from dotenv import load_dotenv

if TYPE_CHECKING:
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc


load_dotenv(dotenv_path=".env", override=False)


@dataclass(slots=True)
class VariantSettings:
    variant: str
    enable_noise_filter: bool
    noise_filter_config: dict[str, Any]


@dataclass(slots=True)
class FormalRunPaths:
    dataset: str
    variant: str
    query_mode: str
    datasets_root: Path
    working_root: Path
    results_root: Path
    context_file: Path
    questions_file: Path
    qa_file: Path
    working_dir: Path
    result_file: Path
    error_file: Path


@dataclass(slots=True)
class QARecord:
    query_id: int
    question: str
    answers: list[str]
    metadata: dict[str, Any]


SUPPORTED_VARIANTS = {"baseline", "noisefilter"}
SUPPORTED_QUERY_MODES = {"local", "global", "hybrid", "mix", "naive"}


def _normalize_variant(variant: str) -> str:
    normalized = variant.strip().lower()
    if normalized not in SUPPORTED_VARIANTS:
        raise ValueError(
            f"Unsupported variant: {variant}. Expected one of {sorted(SUPPORTED_VARIANTS)}."
        )
    return normalized


def _normalize_query_mode(query_mode: str) -> str:
    normalized = query_mode.strip().lower()
    if normalized not in SUPPORTED_QUERY_MODES:
        raise ValueError(
            f"Unsupported query mode: {query_mode}. Expected one of {sorted(SUPPORTED_QUERY_MODES)}."
        )
    return normalized


def resolve_variant_settings(
    variant: str,
    *,
    conf_threshold: float = 0.3,
    soft_mode: bool = True,
    w_freq: float = 0.5,
    w_cons: float = 0.3,
    w_sem: float = 0.2,
    default_conf_score: float = 0.5,
) -> VariantSettings:
    normalized = _normalize_variant(variant)
    if normalized == "baseline":
        return VariantSettings(
            variant="baseline",
            enable_noise_filter=False,
            noise_filter_config={},
        )
    if normalized == "noisefilter":
        return VariantSettings(
            variant="noisefilter",
            enable_noise_filter=True,
            noise_filter_config={
                "conf_threshold": conf_threshold,
                "soft_mode": soft_mode,
                "w_freq": w_freq,
                "w_cons": w_cons,
                "w_sem": w_sem,
                "default_conf_score": default_conf_score,
            },
        )
    raise ValueError(f"Unsupported variant: {variant}")


def build_formal_run_paths(
    *,
    dataset: str,
    variant: str,
    query_mode: str,
    datasets_root: str | Path = "datasets",
    working_root: str | Path = "rag_storage/formal_runs",
    results_root: str | Path = "reproduce/results/formal",
) -> FormalRunPaths:
    dataset_name = dataset.strip().lower()
    variant_name = _normalize_variant(variant)
    query_mode_name = _normalize_query_mode(query_mode)

    datasets_root_path = Path(datasets_root)
    working_root_path = Path(working_root)
    results_root_path = Path(results_root)

    working_dir = working_root_path / dataset_name / variant_name
    result_dir = results_root_path / dataset_name / variant_name
    result_dir.mkdir(parents=True, exist_ok=True)

    return FormalRunPaths(
        dataset=dataset_name,
        variant=variant_name,
        query_mode=query_mode_name,
        datasets_root=datasets_root_path,
        working_root=working_root_path,
        results_root=results_root_path,
        context_file=datasets_root_path
        / "unique_contexts"
        / f"{dataset_name}_unique_contexts.json",
        questions_file=datasets_root_path
        / "questions"
        / f"{dataset_name}_questions.txt",
        qa_file=datasets_root_path / "qa" / f"{dataset_name}_qa.json",
        working_dir=working_dir,
        result_file=result_dir / f"{query_mode_name}_results.json",
        error_file=result_dir / f"{query_mode_name}_errors.json",
    )


def load_unique_contexts(context_file: str | Path) -> list[str]:
    path = Path(context_file)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, list) or any(not isinstance(item, str) for item in payload):
        raise ValueError(f"Expected a JSON array of strings in {path}")
    return payload


def extract_queries(questions_file: str | Path) -> list[str]:
    path = Path(questions_file)
    with path.open("r", encoding="utf-8") as file:
        data = file.read().replace("**", "")
    return [query.strip() for query in re.findall(r"- Question \d+: (.+)", data)]


def load_qa_records(qa_file: str | Path) -> list[QARecord]:
    path = Path(qa_file)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if isinstance(payload, dict):
        rows = payload.get("records", [])
    else:
        rows = payload

    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON array of QA records in {path}")

    records: list[QARecord] = []
    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"QA row {row_index} in {path} must be an object")

        question = str(row.get("question", "")).strip()
        answers = row.get("answers", [])
        if not question:
            raise ValueError(f"QA row {row_index} in {path} is missing question")
        if not isinstance(answers, list) or any(
            not isinstance(answer, str) or not answer.strip() for answer in answers
        ):
            raise ValueError(
                f"QA row {row_index} in {path} must contain a non-empty answers list"
            )

        query_id = row.get("query_id", row_index)
        try:
            normalized_query_id = int(query_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"QA row {row_index} in {path} has invalid query_id: {query_id}"
            ) from exc

        metadata = row.get("metadata", {})
        if not isinstance(metadata, dict):
            raise ValueError(f"QA row {row_index} in {path} metadata must be an object")

        records.append(
            QARecord(
                query_id=normalized_query_id,
                question=question,
                answers=[answer.strip() for answer in answers],
                metadata=metadata,
            )
        )

    return records


def save_json_records(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _build_llm_model_func():
    from lightrag.llm.openai import openai_complete_if_cache

    model = _get_env("LLM_MODEL", "gpt-4o-mini")
    api_key = _get_env("LLM_BINDING_API_KEY") or _get_env("OPENAI_API_KEY")
    base_url = _get_env("LLM_BINDING_HOST")

    async def llm_model_func(
        prompt, system_prompt=None, history_messages=None, **kwargs
    ) -> str:
        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )

    return llm_model_func


def _build_embedding_func() -> EmbeddingFunc:
    from lightrag.llm.openai import openai_embed
    from lightrag.utils import EmbeddingFunc

    embedding_model = _get_env("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_base_url = _get_env("EMBEDDING_BINDING_HOST")
    embedding_api_key = _get_env("EMBEDDING_BINDING_API_KEY") or _get_env(
        "OPENAI_API_KEY"
    )
    embedding_dim = int(_get_env("EMBEDDING_DIM", "1536"))
    max_embed_tokens = int(
        _get_env("MAX_EMBED_TOKENS")
        or _get_env("EMBEDDING_TOKEN_LIMIT")
        or "8192"
    )

    return EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_embed_tokens,
        func=partial(
            openai_embed.func,
            model=embedding_model,
            base_url=embedding_base_url,
            api_key=embedding_api_key,
        ),
    )


async def create_formal_rag(
    *,
    working_dir: str | Path,
    variant_settings: VariantSettings,
) -> LightRAG:
    from lightrag import LightRAG

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=_build_llm_model_func(),
        embedding_func=_build_embedding_func(),
        enable_noise_filter=variant_settings.enable_noise_filter,
        noise_filter_config=variant_settings.noise_filter_config,
    )
    await rag.initialize_storages()
    return rag


async def insert_contexts(
    rag: LightRAG,
    context_file: str | Path,
    *,
    batch_size: int = 50,
    retries: int = 3,
    retry_delay_seconds: float = 10.0,
) -> int:
    unique_contexts = load_unique_contexts(context_file)

    if batch_size <= 0:
        batches = [unique_contexts]
    else:
        batches = [
            unique_contexts[index : index + batch_size]
            for index in range(0, len(unique_contexts), batch_size)
        ]

    for batch in batches:
        attempt = 0
        while attempt < retries:
            try:
                await rag.ainsert(batch)
                break
            except Exception:
                attempt += 1
                if attempt >= retries:
                    raise
                await asyncio.sleep(retry_delay_seconds)

    return len(unique_contexts)


async def run_queries(
    rag: LightRAG,
    *,
    questions_file: str | Path,
    query_mode: str,
    qa_records: list[QARecord] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from lightrag import QueryParam

    if qa_records is None:
        queries = extract_queries(questions_file)
        query_items: list[tuple[int | None, str, list[str], dict[str, Any]]] = [
            (None, query, [], {}) for query in queries
        ]
    else:
        query_items = [
            (record.query_id, record.question, record.answers, record.metadata)
            for record in qa_records
        ]

    query_param = QueryParam(mode=query_mode)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for query_id, query, answers, metadata in query_items:
        try:
            answer = await rag.aquery(query, param=query_param)
            row = {
                "query": query,
                "mode": query_mode,
                "result": answer,
            }
            if query_id is not None:
                row["query_id"] = query_id
            if answers:
                row["ground_truth_answers"] = answers
            if metadata:
                row["metadata"] = metadata
            results.append(row)
        except Exception as exc:
            row = {
                "query": query,
                "mode": query_mode,
                "error": str(exc),
            }
            if query_id is not None:
                row["query_id"] = query_id
            if answers:
                row["ground_truth_answers"] = answers
            if metadata:
                row["metadata"] = metadata
            errors.append(row)

    return results, errors


async def finalize_rag(rag: LightRAG | None) -> None:
    if rag is not None:
        await rag.finalize_storages()
