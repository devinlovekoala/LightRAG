from __future__ import annotations

import asyncio
import json
import os
import re
import threading
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

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


@dataclass(slots=True)
class RoundRobinValues:
    values: list[str]
    _index: int = 0
    _lock: threading.Lock | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def next(self) -> str:
        if not self.values:
            raise ValueError("RoundRobinValues requires at least one value")
        assert self._lock is not None
        with self._lock:
            value = self.values[self._index]
            self._index = (self._index + 1) % len(self.values)
            return value


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


def build_runtime_overrides(
    *,
    chunk_size: int | None = None,
    chunk_overlap_size: int | None = None,
    llm_max_async: int | None = None,
    embedding_max_async: int | None = None,
    max_parallel_insert: int | None = None,
    max_gleaning: int | None = None,
    max_extract_input_tokens: int | None = None,
) -> dict[str, Any]:
    mappings = {
        "chunk_token_size": chunk_size,
        "chunk_overlap_token_size": chunk_overlap_size,
        "llm_model_max_async": llm_max_async,
        "embedding_func_max_async": embedding_max_async,
        "max_parallel_insert": max_parallel_insert,
        "entity_extract_max_gleaning": max_gleaning,
        "max_extract_input_tokens": max_extract_input_tokens,
    }

    overrides: dict[str, Any] = {}
    for key, value in mappings.items():
        if value is None:
            continue
        min_allowed = 1
        if key in {"chunk_overlap_token_size", "entity_extract_max_gleaning"}:
            min_allowed = 0
        if value < min_allowed:
            comparator = ">=" if min_allowed == 0 else ">"
            raise ValueError(
                f"{key} must be {comparator} {min_allowed} when provided, got {value}"
            )
        overrides[key] = value

    return overrides


def _get_env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _get_env_list(name: str) -> list[str]:
    raw = os.getenv(name, "")
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _build_api_key_selector(
    *,
    keys_env: str,
    single_key: str | None,
) -> tuple[Callable[[], str | None], int]:
    keys = _get_env_list(keys_env)
    if not keys and single_key:
        keys = [single_key]
    if not keys:
        return lambda: None, 0
    pool = RoundRobinValues(keys)
    return pool.next, len(keys)


def _build_llm_model_func():
    from lightrag.llm.openai import openai_complete_if_cache

    model = _get_env("LLM_MODEL", "gpt-4o-mini")
    base_url = _get_env("LLM_BINDING_HOST")
    get_api_key, key_count = _build_api_key_selector(
        keys_env="LLM_BINDING_API_KEYS",
        single_key=_get_env("LLM_BINDING_API_KEY") or _get_env("OPENAI_API_KEY"),
    )

    async def llm_model_func(
        prompt, system_prompt=None, history_messages=None, **kwargs
    ) -> str:
        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=get_api_key(),
            base_url=base_url,
            **kwargs,
        )

    setattr(llm_model_func, "_api_key_count", key_count)
    return llm_model_func


def _build_embedding_func() -> EmbeddingFunc:
    from lightrag.llm.openai import openai_embed
    from lightrag.utils import EmbeddingFunc

    embedding_model = _get_env("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_base_url = _get_env("EMBEDDING_BINDING_HOST")
    get_api_key, key_count = _build_api_key_selector(
        keys_env="EMBEDDING_BINDING_API_KEYS",
        single_key=_get_env("EMBEDDING_BINDING_API_KEY")
        or _get_env("OPENAI_API_KEY"),
    )
    embedding_dim = int(_get_env("EMBEDDING_DIM", "1536"))
    max_embed_tokens = int(
        _get_env("MAX_EMBED_TOKENS")
        or _get_env("EMBEDDING_TOKEN_LIMIT")
        or "8192"
    )

    async def embedding_func(texts: list[str], **kwargs: Any):
        return await openai_embed.func(
            texts,
            model=embedding_model,
            base_url=embedding_base_url,
            api_key=get_api_key(),
            **kwargs,
        )

    embedding_func_wrapper = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=max_embed_tokens,
        func=embedding_func,
    )
    setattr(embedding_func_wrapper, "_api_key_count", key_count)
    return embedding_func_wrapper


async def create_formal_rag(
    *,
    working_dir: str | Path,
    variant_settings: VariantSettings,
    runtime_overrides: dict[str, Any] | None = None,
) -> LightRAG:
    from lightrag import LightRAG

    rag = LightRAG(
        working_dir=str(working_dir),
        llm_model_func=_build_llm_model_func(),
        embedding_func=_build_embedding_func(),
        enable_noise_filter=variant_settings.enable_noise_filter,
        noise_filter_config=variant_settings.noise_filter_config,
        **(runtime_overrides or {}),
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
    query_concurrency: int = 1,
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
    if query_concurrency <= 0:
        raise ValueError(
            f"query_concurrency must be > 0, got {query_concurrency}"
        )

    def _build_result_row(
        *,
        query_id: int | None,
        query: str,
        answers: list[str],
        metadata: dict[str, Any],
        payload_key: str,
        payload_value: Any,
    ) -> dict[str, Any]:
        row = {
            "query": query,
            "mode": query_mode,
            payload_key: payload_value,
        }
        if query_id is not None:
            row["query_id"] = query_id
        if answers:
            row["ground_truth_answers"] = answers
        if metadata:
            row["metadata"] = metadata
        return row

    async def _run_one(
        index: int,
        query_id: int | None,
        query: str,
        answers: list[str],
        metadata: dict[str, Any],
        semaphore: asyncio.Semaphore | None,
    ) -> tuple[int, bool, dict[str, Any]]:
        async def _execute() -> tuple[int, bool, dict[str, Any]]:
            try:
                answer = await rag.aquery(query, param=query_param)
                return (
                    index,
                    True,
                    _build_result_row(
                        query_id=query_id,
                        query=query,
                        answers=answers,
                        metadata=metadata,
                        payload_key="result",
                        payload_value=answer,
                    ),
                )
            except Exception as exc:
                return (
                    index,
                    False,
                    _build_result_row(
                        query_id=query_id,
                        query=query,
                        answers=answers,
                        metadata=metadata,
                        payload_key="error",
                        payload_value=str(exc),
                    ),
                )

        if semaphore is None:
            return await _execute()

        async with semaphore:
            return await _execute()

    semaphore = (
        None if query_concurrency == 1 else asyncio.Semaphore(query_concurrency)
    )
    tasks = [
        asyncio.create_task(
            _run_one(index, query_id, query, answers, metadata, semaphore)
        )
        for index, (query_id, query, answers, metadata) in enumerate(query_items)
    ]

    results: list[tuple[int, dict[str, Any]]] = []
    errors: list[tuple[int, dict[str, Any]]] = []
    for index, is_success, row in await asyncio.gather(*tasks):
        if is_success:
            results.append((index, row))
        else:
            errors.append((index, row))

    results.sort(key=lambda item: item[0])
    errors.sort(key=lambda item: item[0])
    return [row for _, row in results], [row for _, row in errors]


async def finalize_rag(rag: LightRAG | None) -> None:
    if rag is not None:
        await rag.finalize_storages()
