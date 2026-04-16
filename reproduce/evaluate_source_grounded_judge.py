from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from statistics import fmean
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce.evaluate_grounding_signals import (
    compute_average_precision,
    compute_roc_auc,
    load_edge_rows,
    load_text_chunks,
    resolve_source_texts,
)

load_dotenv(dotenv_path=".env", override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "180"))
_DEFAULT_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2.0
_DEFAULT_MAX_SOURCE_CHUNKS = int(os.getenv("JUDGE_MAX_SOURCE_CHUNKS", "3"))
_DEFAULT_MAX_SOURCE_CHARS = int(os.getenv("JUDGE_MAX_SOURCE_CHARS", "2400"))
_DEFAULT_REQUEST_DELAY_MS = int(os.getenv("JUDGE_REQUEST_DELAY_MS", "0"))
_DEFAULT_SNIPPET_WINDOW_SENTENCES = int(os.getenv("JUDGE_SNIPPET_WINDOW_SENTENCES", "3"))
_DEFAULT_MAX_SNIPPETS_PER_CHUNK = int(os.getenv("JUDGE_MAX_SNIPPETS_PER_CHUNK", "2"))

JUDGE_VERDICTS = (
    "supported",
    "partially_supported",
    "not_supported",
    "contradicted",
)

SYSTEM_PROMPT = """You are a strict fact-checking judge for graph edges.

Decide whether a relation claim is supported by the provided source text only.
Do not use external knowledge.
`supported` requires explicit support from the source and at least one short exact evidence quote.
If you cannot point to an exact supporting quote from the source text, do not choose supported.
If the source text is truncated or only partially supports the claim, choose partially_supported.
If the source text mentions related entities or topics but does not explicitly state the relation, choose not_supported.
Return valid JSON only.
"""

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_PASSAGE_SPLIT_RE = re.compile(r"(?=Passage\s+\d+\s*:)", re.IGNORECASE)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "then",
    "this",
    "to",
    "was",
    "were",
    "which",
    "with",
}


def build_claim(row: dict[str, Any]) -> str:
    description = str(row.get("description", "")).strip()
    if description:
        return description

    src = str(row.get("src", "")).strip()
    dst = str(row.get("dst", "")).strip()
    keywords = str(row.get("keywords", "")).strip()
    if keywords:
        return f"{src} {keywords} {dst}.".strip()
    return f"{src} is related to {dst}.".strip()


def score_verdict(verdict: str) -> float:
    normalized = str(verdict or "").strip().lower()
    if normalized == "supported":
        return 1.0
    if normalized == "partially_supported":
        return 0.6
    if normalized in {"not_supported", "contradicted"}:
        return 0.0
    return 0.0


def _normalize_verdict(raw: str | None) -> str:
    verdict = str(raw or "").strip().lower()
    return verdict if verdict in JUDGE_VERDICTS else "not_supported"


def _normalize_whitespace(text: str) -> str:
    return " ".join(str(text or "").split())


def _normalize_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(str(text or "").lower()))


def _tokenize_text(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text or "").lower())


def _extract_claim_terms(claim: str, *, max_terms: int = 12) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for token in sorted(_tokenize_text(claim), key=lambda value: (-len(value), value)):
        if token in seen:
            continue
        if token in _STOPWORDS:
            continue
        if len(token) < 3 and not token.isdigit():
            continue
        seen.add(token)
        terms.append(token)
        if len(terms) >= max_terms:
            break
    return terms


def _split_passage_blocks(text: str) -> list[str]:
    raw = str(text or "").strip()
    if not raw:
        return []
    if _PASSAGE_SPLIT_RE.search(raw):
        return [
            block.strip()
            for block in _PASSAGE_SPLIT_RE.split(raw)
            if block and block.strip()
        ]
    return [block.strip() for block in re.split(r"\n\s*\n", raw) if block.strip()]


def _split_sentence_windows(text: str, *, window_sentences: int) -> list[str]:
    normalized = str(text or "").strip()
    if not normalized:
        return []
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT_RE.split(normalized)
        if sentence and sentence.strip()
    ]
    if not sentences:
        return []
    if len(sentences) <= window_sentences:
        return [" ".join(sentences)]
    windows: list[str] = []
    for start in range(0, len(sentences) - window_sentences + 1):
        windows.append(" ".join(sentences[start : start + window_sentences]))
    return windows


def _score_candidate_text(candidate_text: str, claim_terms: list[str]) -> float:
    normalized_candidate = _normalize_text(candidate_text)
    if not normalized_candidate:
        return 0.0
    if not claim_terms:
        return 0.0
    score = 0.0
    unique_hits = 0
    for term in claim_terms:
        if term in normalized_candidate:
            unique_hits += 1
            score += 1.0 + min(len(term), 12) / 12.0
    score += 2.0 if unique_hits >= 2 else 0.0
    score += 3.0 if unique_hits >= 4 else 0.0
    return score


def localize_source_texts(
    claim: str,
    source_texts: list[str],
    *,
    max_source_chunks: int = _DEFAULT_MAX_SOURCE_CHUNKS,
    max_source_chars: int = _DEFAULT_MAX_SOURCE_CHARS,
    snippet_window_sentences: int = _DEFAULT_SNIPPET_WINDOW_SENTENCES,
    max_snippets_per_chunk: int = _DEFAULT_MAX_SNIPPETS_PER_CHUNK,
) -> list[str]:
    claim_terms = _extract_claim_terms(claim)
    candidates: list[tuple[float, int, str]] = []

    for chunk_index, raw_text in enumerate(source_texts[: max(1, max_source_chunks)]):
        passage_blocks = _split_passage_blocks(raw_text)
        block_candidates = passage_blocks if passage_blocks else [str(raw_text or "").strip()]
        chunk_candidates: list[tuple[float, str]] = []

        for block in block_candidates:
            block = block.strip()
            if not block:
                continue
            score = _score_candidate_text(block, claim_terms)
            chunk_candidates.append((score, block))

            if len(block) > 500:
                for window in _split_sentence_windows(
                    block,
                    window_sentences=max(1, snippet_window_sentences),
                ):
                    window_score = _score_candidate_text(window, claim_terms)
                    chunk_candidates.append((window_score + 0.25, window))

        if not chunk_candidates:
            continue

        deduped: list[tuple[float, str]] = []
        seen_texts: set[str] = set()
        for score, text in sorted(chunk_candidates, key=lambda item: (-item[0], len(item[1]))):
            normalized = _normalize_whitespace(text)
            if not normalized or normalized in seen_texts:
                continue
            seen_texts.add(normalized)
            deduped.append((score, normalized))
            if len(deduped) >= max(1, max_snippets_per_chunk):
                break

        for score, text in deduped:
            candidates.append((score, chunk_index, text))

    if not candidates:
        return []

    localized: list[str] = []
    seen_texts: set[str] = set()
    remaining_chars = max(0, max_source_chars)
    for score, _, text in sorted(candidates, key=lambda item: (-item[0], item[1], len(item[2]))):
        if remaining_chars < 32:
            break
        if score <= 0 and localized:
            break
        truncated = _truncate_text(text, remaining_chars)
        if len(truncated.strip()) < 32:
            continue
        normalized = _normalize_whitespace(truncated)
        if normalized in seen_texts:
            continue
        seen_texts.add(normalized)
        localized.append(truncated)
        remaining_chars -= len(truncated)

    return localized


def _truncate_text(text: str, max_chars: int) -> str:
    compact = " ".join(str(text or "").split())
    if len(compact) <= max_chars:
        return compact
    if max_chars <= 16:
        return compact[:max_chars]
    return compact[: max_chars - 16].rstrip() + " [...truncated]"


def prepare_source_texts(
    source_texts: list[str],
    *,
    claim: str = "",
    max_source_chunks: int = _DEFAULT_MAX_SOURCE_CHUNKS,
    max_source_chars: int = _DEFAULT_MAX_SOURCE_CHARS,
    snippet_window_sentences: int = _DEFAULT_SNIPPET_WINDOW_SENTENCES,
    max_snippets_per_chunk: int = _DEFAULT_MAX_SNIPPETS_PER_CHUNK,
) -> list[str]:
    localized = localize_source_texts(
        claim,
        source_texts,
        max_source_chunks=max_source_chunks,
        max_source_chars=max_source_chars,
        snippet_window_sentences=snippet_window_sentences,
        max_snippets_per_chunk=max_snippets_per_chunk,
    )
    if localized:
        return localized

    prepared: list[str] = []
    remaining_chars = max(0, max_source_chars)
    for raw_text in source_texts[: max(1, max_source_chunks)]:
        if remaining_chars < 32:
            break
        truncated = _truncate_text(raw_text, remaining_chars)
        if len(truncated.strip()) < 32:
            continue
        prepared.append(truncated)
        remaining_chars -= len(truncated)
    return prepared


def build_user_prompt(claim: str, source_texts: list[str]) -> str:
    rendered_sources = "\n\n".join(
        f"Source {index + 1}:\n{chunk_text}" for index, chunk_text in enumerate(source_texts)
    )
    return f"""Claim:
{claim}

Source Text:
{rendered_sources}

Return valid JSON with this exact shape:
{{
  "verdict": "supported | partially_supported | not_supported | contradicted",
  "support_score": 0.0,
  "evidence": ["exact short quote from source"],
  "explanation": "short explanation"
}}
"""


def _normalize_evidence_snippets(raw: Any) -> list[str]:
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, str):
        values = [raw]
    else:
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        snippet = " ".join(str(value or "").split()).strip().strip('"')
        if len(snippet) < 8:
            continue
        key = snippet.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(snippet)
    return normalized


def _has_anchored_evidence(evidence_snippets: list[str], source_texts: list[str]) -> bool:
    if not evidence_snippets or not source_texts:
        return False
    normalized_sources = [" ".join(text.split()).casefold() for text in source_texts]
    for snippet in evidence_snippets:
        normalized_snippet = " ".join(snippet.split()).casefold()
        if any(normalized_snippet in source for source in normalized_sources):
            return True
    return False


def calibrate_judge_result(
    verdict: str,
    support_score: float | None,
    explanation: str,
    evidence_snippets: list[str],
    source_texts: list[str],
) -> dict[str, Any]:
    normalized_verdict = _normalize_verdict(verdict)
    normalized_explanation = str(explanation or "").strip()
    anchored_evidence = _has_anchored_evidence(evidence_snippets, source_texts)

    if normalized_verdict == "supported" and not anchored_evidence:
        normalized_verdict = "partially_supported"
        prefix = "Downgraded from supported because no exact evidence quote was grounded in the provided source."
        normalized_explanation = (
            f"{prefix} {normalized_explanation}".strip()
            if normalized_explanation
            else prefix
        )

    normalized_score = score_verdict(normalized_verdict) if support_score is None else float(support_score)
    if normalized_verdict == "supported":
        normalized_score = max(normalized_score, 0.9)
    elif normalized_verdict == "partially_supported":
        normalized_score = min(normalized_score, 0.6)
    else:
        normalized_score = min(normalized_score, 0.1)

    return {
        "verdict": normalized_verdict,
        "support_score": normalized_score,
        "explanation": normalized_explanation,
        "evidence": evidence_snippets,
        "anchored_evidence": anchored_evidence,
    }


async def evaluate_variant_async(
    variant: str,
    rows: list[dict[str, Any]],
    *,
    judge_func: Callable[[str, list[str]], Awaitable[dict[str, Any]]],
    text_chunk_lookup: dict[str, str] | None = None,
    concurrency: int = 4,
) -> dict[str, Any]:
    filtered_rows = [
        row
        for row in rows
        if str(row.get("manual_label", "")).strip().lower()
        in {"correct", "wrong", "ambiguous"}
    ]
    total = len(filtered_rows)
    logger.info("[%s] Judging %d labeled rows (concurrency=%d)", variant, total, concurrency)
    semaphore = asyncio.Semaphore(max(1, concurrency))
    completed = 0
    start_time = time.monotonic()

    async def _judge_row(row: dict[str, Any]) -> dict[str, Any]:
        nonlocal completed
        claim = build_claim(row)
        source_texts = resolve_source_texts(row, text_chunk_lookup)
        async with semaphore:
            result = await judge_func(claim, source_texts)
        verdict = _normalize_verdict(result.get("verdict"))
        support_score = result.get("support_score")
        if support_score is None:
            support_score = score_verdict(verdict)
        completed += 1
        if completed % max(1, total // 10) == 0 or completed == total:
            elapsed = time.monotonic() - start_time
            rate = completed / elapsed if elapsed > 0 else 0.0
            logger.info(
                "[%s] Progress: %d/%d (%.0f%%) — %.2f rows/s — verdict=%s",
                variant, completed, total, 100.0 * completed / total, rate, verdict,
            )
        return {
            **row,
            "judge_claim": claim,
            "judge_verdict": verdict,
            "judge_support_score": float(support_score),
            "judge_explanation": str(result.get("explanation", "")).strip(),
            "judge_evidence": json.dumps(result.get("evidence", []), ensure_ascii=False),
            "judge_anchored_evidence": bool(result.get("anchored_evidence", False)),
            "judge_localized_sources": json.dumps(
                result.get("localized_source_texts", source_texts), ensure_ascii=False
            ),
            "resolved_source_chunk_count": len(source_texts),
        }

    judged_rows = await asyncio.gather(*[_judge_row(row) for row in filtered_rows])
    elapsed = time.monotonic() - start_time
    logger.info("[%s] Done — %d rows judged in %.1fs", variant, total, elapsed)
    return summarize_variant_rows(variant, list(judged_rows))


def summarize_variant_rows(variant: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = {
        label: sum(row["manual_label"] == label for row in rows)
        for label in ("correct", "wrong", "ambiguous")
    }
    verdict_counts = {verdict: 0 for verdict in JUDGE_VERDICTS}
    for row in rows:
        verdict_counts[_normalize_verdict(row.get("judge_verdict"))] += 1

    total_edges = len(rows)
    strict_precision = label_counts["correct"] / total_edges if total_edges else 0.0
    lenient_precision = (
        (label_counts["correct"] + label_counts["ambiguous"]) / total_edges
        if total_edges
        else 0.0
    )

    judge_score_means_by_label = {
        label: fmean(
            float(row.get("judge_support_score", 0.0))
            for row in rows
            if row["manual_label"] == label
        )
        if label_counts[label]
        else 0.0
        for label in ("correct", "wrong", "ambiguous")
    }

    binary_rows = [row for row in rows if row["manual_label"] in {"correct", "wrong"}]
    labels = [1 if row["manual_label"] == "correct" else 0 for row in binary_rows]
    scores = [float(row.get("judge_support_score", 0.0)) for row in binary_rows]
    binary_ranking = {
        "roc_auc": compute_roc_auc(labels, scores),
        "average_precision": compute_average_precision(labels, scores),
    }

    return {
        "variant": variant,
        "total_edges": total_edges,
        "label_counts": label_counts,
        "strict_precision": strict_precision,
        "lenient_precision": lenient_precision,
        "judge_score_means_by_label": judge_score_means_by_label,
        "verdict_counts": verdict_counts,
        "binary_ranking": binary_ranking,
        "rows": rows,
    }


def build_comparison_summary(
    variant_summaries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    baseline = variant_summaries.get("baseline")
    noisefilter = variant_summaries.get("noisefilter")
    if baseline is None or noisefilter is None:
        return {}
    return {
        "precision_delta": {
            "strict": noisefilter["strict_precision"] - baseline["strict_precision"],
            "lenient": noisefilter["lenient_precision"] - baseline["lenient_precision"],
        },
        "ranking_delta": {
            "roc_auc": (
                None
                if baseline["binary_ranking"]["roc_auc"] is None
                or noisefilter["binary_ranking"]["roc_auc"] is None
                else noisefilter["binary_ranking"]["roc_auc"]
                - baseline["binary_ranking"]["roc_auc"]
            ),
            "average_precision": (
                None
                if baseline["binary_ranking"]["average_precision"] is None
                or noisefilter["binary_ranking"]["average_precision"] is None
                else noisefilter["binary_ranking"]["average_precision"]
                - baseline["binary_ranking"]["average_precision"]
            ),
        },
    }


class RemoteSourceJudge:
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        max_source_chunks: int = _DEFAULT_MAX_SOURCE_CHUNKS,
        max_source_chars: int = _DEFAULT_MAX_SOURCE_CHARS,
        snippet_window_sentences: int = _DEFAULT_SNIPPET_WINDOW_SENTENCES,
        max_snippets_per_chunk: int = _DEFAULT_MAX_SNIPPETS_PER_CHUNK,
        request_delay_ms: int = _DEFAULT_REQUEST_DELAY_MS,
    ) -> None:
        import httpx
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=httpx.Timeout(timeout, connect=30.0),
            max_retries=0,  # manual retry with backoff below
        )
        self._model = model
        self._max_retries = max_retries
        self._max_source_chunks = max(1, max_source_chunks)
        self._max_source_chars = max(256, max_source_chars)
        self._snippet_window_sentences = max(1, snippet_window_sentences)
        self._max_snippets_per_chunk = max(1, max_snippets_per_chunk)
        self._request_delay_s = max(0.0, request_delay_ms / 1000.0)
        self._next_request_at = 0.0
        self._request_gate = asyncio.Lock()
        logger.info(
            "RemoteSourceJudge: model=%s base_url=%s timeout=%.0fs max_retries=%d max_source_chunks=%d max_source_chars=%d snippet_window_sentences=%d max_snippets_per_chunk=%d request_delay_ms=%d",
            model,
            base_url or "(default)",
            timeout,
            max_retries,
            self._max_source_chunks,
            self._max_source_chars,
            self._snippet_window_sentences,
            self._max_snippets_per_chunk,
            request_delay_ms,
        )

    async def _wait_for_turn(self) -> None:
        if self._request_delay_s <= 0:
            return
        async with self._request_gate:
            now = time.monotonic()
            wait = max(0.0, self._next_request_at - now)
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_request_at = now + self._request_delay_s

    async def __call__(self, claim: str, source_texts: list[str]) -> dict[str, Any]:
        import httpx
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        prepared_source_texts = prepare_source_texts(
            source_texts,
            claim=claim,
            max_source_chunks=self._max_source_chunks,
            max_source_chars=self._max_source_chars,
            snippet_window_sentences=self._snippet_window_sentences,
            max_snippets_per_chunk=self._max_snippets_per_chunk,
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(claim, prepared_source_texts)},
        ]
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                await self._wait_for_turn()
                response = await self._client.chat.completions.create(
                    model=self._model,
                    temperature=0,
                    response_format={"type": "json_object"},
                    messages=messages,
                )
                content = response.choices[0].message.content or "{}"
                parsed = json.loads(content)
                return calibrate_judge_result(
                    parsed.get("verdict"),
                    parsed.get("support_score"),
                    str(parsed.get("explanation", "")).strip(),
                    _normalize_evidence_snippets(parsed.get("evidence")),
                    prepared_source_texts,
                ) | {"localized_source_texts": prepared_source_texts}
            except (APITimeoutError, APIConnectionError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
                last_exc = exc
                wait = _RETRY_BACKOFF_BASE ** (attempt - 1)
                logger.warning(
                    "Judge request timed out/failed (attempt %d/%d) — retrying in %.0fs: %s",
                    attempt, self._max_retries, wait, exc,
                )
                await asyncio.sleep(wait)
            except APIStatusError as exc:
                last_exc = exc
                if exc.status_code and exc.status_code < 500:
                    logger.error("Judge API error %d (non-retryable): %s", exc.status_code, exc)
                    break
                wait = _RETRY_BACKOFF_BASE ** (attempt - 1)
                logger.warning(
                    "Judge API server error %d (attempt %d/%d) — retrying in %.0fs",
                    exc.status_code, attempt, self._max_retries, wait,
                )
                await asyncio.sleep(wait)
            except json.JSONDecodeError as exc:
                logger.warning("Judge returned invalid JSON: %s", exc)
                last_exc = exc
                break

        logger.error("Judge failed after %d attempts: %s", self._max_retries, last_exc)
        return {
            "verdict": "not_supported",
            "support_score": 0.0,
            "explanation": f"error: {last_exc}",
            "localized_source_texts": prepared_source_texts,
        }

    async def close(self) -> None:
        await self._client.close()


async def build_report_async(
    variant_files: dict[str, Path],
    *,
    text_chunk_files: dict[str, Path],
    judge_func: Callable[[str, list[str]], Awaitable[dict[str, Any]]],
    concurrency: int,
) -> dict[str, Any]:
    variants: dict[str, dict[str, Any]] = {}
    total_variants = len(variant_files)
    for index, (variant, path) in enumerate(variant_files.items(), start=1):
        logger.info("=== Variant %d/%d: %s (%s) ===", index, total_variants, variant, path)
        lookup = (
            load_text_chunks(text_chunk_files[variant])
            if variant in text_chunk_files
            else None
        )
        if lookup is not None:
            logger.info("[%s] Loaded %d text chunks", variant, len(lookup))
        rows = load_edge_rows(path)
        logger.info("[%s] Loaded %d edge rows from %s", variant, len(rows), path)
        variants[variant] = await evaluate_variant_async(
            variant,
            rows,
            judge_func=judge_func,
            text_chunk_lookup=lookup,
            concurrency=concurrency,
        )
    return {
        "variants": variants,
        "comparison": build_comparison_summary(variants),
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    serializable = {
        "variants": {
            variant: {key: value for key, value in summary.items() if key != "rows"}
            for variant, summary in payload.get("variants", {}).items()
        },
        "comparison": payload.get("comparison", {}),
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_csv(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for variant, summary in payload.get("variants", {}).items():
        rows.append(
            {
                "variant": variant,
                "total_edges": summary["total_edges"],
                "correct": summary["label_counts"]["correct"],
                "wrong": summary["label_counts"]["wrong"],
                "ambiguous": summary["label_counts"]["ambiguous"],
                "strict_precision": summary["strict_precision"],
                "lenient_precision": summary["lenient_precision"],
                "roc_auc": summary["binary_ranking"]["roc_auc"],
                "average_precision": summary["binary_ranking"]["average_precision"],
                **{
                    f"verdict_{verdict}": summary["verdict_counts"][verdict]
                    for verdict in JUDGE_VERDICTS
                },
            }
        )

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_rows_csv(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for variant, summary in payload.get("variants", {}).items():
        rows.extend(summary.get("rows", []))

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Source-Grounded Judge Evaluation",
        "",
        "| Variant | Total | Correct | Wrong | Ambiguous | Strict Precision | Lenient Precision | ROC AUC | AP |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, summary in payload.get("variants", {}).items():
        label_counts = summary["label_counts"]
        ranking = summary["binary_ranking"]
        lines.append(
            f"| {variant} | {summary['total_edges']} | {label_counts['correct']} | {label_counts['wrong']} | {label_counts['ambiguous']} | {summary['strict_precision']:.4f} | {summary['lenient_precision']:.4f} | {_fmt(ranking['roc_auc'])} | {_fmt(ranking['average_precision'])} |"
        )
    comparison = payload.get("comparison", {})
    if comparison:
        lines.extend(
            [
                "",
                f"Strict precision delta (noisefilter - baseline): {comparison['precision_delta']['strict']:.4f}",
                f"Lenient precision delta (noisefilter - baseline): {comparison['precision_delta']['lenient']:.4f}",
                f"Judge ROC AUC delta: {_fmt(comparison['ranking_delta']['roc_auc'])}",
                f"Judge AP delta: {_fmt(comparison['ranking_delta']['average_precision'])}",
                "",
            ]
        )
    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _parse_mapping(raw_values: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for raw in raw_values:
        if "=" not in raw:
            raise ValueError(f"Invalid mapping: {raw}")
        variant, path = raw.split("=", 1)
        mapping[variant.strip().lower()] = Path(path).expanduser()
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate manually labeled edges with a remote source-grounded LLM judge.",
    )
    parser.add_argument("--variant-file", action="append", required=True)
    parser.add_argument("--text-chunks-file", action="append", default=[])
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--judge-model",
        default=os.getenv("JUDGE_MODEL") or os.getenv("LLM_MODEL", "gpt-4o-mini"),
    )
    parser.add_argument(
        "--judge-host",
        default=os.getenv("JUDGE_BINDING_HOST") or os.getenv("LLM_BINDING_HOST"),
    )
    parser.add_argument(
        "--judge-api-key",
        default=os.getenv("JUDGE_BINDING_API_KEY") or os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )
    parser.add_argument(
        "--judge-timeout",
        type=float,
        default=float(os.getenv("JUDGE_TIMEOUT") or os.getenv("LLM_TIMEOUT", "180")),
        help="Per-request read timeout in seconds (default: JUDGE_TIMEOUT / LLM_TIMEOUT env or 180)",
    )
    parser.add_argument(
        "--judge-max-retries",
        type=int,
        default=_DEFAULT_MAX_RETRIES,
        help="Number of retry attempts on timeout/connection errors (default: 3)",
    )
    parser.add_argument(
        "--max-source-chunks",
        type=int,
        default=_DEFAULT_MAX_SOURCE_CHUNKS,
        help="Maximum number of source chunks passed to the judge per edge (default: 3)",
    )
    parser.add_argument(
        "--max-source-chars",
        type=int,
        default=_DEFAULT_MAX_SOURCE_CHARS,
        help="Maximum total source characters passed to the judge per edge (default: 2400)",
    )
    parser.add_argument(
        "--snippet-window-sentences",
        type=int,
        default=_DEFAULT_SNIPPET_WINDOW_SENTENCES,
        help="Sentence window size for evidence localization inside long chunks (default: 3)",
    )
    parser.add_argument(
        "--max-snippets-per-chunk",
        type=int,
        default=_DEFAULT_MAX_SNIPPETS_PER_CHUNK,
        help="Maximum localized snippets retained from each source chunk (default: 2)",
    )
    parser.add_argument(
        "--request-delay-ms",
        type=int,
        default=_DEFAULT_REQUEST_DELAY_MS,
        help="Delay inserted before each judge request to smooth TPM pressure (default: 0)",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.judge_api_key:
        raise EnvironmentError(
            "Missing judge API key. Set LLM_BINDING_API_KEY in .env or pass --judge-api-key."
        )

    variant_files = _parse_mapping(args.variant_file)
    text_chunk_files = _parse_mapping(args.text_chunks_file)

    logger.info(
        "Starting source-grounded judge evaluation — model=%s host=%s timeout=%.0fs retries=%d concurrency=%d max_source_chunks=%d max_source_chars=%d snippet_window_sentences=%d max_snippets_per_chunk=%d request_delay_ms=%d",
        args.judge_model,
        args.judge_host or "(default)",
        args.judge_timeout,
        args.judge_max_retries,
        args.concurrency,
        args.max_source_chunks,
        args.max_source_chars,
        args.snippet_window_sentences,
        args.max_snippets_per_chunk,
        args.request_delay_ms,
    )
    logger.info("Variants: %s", list(variant_files.keys()))

    async def _run() -> dict[str, Any]:
        judge = RemoteSourceJudge(
            model=args.judge_model,
            api_key=args.judge_api_key,
            base_url=args.judge_host,
            timeout=args.judge_timeout,
            max_retries=args.judge_max_retries,
            max_source_chunks=args.max_source_chunks,
            max_source_chars=args.max_source_chars,
            snippet_window_sentences=args.snippet_window_sentences,
            max_snippets_per_chunk=args.max_snippets_per_chunk,
            request_delay_ms=args.request_delay_ms,
        )
        try:
            return await build_report_async(
                variant_files,
                text_chunk_files=text_chunk_files,
                judge_func=judge,
                concurrency=args.concurrency,
            )
        finally:
            await judge.close()

    report = asyncio.run(_run())
    output_prefix = Path(args.output_prefix)
    save_json(output_prefix.with_suffix(".json"), report)
    save_csv(output_prefix.with_suffix(".csv"), report)
    save_markdown(output_prefix.with_suffix(".md"), report)
    save_rows_csv(output_prefix.with_name(output_prefix.name + ".rows.csv"), report)

    print("Source-grounded judge evaluation completed.")
    print(f"  json: {output_prefix.with_suffix('.json')}")
    print(f"  csv: {output_prefix.with_suffix('.csv')}")
    print(f"  markdown: {output_prefix.with_suffix('.md')}")
    print(f"  rows: {output_prefix.with_name(output_prefix.name + '.rows.csv')}")
    for variant, summary in report["variants"].items():
        print(
            f"  {variant}: roc_auc={_fmt(summary['binary_ranking']['roc_auc'])} "
            f"ap={_fmt(summary['binary_ranking']['average_precision'])}"
        )


if __name__ == "__main__":
    main()
