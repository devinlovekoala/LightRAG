from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
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

JUDGE_VERDICTS = (
    "supported",
    "partially_supported",
    "not_supported",
    "contradicted",
)
SOURCE_MODES = ("auto", "exported", "resolved")

SYSTEM_PROMPT = """You are a strict fact-checking judge for graph edges.

Decide whether a relation claim is supported by the provided source text only.
Do not use external knowledge.
`supported` requires explicit support from the source and at least one short exact evidence quote.
If you cannot point to an exact supporting quote from the source text, do not choose supported.
If the source text is truncated or only partially supports the claim, choose partially_supported.
If the source text mentions related entities or topics but does not explicitly state the relation, choose not_supported.
Return valid JSON only.
"""


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


def resolve_judge_source_texts(
    row: dict[str, Any],
    *,
    text_chunk_lookup: dict[str, str] | None = None,
    source_mode: str = "auto",
) -> list[str]:
    normalized_mode = str(source_mode or "auto").strip().lower()
    if normalized_mode not in SOURCE_MODES:
        raise ValueError(
            f"Unsupported source_mode: {source_mode}. Expected one of {SOURCE_MODES}."
        )

    exported_source_texts = [
        str(chunk)
        for chunk in row.get("source_chunks", [])
        if str(chunk or "").strip()
    ]

    if normalized_mode == "exported":
        return exported_source_texts

    resolved_source_texts = resolve_source_texts(row, text_chunk_lookup)
    if normalized_mode == "resolved":
        return resolved_source_texts

    return resolved_source_texts or exported_source_texts


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
    max_source_chunks: int = _DEFAULT_MAX_SOURCE_CHUNKS,
    max_source_chars: int = _DEFAULT_MAX_SOURCE_CHARS,
) -> list[str]:
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
    source_mode: str = "auto",
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
        source_texts = resolve_judge_source_texts(
            row,
            text_chunk_lookup=text_chunk_lookup,
            source_mode=source_mode,
        )
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
        self._request_delay_s = max(0.0, request_delay_ms / 1000.0)
        self._next_request_at = 0.0
        self._request_gate = asyncio.Lock()
        logger.info(
            "RemoteSourceJudge: model=%s base_url=%s timeout=%.0fs max_retries=%d max_source_chunks=%d max_source_chars=%d request_delay_ms=%d",
            model,
            base_url or "(default)",
            timeout,
            max_retries,
            self._max_source_chunks,
            self._max_source_chars,
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
            max_source_chunks=self._max_source_chunks,
            max_source_chars=self._max_source_chars,
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
                )
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
        return {"verdict": "not_supported", "support_score": 0.0, "explanation": f"error: {last_exc}"}

    async def close(self) -> None:
        await self._client.close()


async def build_report_async(
    variant_files: dict[str, Path],
    *,
    text_chunk_files: dict[str, Path],
    judge_func: Callable[[str, list[str]], Awaitable[dict[str, Any]]],
    concurrency: int,
    source_mode: str,
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
            source_mode=source_mode,
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
        "--source-mode",
        choices=SOURCE_MODES,
        default="auto",
        help="Which source texts to judge against: exported preview chunks, resolved chunk-id texts, or auto fallback (default: auto)",
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
        "Starting source-grounded judge evaluation — model=%s host=%s timeout=%.0fs retries=%d concurrency=%d source_mode=%s max_source_chunks=%d max_source_chars=%d request_delay_ms=%d",
        args.judge_model,
        args.judge_host or "(default)",
        args.judge_timeout,
        args.judge_max_retries,
        args.concurrency,
        args.source_mode,
        args.max_source_chunks,
        args.max_source_chars,
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
            request_delay_ms=args.request_delay_ms,
        )
        try:
            return await build_report_async(
                variant_files,
                text_chunk_files=text_chunk_files,
                judge_func=judge,
                concurrency=args.concurrency,
                source_mode=args.source_mode,
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
