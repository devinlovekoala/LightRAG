from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from pathlib import Path
from statistics import fmean
from typing import Any, Awaitable, Callable

from dotenv import load_dotenv

from reproduce.evaluate_grounding_signals import (
    compute_average_precision,
    compute_roc_auc,
    load_edge_rows,
    load_text_chunks,
    resolve_source_texts,
)

load_dotenv(dotenv_path=".env", override=False)

JUDGE_VERDICTS = (
    "supported",
    "partially_supported",
    "not_supported",
    "contradicted",
)

SYSTEM_PROMPT = """You are a strict fact-checking judge for graph edges.

Decide whether a relation claim is supported by the provided source text only.
Do not use external knowledge.
If the source text is truncated or only partially supports the claim, choose partially_supported.
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
  "explanation": "short explanation"
}}
"""


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
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _judge_row(row: dict[str, Any]) -> dict[str, Any]:
        claim = build_claim(row)
        source_texts = resolve_source_texts(row, text_chunk_lookup)
        async with semaphore:
            result = await judge_func(claim, source_texts)
        verdict = _normalize_verdict(result.get("verdict"))
        support_score = result.get("support_score")
        if support_score is None:
            support_score = score_verdict(verdict)
        return {
            **row,
            "judge_claim": claim,
            "judge_verdict": verdict,
            "judge_support_score": float(support_score),
            "judge_explanation": str(result.get("explanation", "")).strip(),
            "resolved_source_chunk_count": len(source_texts),
        }

    judged_rows = await asyncio.gather(*[_judge_row(row) for row in filtered_rows])
    return summarize_variant_rows(variant, judged_rows)


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
    ) -> None:
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def __call__(self, claim: str, source_texts: list[str]) -> dict[str, Any]:
        response = await self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(claim, source_texts)},
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
        verdict = _normalize_verdict(parsed.get("verdict"))
        score = parsed.get("support_score")
        return {
            "verdict": verdict,
            "support_score": score_verdict(verdict) if score is None else float(score),
            "explanation": str(parsed.get("explanation", "")).strip(),
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
    for variant, path in variant_files.items():
        lookup = (
            load_text_chunks(text_chunk_files[variant])
            if variant in text_chunk_files
            else None
        )
        variants[variant] = await evaluate_variant_async(
            variant,
            load_edge_rows(path),
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
    parser.add_argument("--judge-model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    parser.add_argument("--judge-host", default=os.getenv("LLM_BINDING_HOST"))
    parser.add_argument(
        "--judge-api-key",
        default=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
    )
    parser.add_argument("--concurrency", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.judge_api_key:
        raise EnvironmentError("Missing judge API key.")

    variant_files = _parse_mapping(args.variant_file)
    text_chunk_files = _parse_mapping(args.text_chunks_file)

    async def _run() -> dict[str, Any]:
        judge = RemoteSourceJudge(
            model=args.judge_model,
            api_key=args.judge_api_key,
            base_url=args.judge_host,
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
