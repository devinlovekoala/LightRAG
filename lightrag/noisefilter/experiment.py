from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def parse_thresholds(raw: str | Iterable[float]) -> list[float]:
    if isinstance(raw, str):
        values = [segment.strip() for segment in raw.split(",")]
    else:
        values = [str(value).strip() for value in raw]

    thresholds = sorted(
        {
            max(0.0, min(1.0, float(value)))
            for value in values
            if value not in {"", "None", "null"}
        }
    )
    return thresholds


def _summary_key(row: dict[str, Any]) -> str:
    return f"{row['injector']}@{float(row['noise_ratio']):.2f}"


def summarize_experiment_rows(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best_hard_by_f1: dict[str, dict[str, Any]] = {}
    best_overall_retrieval: dict[str, dict[str, Any]] = {}

    for row in rows:
        key = _summary_key(row)
        strategy = row.get("strategy")

        if strategy == "hard":
            current = best_hard_by_f1.get(key)
            if current is None or float(row.get("filter_f1", -1.0)) > float(
                current.get("filter_f1", -1.0)
            ):
                best_hard_by_f1[key] = dict(row)

        current_retrieval = best_overall_retrieval.get(key)
        if current_retrieval is None or float(
            row.get("retrieval_noise_rate_at_k", 1.0)
        ) < float(current_retrieval.get("retrieval_noise_rate_at_k", 1.0)):
            best_overall_retrieval[key] = dict(row)

    return {
        "best_hard_by_f1": best_hard_by_f1,
        "best_overall_retrieval": best_overall_retrieval,
    }


def render_markdown_report(
    rows: list[dict[str, Any]],
    summary: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> str:
    lines = [
        "# NoiseFilter-RAG Experiment Report",
        "",
        "## Configuration",
        "",
        "```json",
        json.dumps(config, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Best Hard Filter by F1",
        "",
        "| Setting | Threshold | Filter F1 | Retrieval Noise@K |",
        "| --- | ---: | ---: | ---: |",
    ]

    for key, row in summary.get("best_hard_by_f1", {}).items():
        lines.append(
            f"| {key} | {row.get('threshold')} | {float(row.get('filter_f1', 0.0)):.3f} | "
            f"{float(row.get('retrieval_noise_rate_at_k', 0.0)):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Best Retrieval Strategy",
            "",
            "| Setting | Strategy | Threshold | Retrieval Noise@K | Returned Edges |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )

    for key, row in summary.get("best_overall_retrieval", {}).items():
        threshold = row.get("threshold")
        threshold_text = "-" if threshold is None else str(threshold)
        lines.append(
            f"| {key} | {row.get('strategy')} | {threshold_text} | "
            f"{float(row.get('retrieval_noise_rate_at_k', 0.0)):.3f} | "
            f"{float(row.get('avg_returned_edges', 0.0)):.2f} |"
        )

    lines.extend(
        [
            "",
            "## Raw Rows",
            "",
            "| Injector | Noise Ratio | Strategy | Threshold | Filter F1 | Retrieval Noise@K | Returned Edges |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in rows:
        threshold = row.get("threshold")
        threshold_text = "-" if threshold is None else str(threshold)
        lines.append(
            f"| {row.get('injector')} | {float(row.get('noise_ratio', 0.0)):.2f} | "
            f"{row.get('strategy')} | {threshold_text} | "
            f"{float(row.get('filter_f1', 0.0)):.3f} | "
            f"{float(row.get('retrieval_noise_rate_at_k', 0.0)):.3f} | "
            f"{float(row.get('avg_returned_edges', 0.0)):.2f} |"
        )

    lines.append("")
    return "\n".join(lines)


def write_experiment_bundle(
    output_dir: str | Path,
    rows: list[dict[str, Any]],
    summary: dict[str, dict[str, Any]],
    config: dict[str, Any],
    *,
    stem: str = "noisefilter_experiment",
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_path / f"{stem}_{timestamp}"

    json_path = base.with_suffix(".json")
    csv_path = base.with_suffix(".csv")
    markdown_path = base.with_suffix(".md")

    payload = {
        "config": config,
        "summary": summary,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    markdown_path.write_text(
        render_markdown_report(rows, summary, config),
        encoding="utf-8",
    )

    return {
        "json": json_path,
        "csv": csv_path,
        "markdown": markdown_path,
    }
