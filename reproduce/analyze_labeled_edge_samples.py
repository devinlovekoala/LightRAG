from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any


METRIC_FIELDS = (
    "conf_score",
    "conf_freq_score",
    "conf_consistency_score",
    "conf_semantic_score",
)
LABELS = ("correct", "wrong", "ambiguous")


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().lower()


def load_labeled_rows(path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows: list[dict[str, Any]] = []
        for row in reader:
            normalized = dict(row)
            normalized["variant"] = str(row.get("variant", "")).strip().lower()
            normalized["manual_label"] = _normalize_label(row.get("manual_label"))
            for metric in METRIC_FIELDS:
                normalized[metric] = _coerce_float(row.get(metric), 0.0)
            rows.append(normalized)
    return rows


def _mean_metric(rows: list[dict[str, Any]], metric: str) -> float:
    if not rows:
        return 0.0
    return fmean(float(row.get(metric, 0.0) or 0.0) for row in rows)


def _signal_status(correct_mean: float, wrong_mean: float, eps: float = 1e-6) -> str:
    diff = correct_mean - wrong_mean
    if abs(diff) <= eps:
        return "dead"
    if diff < 0:
        return "inverted"
    return "discriminative"


def summarize_variant(variant: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    filtered_rows = [
        row for row in rows if _normalize_label(row.get("manual_label")) in LABELS
    ]
    label_groups = {
        label: [
            row
            for row in filtered_rows
            if _normalize_label(row.get("manual_label")) == label
        ]
        for label in LABELS
    }
    total_edges = len(filtered_rows)
    label_counts = {label: len(group) for label, group in label_groups.items()}

    metric_means_by_label: dict[str, dict[str, float]] = {}
    signal_diagnostics: dict[str, dict[str, Any]] = {}

    for metric in METRIC_FIELDS:
        metric_means_by_label[metric] = {
            label: _mean_metric(group, metric) for label, group in label_groups.items()
        }
        correct_mean = metric_means_by_label[metric]["correct"]
        wrong_mean = metric_means_by_label[metric]["wrong"]
        signal_diagnostics[metric] = {
            "status": _signal_status(correct_mean, wrong_mean),
            "correct_minus_wrong_mean": correct_mean - wrong_mean,
            "correct_mean": correct_mean,
            "wrong_mean": wrong_mean,
            "ambiguous_mean": metric_means_by_label[metric]["ambiguous"],
        }

    strict_precision = (
        label_counts["correct"] / total_edges if total_edges else 0.0
    )
    lenient_precision = (
        (label_counts["correct"] + label_counts["ambiguous"]) / total_edges
        if total_edges
        else 0.0
    )

    return {
        "variant": variant,
        "total_edges": total_edges,
        "label_counts": label_counts,
        "strict_precision": strict_precision,
        "lenient_precision": lenient_precision,
        "metric_means_by_label": metric_means_by_label,
        "signal_diagnostics": signal_diagnostics,
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
        "signal_delta": {
            metric: noisefilter["signal_diagnostics"][metric][
                "correct_minus_wrong_mean"
            ]
            - baseline["signal_diagnostics"][metric]["correct_minus_wrong_mean"]
            for metric in METRIC_FIELDS
        },
    }


def build_report(variant_files: dict[str, str | Path]) -> dict[str, Any]:
    variants: dict[str, dict[str, Any]] = {}
    for variant, path in variant_files.items():
        variants[variant] = summarize_variant(variant, load_labeled_rows(path))

    return {
        "variants": variants,
        "comparison": build_comparison_summary(variants),
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
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
                **{
                    f"{metric}_correct_minus_wrong": summary["signal_diagnostics"][
                        metric
                    ]["correct_minus_wrong_mean"]
                    for metric in METRIC_FIELDS
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


def save_markdown(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Labeled Edge Sample Analysis",
        "",
        "| Variant | Total | Correct | Wrong | Ambiguous | Strict Precision | Lenient Precision |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for variant, summary in payload.get("variants", {}).items():
        label_counts = summary["label_counts"]
        lines.append(
            f"| {variant} | {summary['total_edges']} | {label_counts['correct']} | {label_counts['wrong']} | {label_counts['ambiguous']} | {summary['strict_precision']:.4f} | {summary['lenient_precision']:.4f} |"
        )

    comparison = payload.get("comparison", {})
    if comparison:
        lines.extend(
            [
                "",
                f"Strict precision delta (noisefilter - baseline): {comparison['precision_delta']['strict']:.4f}",
                f"Lenient precision delta (noisefilter - baseline): {comparison['precision_delta']['lenient']:.4f}",
                "",
                "## Signal Diagnostics",
                "",
            ]
        )

        for variant, summary in payload.get("variants", {}).items():
            lines.extend(
                [
                    f"### {variant}",
                    "",
                    "| Metric | Status | Correct Mean | Wrong Mean | Ambiguous Mean | Correct-Wrong |",
                    "| --- | --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for metric in METRIC_FIELDS:
                diagnosis = summary.get("signal_diagnostics", {}).get(
                    metric,
                    {
                        "status": "n/a",
                        "correct_mean": 0.0,
                        "wrong_mean": 0.0,
                        "ambiguous_mean": 0.0,
                        "correct_minus_wrong_mean": 0.0,
                    },
                )
                lines.append(
                    f"| {metric} | {diagnosis['status']} | {diagnosis['correct_mean']:.4f} | {diagnosis['wrong_mean']:.4f} | {diagnosis['ambiguous_mean']:.4f} | {diagnosis['correct_minus_wrong_mean']:.4f} |"
                )
            lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize manually labeled graph-edge samples and compare baseline vs NoiseFilter precision."
        )
    )
    parser.add_argument(
        "--variant-file",
        action="append",
        required=True,
        help="Variant mapping in the format variant=path/to/labeled.csv",
    )
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def _parse_variant_files(raw_pairs: list[str]) -> dict[str, Path]:
    variant_files: dict[str, Path] = {}
    for raw_pair in raw_pairs:
        if "=" not in raw_pair:
            raise ValueError(
                f"Invalid --variant-file value: {raw_pair}. Expected variant=path."
            )
        variant, path = raw_pair.split("=", 1)
        normalized_variant = variant.strip().lower()
        if not normalized_variant:
            raise ValueError(f"Invalid variant in --variant-file: {raw_pair}")
        variant_files[normalized_variant] = Path(path).expanduser()
    return variant_files


def main() -> None:
    args = parse_args()
    report = build_report(_parse_variant_files(args.variant_file))
    output_prefix = Path(args.output_prefix)

    save_json(output_prefix.with_suffix(".json"), report)
    save_csv(output_prefix.with_suffix(".csv"), report)
    save_markdown(output_prefix.with_suffix(".md"), report)

    print("Labeled edge analysis completed.")
    print(f"  json: {output_prefix.with_suffix('.json')}")
    print(f"  csv: {output_prefix.with_suffix('.csv')}")
    print(f"  markdown: {output_prefix.with_suffix('.md')}")
    for variant, summary in report["variants"].items():
        print(
            f"  {variant}: strict_precision={summary['strict_precision']:.4f} "
            f"lenient_precision={summary['lenient_precision']:.4f}"
        )


if __name__ == "__main__":
    main()
