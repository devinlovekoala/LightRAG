from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce.evaluate_grounding_signals import (
    LABELS,
    compute_anchor_scores,
    load_edge_rows,
    load_text_chunks,
    resolve_source_texts,
)


def load_json_dict(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def normalize_chunk_ids(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [item for item in raw.split("<SEP>") if item]
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, str) and item]
    return []


def make_relation_key(src: str, dst: str) -> str:
    return "<SEP>".join(sorted([src, dst]))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def audit_graph_edges(
    graph: nx.Graph,
    relation_chunks: dict[str, Any],
    text_chunks: dict[str, str],
    *,
    variant: str,
) -> dict[str, Any]:
    issue_rows: list[dict[str, Any]] = []
    totals = {
        "total_edges": 0,
        "tracking_mismatch_count": 0,
        "missing_relation_tracking_count": 0,
        "missing_graph_source_count": 0,
        "weak_endpoint_binding_count": 0,
        "both_endpoints_missing_count": 0,
        "empty_resolved_source_count": 0,
    }

    for src, dst, data in graph.edges(data=True):
        totals["total_edges"] += 1
        storage_key = make_relation_key(src, dst)
        relation_entry = relation_chunks.get(storage_key, {})
        relation_chunk_ids = normalize_chunk_ids(relation_entry.get("chunk_ids"))
        graph_chunk_ids = normalize_chunk_ids(data.get("source_id"))
        resolved_chunk_ids = relation_chunk_ids or graph_chunk_ids
        source_texts = [text_chunks.get(chunk_id, "") for chunk_id in resolved_chunk_ids]
        source_texts = [text for text in source_texts if text]
        anchor_scores = compute_anchor_scores(src, dst, source_texts)

        tracking_mismatch = bool(relation_chunk_ids and graph_chunk_ids and relation_chunk_ids != graph_chunk_ids)
        missing_relation_tracking = not relation_chunk_ids
        missing_graph_source = not graph_chunk_ids
        empty_resolved_source = not source_texts
        weak_endpoint_binding = (
            anchor_scores["src_anchor_score"] <= 0.0
            or anchor_scores["dst_anchor_score"] <= 0.0
        )
        both_endpoints_missing = (
            anchor_scores["src_anchor_score"] <= 0.0
            and anchor_scores["dst_anchor_score"] <= 0.0
        )

        if tracking_mismatch:
            totals["tracking_mismatch_count"] += 1
        if missing_relation_tracking:
            totals["missing_relation_tracking_count"] += 1
        if missing_graph_source:
            totals["missing_graph_source_count"] += 1
        if weak_endpoint_binding:
            totals["weak_endpoint_binding_count"] += 1
        if both_endpoints_missing:
            totals["both_endpoints_missing_count"] += 1
        if empty_resolved_source:
            totals["empty_resolved_source_count"] += 1

        if (
            tracking_mismatch
            or missing_relation_tracking
            or missing_graph_source
            or weak_endpoint_binding
            or empty_resolved_source
        ):
            issue_rows.append(
                {
                    "variant": variant,
                    "scope": "graph",
                    "src": src,
                    "dst": dst,
                    "description": str(data.get("description", "")),
                    "relation_key": storage_key,
                    "graph_chunk_ids": json.dumps(graph_chunk_ids, ensure_ascii=False),
                    "tracking_chunk_ids": json.dumps(relation_chunk_ids, ensure_ascii=False),
                    "resolved_chunk_ids": json.dumps(resolved_chunk_ids, ensure_ascii=False),
                    "src_anchor_score": anchor_scores["src_anchor_score"],
                    "dst_anchor_score": anchor_scores["dst_anchor_score"],
                    "endpoint_anchor_score": anchor_scores["endpoint_anchor_score"],
                    "cooccurrence_score": anchor_scores["cooccurrence_score"],
                    "tracking_mismatch": tracking_mismatch,
                    "missing_relation_tracking": missing_relation_tracking,
                    "missing_graph_source": missing_graph_source,
                    "weak_endpoint_binding": weak_endpoint_binding,
                    "both_endpoints_missing": both_endpoints_missing,
                    "empty_resolved_source": empty_resolved_source,
                    "source_excerpt": " ".join(source_texts)[:240],
                }
            )

    total_edges = totals["total_edges"]
    summary = {
        **totals,
        "tracking_mismatch_rate": totals["tracking_mismatch_count"] / total_edges if total_edges else 0.0,
        "weak_endpoint_binding_rate": totals["weak_endpoint_binding_count"] / total_edges if total_edges else 0.0,
        "both_endpoints_missing_rate": totals["both_endpoints_missing_count"] / total_edges if total_edges else 0.0,
    }
    return {"summary": summary, "issue_rows": issue_rows}


def audit_labeled_rows(
    rows: list[dict[str, Any]],
    *,
    text_chunk_lookup: dict[str, str] | None = None,
    variant: str,
) -> dict[str, Any]:
    filtered_rows = [
        row
        for row in rows
        if str(row.get("manual_label", "")).strip().lower() in LABELS
    ]

    issue_rows: list[dict[str, Any]] = []
    by_label: dict[str, dict[str, Any]] = {}
    for label in LABELS:
        labeled_rows = [row for row in filtered_rows if row["manual_label"] == label]
        exported_weak_count = 0
        exported_both_missing_count = 0
        exported_empty_source_count = 0
        resolved_weak_count = 0
        resolved_both_missing_count = 0
        resolved_empty_source_count = 0
        for row in labeled_rows:
            exported_source_texts = [
                str(chunk) for chunk in row.get("source_chunks", []) if str(chunk).strip()
            ]
            resolved_source_texts = resolve_source_texts(row, text_chunk_lookup)
            exported_anchor_scores = compute_anchor_scores(
                row["src"], row["dst"], exported_source_texts
            )
            resolved_anchor_scores = compute_anchor_scores(
                row["src"], row["dst"], resolved_source_texts
            )
            exported_weak_endpoint_binding = (
                exported_anchor_scores["src_anchor_score"] <= 0.0
                or exported_anchor_scores["dst_anchor_score"] <= 0.0
            )
            exported_both_endpoints_missing = (
                exported_anchor_scores["src_anchor_score"] <= 0.0
                and exported_anchor_scores["dst_anchor_score"] <= 0.0
            )
            resolved_weak_endpoint_binding = (
                resolved_anchor_scores["src_anchor_score"] <= 0.0
                or resolved_anchor_scores["dst_anchor_score"] <= 0.0
            )
            resolved_both_endpoints_missing = (
                resolved_anchor_scores["src_anchor_score"] <= 0.0
                and resolved_anchor_scores["dst_anchor_score"] <= 0.0
            )
            exported_empty_source = not exported_source_texts
            resolved_empty_source = not resolved_source_texts

            if exported_weak_endpoint_binding:
                exported_weak_count += 1
            if exported_both_endpoints_missing:
                exported_both_missing_count += 1
            if exported_empty_source:
                exported_empty_source_count += 1
            if resolved_weak_endpoint_binding:
                resolved_weak_count += 1
            if resolved_both_endpoints_missing:
                resolved_both_missing_count += 1
            if resolved_empty_source:
                resolved_empty_source_count += 1

            if (
                exported_weak_endpoint_binding
                or resolved_weak_endpoint_binding
                or exported_empty_source
                or resolved_empty_source
            ):
                issue_rows.append(
                    {
                        "variant": variant,
                        "scope": "labeled",
                        "manual_label": label,
                        "src": row["src"],
                        "dst": row["dst"],
                        "description": row.get("description", ""),
                        "chunk_ids": json.dumps(row.get("chunk_ids", []), ensure_ascii=False),
                        "exported_src_anchor_score": exported_anchor_scores["src_anchor_score"],
                        "exported_dst_anchor_score": exported_anchor_scores["dst_anchor_score"],
                        "exported_endpoint_anchor_score": exported_anchor_scores["endpoint_anchor_score"],
                        "exported_cooccurrence_score": exported_anchor_scores["cooccurrence_score"],
                        "resolved_src_anchor_score": resolved_anchor_scores["src_anchor_score"],
                        "resolved_dst_anchor_score": resolved_anchor_scores["dst_anchor_score"],
                        "resolved_endpoint_anchor_score": resolved_anchor_scores["endpoint_anchor_score"],
                        "resolved_cooccurrence_score": resolved_anchor_scores["cooccurrence_score"],
                        "exported_weak_endpoint_binding": exported_weak_endpoint_binding,
                        "exported_both_endpoints_missing": exported_both_endpoints_missing,
                        "exported_empty_source": exported_empty_source,
                        "resolved_weak_endpoint_binding": resolved_weak_endpoint_binding,
                        "resolved_both_endpoints_missing": resolved_both_endpoints_missing,
                        "resolved_empty_source": resolved_empty_source,
                        "source_excerpt": " ".join(exported_source_texts)[:240],
                        "resolved_source_excerpt": " ".join(resolved_source_texts)[:240],
                        "manual_notes": row.get("manual_notes", ""),
                    }
                )

        count = len(labeled_rows)
        by_label[label] = {
            "count": count,
            "exported_weak_endpoint_binding_count": exported_weak_count,
            "exported_both_endpoints_missing_count": exported_both_missing_count,
            "exported_empty_source_count": exported_empty_source_count,
            "exported_weak_endpoint_binding_rate": exported_weak_count / count if count else 0.0,
            "exported_both_endpoints_missing_rate": exported_both_missing_count / count if count else 0.0,
            "resolved_weak_endpoint_binding_count": resolved_weak_count,
            "resolved_both_endpoints_missing_count": resolved_both_missing_count,
            "resolved_empty_source_count": resolved_empty_source_count,
            "resolved_weak_endpoint_binding_rate": resolved_weak_count / count if count else 0.0,
            "resolved_both_endpoints_missing_rate": resolved_both_missing_count / count if count else 0.0,
        }

    return {
        "summary": {
            "total_rows": len(filtered_rows),
            "by_label": by_label,
        },
        "issue_rows": issue_rows,
    }


def build_variant_report(
    *,
    variant: str,
    working_dir: Path,
    labeled_csv: Path | None = None,
) -> dict[str, Any]:
    graph = nx.read_graphml(working_dir / "graph_chunk_entity_relation.graphml")
    relation_chunks = load_json_dict(working_dir / "kv_store_relation_chunks.json")
    text_chunks = load_text_chunks(working_dir / "kv_store_text_chunks.json")

    graph_audit = audit_graph_edges(
        graph,
        relation_chunks,
        text_chunks,
        variant=variant,
    )

    labeled_audit: dict[str, Any] | None = None
    if labeled_csv is not None:
        labeled_audit = audit_labeled_rows(
            load_edge_rows(labeled_csv),
            text_chunk_lookup=text_chunks,
            variant=variant,
        )

    return {
        "variant": variant,
        "working_dir": str(working_dir),
        "graph": graph_audit["summary"],
        "labeled": None if labeled_audit is None else labeled_audit["summary"],
        "issue_rows": graph_audit["issue_rows"]
        + ([] if labeled_audit is None else labeled_audit["issue_rows"]),
    }


def build_comparison(variants: dict[str, dict[str, Any]]) -> dict[str, Any]:
    baseline = variants.get("baseline")
    noisefilter = variants.get("noisefilter")
    if baseline is None or noisefilter is None:
        return {}

    comparison = {
        "graph": {
            "weak_endpoint_binding_rate_delta": (
                _safe_float(noisefilter["graph"]["weak_endpoint_binding_rate"])
                - _safe_float(baseline["graph"]["weak_endpoint_binding_rate"])
            ),
            "both_endpoints_missing_rate_delta": (
                _safe_float(noisefilter["graph"]["both_endpoints_missing_rate"])
                - _safe_float(baseline["graph"]["both_endpoints_missing_rate"])
            ),
        }
    }

    if baseline.get("labeled") and noisefilter.get("labeled"):
        comparison["labeled"] = {}
        for label in LABELS:
            comparison["labeled"][label] = {
                "weak_endpoint_binding_rate_delta": (
                    _safe_float(
                        noisefilter["labeled"]["by_label"][label]["exported_weak_endpoint_binding_rate"]
                    )
                    - _safe_float(
                        baseline["labeled"]["by_label"][label]["exported_weak_endpoint_binding_rate"]
                    )
                ),
                "both_endpoints_missing_rate_delta": (
                    _safe_float(
                        noisefilter["labeled"]["by_label"][label]["exported_both_endpoints_missing_rate"]
                    )
                    - _safe_float(
                        baseline["labeled"]["by_label"][label]["exported_both_endpoints_missing_rate"]
                    )
                ),
            }
    return comparison


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_csv(path: str | Path, variants: dict[str, dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for variant, payload in variants.items():
        row = {
            "variant": variant,
            **{f"graph_{key}": value for key, value in payload["graph"].items()},
        }
        labeled = payload.get("labeled")
        if labeled:
            row["labeled_total_rows"] = labeled["total_rows"]
            for label in LABELS:
                for key, value in labeled["by_label"][label].items():
                    row[f"labeled_{label}_{key}"] = value
        rows.append(row)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_rows_csv(path: str | Path, variants: dict[str, dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for payload in variants.values():
        rows.extend(payload.get("issue_rows", []))
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Relation Source-Binding Audit",
        "",
        "| Variant | Graph Edges | Weak Binding | Weak Rate | Both Missing | Both Missing Rate | Tracking Mismatch |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant, summary in payload["variants"].items():
        graph = summary["graph"]
        lines.append(
            f"| {variant} | {graph['total_edges']} | {graph['weak_endpoint_binding_count']} | {graph['weak_endpoint_binding_rate']:.4f} | {graph['both_endpoints_missing_count']} | {graph['both_endpoints_missing_rate']:.4f} | {graph['tracking_mismatch_count']} |"
        )

    if payload.get("comparison"):
        comparison = payload["comparison"]
        lines.extend(
            [
                "",
                f"Graph weak binding rate delta (noisefilter - baseline): {comparison['graph']['weak_endpoint_binding_rate_delta']:.4f}",
                f"Graph both-endpoints-missing rate delta (noisefilter - baseline): {comparison['graph']['both_endpoints_missing_rate_delta']:.4f}",
            ]
        )
        labeled = comparison.get("labeled")
        if labeled:
            lines.append("")
            lines.append("| Label | Weak Rate Delta | Both Missing Rate Delta |")
            lines.append("| --- | ---: | ---: |")
            for label in LABELS:
                lines.append(
                    f"| {label} | {labeled[label]['weak_endpoint_binding_rate_delta']:.4f} | {labeled[label]['both_endpoints_missing_rate_delta']:.4f} |"
                )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


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
        description="Audit relation source-binding quality across formal LightRAG working directories.",
    )
    parser.add_argument("--working-dir", action="append", required=True)
    parser.add_argument("--labeled-csv", action="append", default=[])
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    working_dirs = _parse_mapping(args.working_dir)
    labeled_csvs = _parse_mapping(args.labeled_csv)

    variants: dict[str, dict[str, Any]] = {}
    for variant, working_dir in working_dirs.items():
        variants[variant] = build_variant_report(
            variant=variant,
            working_dir=working_dir,
            labeled_csv=labeled_csvs.get(variant),
        )

    report = {
        "variants": variants,
        "comparison": build_comparison(variants),
    }

    output_prefix = Path(args.output_prefix)
    save_json(output_prefix.with_suffix(".json"), report)
    save_csv(output_prefix.with_suffix(".csv"), variants)
    save_rows_csv(output_prefix.with_suffix(".rows.csv"), variants)
    save_markdown(output_prefix.with_suffix(".md"), report)

    print("Relation source-binding audit completed.")
    print(f"  json: {output_prefix.with_suffix('.json')}")
    print(f"  csv: {output_prefix.with_suffix('.csv')}")
    print(f"  markdown: {output_prefix.with_suffix('.md')}")
    print(f"  rows: {output_prefix.with_suffix('.rows.csv')}")
    for variant, payload in variants.items():
        graph = payload["graph"]
        print(
            f"  {variant}: weak_rate={graph['weak_endpoint_binding_rate']:.4f} both_missing_rate={graph['both_endpoints_missing_rate']:.4f} tracking_mismatch={graph['tracking_mismatch_count']}"
        )


if __name__ == "__main__":
    main()
