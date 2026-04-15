from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


RULES = (
    {
        "rule_id": "undirected_graph_storage",
        "severity": "high",
        "path": "lightrag/kg/networkx_impl.py",
        "pattern": "nx.Graph()",
        "message": "NetworkX graph storage is undirected, so edge direction cannot be preserved.",
    },
    {
        "rule_id": "sorted_relation_chunk_key",
        "severity": "high",
        "path": "lightrag/utils.py",
        "pattern": "sorted((src, tgt))",
        "message": "Relation chunk storage key sorts endpoints, collapsing direction.",
    },
    {
        "rule_id": "sorted_confidence_edge_pair",
        "severity": "high",
        "path": "lightrag/noisefilter/confidence.py",
        "pattern": "tuple(sorted((src_id, tgt_id)))",
        "message": "Confidence evidence collection normalizes edge pairs as undirected.",
    },
    {
        "rule_id": "set_based_relation_match",
        "severity": "high",
        "path": "lightrag/noisefilter/confidence.py",
        "pattern": "if {source, target} != {src_id, tgt_id}:",
        "message": "Relation mention parsing matches source and target as an unordered set.",
    },
    {
        "rule_id": "reverse_edge_fallback",
        "severity": "medium",
        "path": "lightrag/noisefilter/confidence.py",
        "pattern": "edge_data = await graph_storage.get_edge(tgt_id, src_id)",
        "message": "Confidence lookup falls back to reversed edges, masking directionality issues.",
    },
    {
        "rule_id": "sorted_export_edge_key",
        "severity": "medium",
        "path": "reproduce/export_graph_edge_samples.py",
        "pattern": 'edge_key = "<SEP>".join(sorted([src, dst]))',
        "message": "Manual edge export collapses relation chunk lookup into an undirected key.",
    },
)


def collect_findings_from_texts(text_by_path: dict[str, str]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for rule in RULES:
        content = text_by_path.get(rule["path"], "")
        if rule["pattern"] in content:
            findings.append(
                {
                    "rule_id": rule["rule_id"],
                    "severity": rule["severity"],
                    "path": rule["path"],
                    "pattern": rule["pattern"],
                    "message": rule["message"],
                }
            )
    return findings


def summarize_findings(findings: list[dict[str, Any]]) -> dict[str, Any]:
    by_severity: dict[str, int] = {}
    for finding in findings:
        severity = str(finding["severity"])
        by_severity[severity] = by_severity.get(severity, 0) + 1
    return {
        "total_findings": len(findings),
        "by_severity": by_severity,
    }


def audit_repo(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root)
    text_by_path: dict[str, str] = {}
    for rule in RULES:
        file_path = root / rule["path"]
        text_by_path[rule["path"]] = (
            file_path.read_text(encoding="utf-8") if file_path.exists() else ""
        )
    findings = collect_findings_from_texts(text_by_path)
    return {
        "summary": summarize_findings(findings),
        "findings": findings,
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_csv(path: str | Path, findings: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not findings:
        output_path.write_text("", encoding="utf-8")
        return
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(findings[0].keys()))
        writer.writeheader()
        writer.writerows(findings)


def save_markdown(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Relation Directionality Audit",
        "",
        f"- total_findings: {payload['summary']['total_findings']}",
        "",
        "| Severity | Rule ID | Path | Message |",
        "| --- | --- | --- | --- |",
    ]
    for finding in payload["findings"]:
        lines.append(
            f"| {finding['severity']} | {finding['rule_id']} | {finding['path']} | {finding['message']} |"
        )
    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit relation directionality and source-binding risks in the current LightRAG pipeline.",
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = audit_repo(args.repo_root)
    output_prefix = Path(args.output_prefix)
    save_json(output_prefix.with_suffix(".json"), report)
    save_csv(output_prefix.with_suffix(".csv"), report["findings"])
    save_markdown(output_prefix.with_suffix(".md"), report)
    print("Relation directionality audit completed.")
    print(f"  json: {output_prefix.with_suffix('.json')}")
    print(f"  csv: {output_prefix.with_suffix('.csv')}")
    print(f"  markdown: {output_prefix.with_suffix('.md')}")
    print(f"  total_findings: {report['summary']['total_findings']}")


if __name__ == "__main__":
    main()
