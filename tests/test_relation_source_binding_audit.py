from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

from reproduce.audit_relation_source_binding import (
    audit_graph_edges,
    audit_labeled_rows,
    build_comparison,
    build_variant_report,
)


def test_audit_graph_edges_detects_mismatch_and_missing_endpoints() -> None:
    graph = nx.Graph()
    graph.add_edge(
        "Alice",
        "Acme",
        description="Alice works at Acme.",
        source_id="chunk-1",
    )
    graph.add_edge(
        "Bob",
        "Beta",
        description="Bob founded Beta.",
        source_id="chunk-2",
    )

    relation_chunks = {
        "Acme<SEP>Alice": {"chunk_ids": ["chunk-1"]},
        "Beta<SEP>Bob": {"chunk_ids": ["chunk-3"]},
    }
    text_chunks = {
        "chunk-1": "Alice works at Acme in 2020.",
        "chunk-2": "Completely unrelated sports article.",
        "chunk-3": "Completely unrelated finance article.",
    }

    report = audit_graph_edges(
        graph,
        relation_chunks,
        text_chunks,
        variant="baseline",
    )

    assert report["summary"]["total_edges"] == 2
    assert report["summary"]["tracking_mismatch_count"] == 1
    assert report["summary"]["weak_endpoint_binding_count"] == 1
    assert report["summary"]["both_endpoints_missing_count"] == 1
    assert any(row["tracking_mismatch"] for row in report["issue_rows"])


def test_audit_labeled_rows_summarizes_by_manual_label() -> None:
    rows = [
        {
            "variant": "baseline",
            "src": "Alice",
            "dst": "Acme",
            "description": "Alice works at Acme.",
            "manual_label": "correct",
            "chunk_ids": ["c1"],
            "source_chunks": ["Alice works at Acme in 2020."],
            "manual_notes": "",
        },
        {
            "variant": "baseline",
            "src": "Bob",
            "dst": "Beta",
            "description": "Bob founded Beta.",
            "manual_label": "wrong",
            "chunk_ids": ["c2"],
            "source_chunks": ["A sports event recap."],
            "manual_notes": "",
        },
    ]

    report = audit_labeled_rows(rows, variant="baseline")

    assert report["summary"]["total_rows"] == 2
    assert report["summary"]["by_label"]["correct"]["exported_weak_endpoint_binding_count"] == 0
    assert report["summary"]["by_label"]["wrong"]["exported_weak_endpoint_binding_count"] == 1
    assert report["summary"]["by_label"]["wrong"]["exported_both_endpoints_missing_count"] == 1


def test_build_variant_report_and_comparison(tmp_path: Path) -> None:
    working_dir = tmp_path / "baseline"
    working_dir.mkdir()

    graph = nx.Graph()
    graph.add_edge(
        "Alice",
        "Acme",
        description="Alice works at Acme.",
        source_id="chunk-1",
    )
    nx.write_graphml(graph, working_dir / "graph_chunk_entity_relation.graphml")

    (working_dir / "kv_store_relation_chunks.json").write_text(
        json.dumps({"Acme<SEP>Alice": {"chunk_ids": ["chunk-1"]}}),
        encoding="utf-8",
    )
    (working_dir / "kv_store_text_chunks.json").write_text(
        json.dumps({"chunk-1": {"content": "Alice works at Acme in 2020."}}),
        encoding="utf-8",
    )

    labeled_csv = tmp_path / "labeled.csv"
    labeled_csv.write_text(
        (
            "variant,src,dst,description,chunk_ids,source_chunks,manual_label,manual_notes\n"
            'baseline,Alice,Acme,Alice works at Acme.,"[""chunk-1""]","[""Alice works at Acme in 2020.""]",correct,\n'
        ),
        encoding="utf-8",
    )

    baseline_report = build_variant_report(
        variant="baseline",
        working_dir=working_dir,
        labeled_csv=labeled_csv,
    )
    noisefilter_report = {
        "variant": "noisefilter",
        "graph": {
            "weak_endpoint_binding_rate": 0.05,
            "both_endpoints_missing_rate": 0.01,
        },
        "labeled": {
            "by_label": {
                "correct": {
                    "exported_weak_endpoint_binding_rate": 0.0,
                    "exported_both_endpoints_missing_rate": 0.0,
                },
                "wrong": {
                    "exported_weak_endpoint_binding_rate": 0.5,
                    "exported_both_endpoints_missing_rate": 0.5,
                },
                "ambiguous": {
                    "exported_weak_endpoint_binding_rate": 0.4,
                    "exported_both_endpoints_missing_rate": 0.2,
                },
            }
        },
    }

    comparison = build_comparison(
        {
            "baseline": baseline_report,
            "noisefilter": noisefilter_report,
        }
    )

    assert baseline_report["graph"]["total_edges"] == 1
    assert comparison["graph"]["weak_endpoint_binding_rate_delta"] == 0.05
    assert comparison["labeled"]["wrong"]["both_endpoints_missing_rate_delta"] == 0.5
