from __future__ import annotations

from reproduce.audit_relation_directionality import (
    collect_findings_from_texts,
    summarize_findings,
)


def test_collect_findings_detects_directionality_risks() -> None:
    findings = collect_findings_from_texts(
        {
            "lightrag/kg/networkx_impl.py": "self._graph = preloaded_graph or nx.Graph()",
            "lightrag/utils.py": "return GRAPH_FIELD_SEP.join(sorted((src, tgt)))",
            "lightrag/noisefilter/confidence.py": (
                "edge_pair = tuple(sorted((src_id, tgt_id)))\n"
                "if {source, target} != {src_id, tgt_id}:"
            ),
            "reproduce/export_graph_edge_samples.py": 'edge_key = "<SEP>".join(sorted([src, dst]))',
        }
    )

    rule_ids = {finding["rule_id"] for finding in findings}
    assert "undirected_graph_storage" in rule_ids
    assert "sorted_relation_chunk_key" in rule_ids
    assert "sorted_confidence_edge_pair" in rule_ids
    assert "set_based_relation_match" in rule_ids
    assert "sorted_export_edge_key" in rule_ids


def test_summarize_findings_counts_severity() -> None:
    summary = summarize_findings(
        [
            {"severity": "high", "rule_id": "a"},
            {"severity": "medium", "rule_id": "b"},
            {"severity": "high", "rule_id": "c"},
        ]
    )

    assert summary["total_findings"] == 3
    assert summary["by_severity"] == {"high": 2, "medium": 1}
