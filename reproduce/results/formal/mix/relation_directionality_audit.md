# Relation Directionality Audit

- total_findings: 6

| Severity | Rule ID | Path | Message |
| --- | --- | --- | --- |
| high | undirected_graph_storage | lightrag/kg/networkx_impl.py | NetworkX graph storage is undirected, so edge direction cannot be preserved. |
| high | sorted_relation_chunk_key | lightrag/utils.py | Relation chunk storage key sorts endpoints, collapsing direction. |
| high | sorted_confidence_edge_pair | lightrag/noisefilter/confidence.py | Confidence evidence collection normalizes edge pairs as undirected. |
| high | set_based_relation_match | lightrag/noisefilter/confidence.py | Relation mention parsing matches source and target as an unordered set. |
| medium | reverse_edge_fallback | lightrag/noisefilter/confidence.py | Confidence lookup falls back to reversed edges, masking directionality issues. |
| medium | sorted_export_edge_key | reproduce/export_graph_edge_samples.py | Manual edge export collapses relation chunk lookup into an undirected key. |
