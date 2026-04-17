from __future__ import annotations

import networkx as nx

from reproduce.export_graph_edge_samples import normalize_chunk_ids, sample_edges


def test_normalize_chunk_ids_supports_sep_string_and_list():
    assert normalize_chunk_ids("chunk-1<SEP>chunk-2") == ["chunk-1", "chunk-2"]
    assert normalize_chunk_ids(["chunk-1", "chunk-2", 3]) == ["chunk-1", "chunk-2"]


def test_sample_edges_includes_chunk_context_and_confidence():
    graph = nx.Graph()
    graph.add_edge(
        "Alice",
        "Acme",
        description="Alice worked at Acme.",
        keywords="employment",
        weight=1.0,
        conf_score=0.75,
        conf_freq_score=0.5,
        conf_consistency_score=1.0,
        conf_semantic_score=0.8,
        conf_support=2,
        source_id="chunk-1",
    )

    relation_chunks = {
        "Acme<SEP>Alice": {
            "chunk_ids": ["chunk-1", "chunk-2"],
        }
    }
    text_chunks = {
        "chunk-1": {"content": "Alice joined Acme in 2020."},
        "chunk-2": {"content": "Acme promoted Alice in 2021."},
    }

    rows = sample_edges(
        graph,
        relation_chunks,
        text_chunks,
        variant="noisefilter",
        sample_size=10,
        seed=42,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["src"] == "Alice"
    assert row["dst"] == "Acme"
    assert row["conf_score"] == 0.75
    assert row["chunk_ids"] == ["chunk-1", "chunk-2"]
    assert row["source_chunks"][0] == "Alice joined Acme in 2020."


def test_sample_edges_keeps_all_source_chunks_for_manual_review():
    graph = nx.Graph()
    graph.add_edge(
        "Alice",
        "Acme",
        description="Alice worked at Acme.",
        keywords="employment",
        weight=1.0,
        conf_score=0.75,
        source_id="chunk-1<SEP>chunk-2<SEP>chunk-3<SEP>chunk-4",
    )

    relation_chunks = {
        "Acme<SEP>Alice": {
            "chunk_ids": ["chunk-1", "chunk-2", "chunk-3", "chunk-4"],
        }
    }
    text_chunks = {
        "chunk-1": {"content": "Chunk 1"},
        "chunk-2": {"content": "Chunk 2"},
        "chunk-3": {"content": "Chunk 3"},
        "chunk-4": {"content": "Chunk 4"},
    }

    rows = sample_edges(
        graph,
        relation_chunks,
        text_chunks,
        variant="baseline",
        sample_size=10,
        seed=42,
    )

    assert rows[0]["chunk_ids"] == ["chunk-1", "chunk-2", "chunk-3", "chunk-4"]
    assert rows[0]["source_chunks"] == ["Chunk 1", "Chunk 2", "Chunk 3", "Chunk 4"]
