from __future__ import annotations

import asyncio
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.utils import EmbeddingFunc, make_relation_chunk_key


async def _dummy_embedding(texts, **kwargs):
    vectors = []
    for text in texts:
        normalized = text.strip().lower()
        if normalized in {"alice", "person a"}:
            vectors.append(np.array([0.0, 0.0, 0.0], dtype=np.float32))
        elif normalized in {"acme", "company b"}:
            vectors.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        elif "works at" in normalized or "employ" in normalized:
            vectors.append(np.array([1.0, 0.0, 0.0], dtype=np.float32))
        elif "founded" in normalized:
            vectors.append(np.array([0.0, 1.0, 0.0], dtype=np.float32))
        else:
            vectors.append(np.array([0.5, 0.5, 0.0], dtype=np.float32))
    return np.stack(vectors)


def _build_embedding_func() -> EmbeddingFunc:
    return EmbeddingFunc(embedding_dim=3, max_token_size=512, func=_dummy_embedding)


async def _build_noise_storages(tmp_path: Path, workspace: str):
    initialize_share_data()
    embedding = _build_embedding_func()
    global_config = {
        "working_dir": str(tmp_path),
        "embedding_batch_num": 8,
        "vector_db_storage_cls_kwargs": {
            "cosine_better_than_threshold": 0.2,
        },
    }

    graph = NetworkXStorage(
        namespace="noise_graph",
        workspace=workspace,
        global_config=global_config,
        embedding_func=embedding,
    )
    relation_chunks = JsonKVStorage(
        namespace="noise_relation_chunks",
        workspace=workspace,
        global_config=global_config,
        embedding_func=embedding,
    )
    text_chunks = JsonKVStorage(
        namespace="noise_text_chunks",
        workspace=workspace,
        global_config=global_config,
        embedding_func=embedding,
    )
    llm_cache = JsonKVStorage(
        namespace="noise_llm_cache",
        workspace=workspace,
        global_config=global_config,
        embedding_func=embedding,
    )

    await graph.initialize()
    await relation_chunks.initialize()
    await text_chunks.initialize()
    await llm_cache.initialize()

    return graph, relation_chunks, text_chunks, llm_cache, embedding


def _relationship_cache(src: str, tgt: str, keywords: str, description: str) -> str:
    return "\n".join(
        [
            f"entity<|#|>{src}<|#|>person<|#|>{src} is a person.",
            f"entity<|#|>{tgt}<|#|>organization<|#|>{tgt} is an organization.",
            f"relation<|#|>{src}<|#|>{tgt}<|#|>{keywords}<|#|>{description}",
            "<|COMPLETE|>",
        ]
    )


async def _seed_edge_evidence(
    *,
    graph,
    relation_chunks,
    text_chunks,
    llm_cache,
    src: str,
    tgt: str,
    edge_keywords: str,
    edge_description: str,
    chunk_payloads: list[tuple[str, str]],
) -> None:
    chunk_ids = []
    text_chunk_payload = {}
    cache_payload = {}

    for index, (keywords, description) in enumerate(chunk_payloads, start=1):
        chunk_id = f"chunk-{index}"
        cache_id = f"cache-{index}"
        chunk_ids.append(chunk_id)
        text_chunk_payload[chunk_id] = {
            "content": f"{src} {keywords} {tgt}",
            "llm_cache_list": [cache_id],
            "file_path": "test.md",
        }
        cache_payload[cache_id] = {
            "cache_type": "extract",
            "chunk_id": chunk_id,
            "return": _relationship_cache(src, tgt, keywords, description),
            "create_time": index,
        }

    await graph.upsert_node(src, {"entity_id": src, "description": src})
    await graph.upsert_node(tgt, {"entity_id": tgt, "description": tgt})
    await graph.upsert_edge(
        src,
        tgt,
        {
            "keywords": edge_keywords,
            "description": edge_description,
            "source_id": "<SEP>".join(chunk_ids),
            "weight": 1.0,
        },
    )
    await relation_chunks.upsert(
        {
            make_relation_chunk_key(src, tgt): {
                "chunk_ids": chunk_ids,
                "count": len(chunk_ids),
            }
        }
    )
    await text_chunks.upsert(text_chunk_payload)
    await llm_cache.upsert(cache_payload)


@pytest.mark.offline
def test_confidence_scoring_updates_graph_with_chunk_support(tmp_path):
    from lightrag.noisefilter.confidence import ConfidenceScoringEngine

    async def _run():
        graph, relation_chunks, text_chunks, llm_cache, embedding = (
            await _build_noise_storages(tmp_path, "consistent")
        )
        await _seed_edge_evidence(
            graph=graph,
            relation_chunks=relation_chunks,
            text_chunks=text_chunks,
            llm_cache=llm_cache,
            src="Alice",
            tgt="Acme",
            edge_keywords="works at",
            edge_description="Alice works at Acme.",
            chunk_payloads=[
                ("works at", "Alice works at Acme."),
                ("works at", "Alice is employed by Acme."),
            ],
        )

        scorer = ConfidenceScoringEngine(
            embedding_model=embedding,
            relation_chunks_storage=relation_chunks,
            text_chunks_storage=text_chunks,
            llm_response_cache=llm_cache,
        )
        result = await scorer.score_and_update_graph(
            graph, edge_pairs=[("Alice", "Acme")]
        )
        return result, await graph.get_edge("Alice", "Acme")

    result, edge = asyncio.run(_run())
    score_info = result[("Alice", "Acme")]

    assert score_info["support_count"] == 2
    assert score_info["freq_score"] == pytest.approx(2 / 3, rel=1e-3)
    assert score_info["consistency_score"] == pytest.approx(1.0)
    assert score_info["semantic_score"] == pytest.approx(1.0)
    assert score_info["conf_score"] > 0.8
    assert edge["conf_score"] == pytest.approx(score_info["conf_score"])
    assert edge["conf_support"] == 2


@pytest.mark.offline
def test_confidence_scoring_penalizes_conflicting_relations(tmp_path):
    from lightrag.noisefilter.confidence import ConfidenceScoringEngine

    async def _run():
        graph, relation_chunks, text_chunks, llm_cache, embedding = (
            await _build_noise_storages(tmp_path, "conflicting")
        )
        await _seed_edge_evidence(
            graph=graph,
            relation_chunks=relation_chunks,
            text_chunks=text_chunks,
            llm_cache=llm_cache,
            src="Alice",
            tgt="Acme",
            edge_keywords="works at",
            edge_description="Alice works at Acme.",
            chunk_payloads=[
                ("works at", "Alice works at Acme."),
                ("founded", "Alice founded Acme."),
            ],
        )

        scorer = ConfidenceScoringEngine(
            embedding_model=embedding,
            relation_chunks_storage=relation_chunks,
            text_chunks_storage=text_chunks,
            llm_response_cache=llm_cache,
        )
        return await scorer.score_and_update_graph(
            graph, edge_pairs=[("Alice", "Acme")]
        )

    result = asyncio.run(_run())

    score_info = result[("Alice", "Acme")]
    assert score_info["support_count"] == 2
    assert score_info["freq_score"] == pytest.approx(1 / 3, rel=1e-3)
    assert score_info["consistency_score"] == pytest.approx(0.5)
    assert score_info["conf_score"] < 0.6


@pytest.mark.offline
def test_noise_aware_retriever_soft_and_hard_modes():
    from lightrag.noisefilter.retriever import NoiseAwareRetriever

    local_edges = [
        {
            "src_id": "A",
            "tgt_id": "B",
            "rank": 5,
            "weight": 1.0,
            "conf_score": 0.2,
        },
        {
            "src_id": "A",
            "tgt_id": "C",
            "rank": 1,
            "weight": 1.0,
            "conf_score": 0.9,
        },
    ]

    soft = NoiseAwareRetriever(conf_threshold=0.4, soft_mode=True)
    hard = NoiseAwareRetriever(conf_threshold=0.4, soft_mode=False)

    soft_ranked = soft.rank_local_edges(local_edges)
    hard_ranked = hard.rank_local_edges(local_edges)

    assert [edge["tgt_id"] for edge in soft_ranked] == ["C", "B"]
    assert (
        soft_ranked[0]["noise_adjusted_score"]
        > soft_ranked[1]["noise_adjusted_score"]
    )
    assert [edge["tgt_id"] for edge in hard_ranked] == ["C"]

    global_edges = [
        {"src_id": "A", "tgt_id": "B", "distance": 0.9, "conf_score": 0.2},
        {"src_id": "A", "tgt_id": "C", "distance": 0.4, "conf_score": 0.9},
    ]
    assert [edge["tgt_id"] for edge in soft.rank_global_edges(global_edges)] == [
        "C",
        "B",
    ]


@pytest.mark.offline
def test_noise_injector_and_evaluate_filter():
    from lightrag.noisefilter.benchmark import NoiseInjector, evaluate_filter

    graph = nx.MultiDiGraph()
    graph.add_edge("A", "B", key="clean-1", relation="connected_to", conf_score=0.9)
    graph.add_edge("B", "C", key="clean-2", relation="connected_to", conf_score=0.9)
    graph.add_edge("C", "D", key="clean-3", relation="connected_to", conf_score=0.9)
    graph.add_edge("D", "A", key="clean-4", relation="connected_to", conf_score=0.9)

    injector = NoiseInjector(graph, noise_ratio=0.5, seed=7).inject_random_relations()
    metrics = evaluate_filter(
        injector.graph,
        injector.get_ground_truth(),
        conf_threshold=0.2,
    )

    assert injector.injected_edges
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
