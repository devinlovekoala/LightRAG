from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lightrag.kg.json_kv_impl import JsonKVStorage
from lightrag.kg.networkx_impl import NetworkXStorage
from lightrag.kg.shared_storage import initialize_share_data
from lightrag.noisefilter import (
    ConfidenceScoringEngine,
    NoiseAwareRetriever,
    NoiseInjector,
    evaluate_filter,
)
from lightrag.utils import EmbeddingFunc, make_relation_chunk_key


async def demo_embedding(texts, **kwargs):
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


def relationship_cache(src: str, tgt: str, keywords: str, description: str) -> str:
    return "\n".join(
        [
            f"entity<|#|>{src}<|#|>person<|#|>{src} is a person.",
            f"entity<|#|>{tgt}<|#|>organization<|#|>{tgt} is an organization.",
            f"relation<|#|>{src}<|#|>{tgt}<|#|>{keywords}<|#|>{description}",
            "<|COMPLETE|>",
        ]
    )


async def seed_edge(
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
    chunk_updates = {}
    cache_updates = {}

    for index, (keywords, description) in enumerate(chunk_payloads, start=1):
        chunk_id = f"{src}-{tgt}-chunk-{index}"
        cache_id = f"{src}-{tgt}-cache-{index}"
        chunk_ids.append(chunk_id)
        chunk_updates[chunk_id] = {
            "content": f"{src} {keywords} {tgt}",
            "llm_cache_list": [cache_id],
            "file_path": "demo.md",
        }
        cache_updates[cache_id] = {
            "cache_type": "extract",
            "chunk_id": chunk_id,
            "return": relationship_cache(src, tgt, keywords, description),
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
            "source_id": "|".join(chunk_ids),
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
    await text_chunks.upsert(chunk_updates)
    await llm_cache.upsert(cache_updates)


async def main():
    initialize_share_data()
    embedding = EmbeddingFunc(embedding_dim=3, max_token_size=512, func=demo_embedding)

    with tempfile.TemporaryDirectory(prefix="lightrag-noisefilter-") as temp_dir:
        global_config = {
            "working_dir": temp_dir,
            "embedding_batch_num": 8,
            "vector_db_storage_cls_kwargs": {
                "cosine_better_than_threshold": 0.2,
            },
        }

        graph = NetworkXStorage(
            namespace="noise_demo_graph",
            workspace="demo",
            global_config=global_config,
            embedding_func=embedding,
        )
        relation_chunks = JsonKVStorage(
            namespace="noise_demo_relation_chunks",
            workspace="demo",
            global_config=global_config,
            embedding_func=embedding,
        )
        text_chunks = JsonKVStorage(
            namespace="noise_demo_text_chunks",
            workspace="demo",
            global_config=global_config,
            embedding_func=embedding,
        )
        llm_cache = JsonKVStorage(
            namespace="noise_demo_llm_cache",
            workspace="demo",
            global_config=global_config,
            embedding_func=embedding,
        )

        for storage in (graph, relation_chunks, text_chunks, llm_cache):
            await storage.initialize()

        await seed_edge(
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
        await seed_edge(
            graph=graph,
            relation_chunks=relation_chunks,
            text_chunks=text_chunks,
            llm_cache=llm_cache,
            src="Alice",
            tgt="Beta",
            edge_keywords="works at",
            edge_description="Alice works at Beta.",
            chunk_payloads=[
                ("works at", "Alice works at Beta."),
                ("founded", "Alice founded Beta."),
            ],
        )

        scorer = ConfidenceScoringEngine(
            embedding_model=embedding,
            relation_chunks_storage=relation_chunks,
            text_chunks_storage=text_chunks,
            llm_response_cache=llm_cache,
        )
        scores = await scorer.score_and_update_graph(graph)

        print("\nConfidence scores")
        for edge_pair, score in sorted(scores.items()):
            print(
                f"  {edge_pair[0]} -> {edge_pair[1]}: conf={score['conf_score']:.3f} "
                f"(freq={score['freq_score']:.3f}, cons={score['consistency_score']:.3f}, sem={score['semantic_score']:.3f})"
            )

        edges = []
        for edge in await graph.get_all_edges():
            edges.append(
                {
                    "src_id": edge["source"],
                    "tgt_id": edge["target"],
                    "rank": 1.0,
                    "weight": edge.get("weight", 1.0),
                    "conf_score": edge.get("conf_score", 0.5),
                }
            )

        print("\nSoft vs hard retrieval")
        soft = NoiseAwareRetriever(conf_threshold=0.5, soft_mode=True)
        hard = NoiseAwareRetriever(conf_threshold=0.5, soft_mode=False)
        print("  soft :", [edge["tgt_id"] for edge in soft.rank_local_edges(edges)])
        print("  hard :", [edge["tgt_id"] for edge in hard.rank_local_edges(edges)])

        benchmark_graph = nx.MultiDiGraph()
        benchmark_graph.add_edge(
            "A", "B", key="clean-1", relation="related", conf_score=0.9
        )
        benchmark_graph.add_edge(
            "B", "C", key="clean-2", relation="related", conf_score=0.9
        )
        benchmark_graph.add_edge(
            "C", "D", key="clean-3", relation="related", conf_score=0.9
        )
        benchmark_graph.add_edge(
            "D", "A", key="clean-4", relation="related", conf_score=0.9
        )

        injector = NoiseInjector(benchmark_graph, noise_ratio=0.5, seed=7).inject_random_relations()
        metrics = evaluate_filter(
            injector.graph,
            injector.get_ground_truth(),
            conf_threshold=0.2,
        )

        print("\nNoise benchmark")
        print(
            f"  injected={len(injector.injected_edges)} "
            f"precision={metrics['precision']:.3f} "
            f"recall={metrics['recall']:.3f} "
            f"f1={metrics['f1']:.3f}"
        )
        print(f"\nDemo data stored in: {Path(temp_dir)}")


if __name__ == "__main__":
    asyncio.run(main())
