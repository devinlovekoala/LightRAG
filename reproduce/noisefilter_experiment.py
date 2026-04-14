from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightrag.noisefilter import NoiseAwareRetriever, NoiseInjector, evaluate_filter
from lightrag.noisefilter.experiment import (
    parse_thresholds,
    summarize_experiment_rows,
    write_experiment_bundle,
)


def build_clean_graph(num_nodes: int, clean_degree: int, seed: int) -> nx.MultiDiGraph:
    rng = random.Random(seed)
    graph = nx.MultiDiGraph()
    nodes = [f"Node_{index:02d}" for index in range(num_nodes)]
    for node in nodes:
        graph.add_node(node)

    for index, src in enumerate(nodes):
        for offset in range(1, clean_degree + 1):
            dst = nodes[(index + offset) % num_nodes]
            edge_key = f"clean-{index}-{offset}"
            graph.add_edge(
                src,
                dst,
                key=edge_key,
                relation="related_to",
                conf_score=0.9,
                weight=1.0,
                retrieval_rank=rng.random(),
                is_noise=False,
            )
    return graph


def ensure_retrieval_rank(graph: nx.MultiDiGraph, seed: int) -> None:
    rng = random.Random(seed)
    for _, _, _, data in graph.edges(keys=True, data=True):
        data.setdefault("retrieval_rank", rng.random())
        data.setdefault("weight", 1.0)
        data.setdefault("is_noise", False)


def collect_candidate_edges(graph: nx.MultiDiGraph, node: str) -> list[dict]:
    candidates = []
    for src, dst, key, data in graph.edges(node, keys=True, data=True):
        other = dst if src == node else src
        candidates.append(
            {
                "src_id": src,
                "tgt_id": other if src == node else src,
                "neighbor": other,
                "edge_key": key,
                "rank": float(data.get("retrieval_rank", 0.0)),
                "weight": float(data.get("weight", 1.0)),
                "conf_score": float(data.get("conf_score", 0.5)),
                "is_noise": bool(data.get("is_noise", False)),
            }
        )
    return candidates


def measure_retrieval_noise_rate(
    graph: nx.MultiDiGraph,
    *,
    strategy: str,
    top_k: int,
    threshold: float | None,
) -> dict[str, float]:
    if strategy == "baseline":
        retriever = NoiseAwareRetriever(enabled=False)
    elif strategy == "soft":
        retriever = NoiseAwareRetriever(enabled=True, soft_mode=True)
    elif strategy == "hard":
        retriever = NoiseAwareRetriever(
            enabled=True,
            soft_mode=False,
            conf_threshold=threshold if threshold is not None else 0.3,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    total_noise = 0
    total_edges = 0
    returned_counts = []

    for node in graph.nodes():
        candidates = collect_candidate_edges(graph, node)
        ranked = retriever.rank_local_edges(candidates)[:top_k]
        if not ranked:
            returned_counts.append(0)
            continue

        noise_count = sum(1 for edge in ranked if edge.get("is_noise"))
        total_noise += noise_count
        total_edges += len(ranked)
        returned_counts.append(len(ranked))

    noise_rate = total_noise / total_edges if total_edges else 0.0
    avg_returned = sum(returned_counts) / len(returned_counts) if returned_counts else 0.0

    return {
        "retrieval_noise_rate_at_k": noise_rate,
        "avg_returned_edges": avg_returned,
    }


def run_experiment(
    *,
    noise_ratios: list[float],
    thresholds: list[float],
    num_nodes: int,
    clean_degree: int,
    top_k: int,
    seed: int,
    injectors: list[str],
) -> list[dict[str, float | str | None]]:
    rows: list[dict[str, float | str | None]] = []

    for injector_name in injectors:
        for noise_ratio in noise_ratios:
            clean_graph = build_clean_graph(num_nodes, clean_degree, seed)
            injector = NoiseInjector(clean_graph, noise_ratio=noise_ratio, seed=seed)

            if injector_name == "random":
                injector.inject_random_relations()
            elif injector_name == "contradictory":
                injector.inject_contradictory_relations()
            else:
                raise ValueError(f"Unknown injector: {injector_name}")

            graph = injector.graph
            ensure_retrieval_rank(graph, seed)
            truth = injector.get_ground_truth()

            for strategy in ("baseline", "soft"):
                row = {
                    "injector": injector_name,
                    "noise_ratio": noise_ratio,
                    "strategy": strategy,
                    "threshold": None,
                    "filter_precision": 0.0,
                    "filter_recall": 0.0,
                    "filter_f1": 0.0,
                }
                row.update(
                    measure_retrieval_noise_rate(
                        graph,
                        strategy=strategy,
                        top_k=top_k,
                        threshold=None,
                    )
                )
                rows.append(row)

            for threshold in thresholds:
                row = {
                    "injector": injector_name,
                    "noise_ratio": noise_ratio,
                    "strategy": "hard",
                    "threshold": threshold,
                }
                row.update(evaluate_filter(graph, truth, conf_threshold=threshold))
                row["filter_precision"] = row.pop("precision")
                row["filter_recall"] = row.pop("recall")
                row["filter_f1"] = row.pop("f1")
                row.pop("filtered_count", None)
                row.update(
                    measure_retrieval_noise_rate(
                        graph,
                        strategy="hard",
                        top_k=top_k,
                        threshold=threshold,
                    )
                )
                rows.append(row)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a synthetic NoiseFilter-RAG benchmark and export JSON/CSV/Markdown results.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reproduce/results/noisefilter",
        help="Directory to store experiment outputs.",
    )
    parser.add_argument(
        "--noise-ratios",
        type=str,
        default="0.1,0.2,0.3",
        help="Comma-separated list of noise ratios.",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated list of hard-filter thresholds.",
    )
    parser.add_argument("--num-nodes", type=int, default=24)
    parser.add_argument("--clean-degree", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--injectors",
        type=str,
        default="random,contradictory",
        help="Comma-separated list of injectors: random, contradictory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    noise_ratios = parse_thresholds(args.noise_ratios)
    thresholds = parse_thresholds(args.thresholds)
    injectors = [item.strip() for item in args.injectors.split(",") if item.strip()]

    rows = run_experiment(
        noise_ratios=noise_ratios,
        thresholds=thresholds,
        num_nodes=args.num_nodes,
        clean_degree=args.clean_degree,
        top_k=args.top_k,
        seed=args.seed,
        injectors=injectors,
    )
    summary = summarize_experiment_rows(rows)
    config = {
        "noise_ratios": noise_ratios,
        "thresholds": thresholds,
        "num_nodes": args.num_nodes,
        "clean_degree": args.clean_degree,
        "top_k": args.top_k,
        "seed": args.seed,
        "injectors": injectors,
    }
    outputs = write_experiment_bundle(args.output_dir, rows, summary, config)

    print("NoiseFilter experiment completed.")
    for label, path in outputs.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
