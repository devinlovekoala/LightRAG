from __future__ import annotations

import random
from typing import Any

import networkx as nx


def _coerce_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


class NoiseInjector:
    def __init__(
        self,
        clean_graph: nx.Graph,
        noise_ratio: float = 0.2,
        seed: int | None = None,
    ) -> None:
        self.graph = clean_graph.copy()
        self.noise_ratio = max(0.0, noise_ratio)
        self.random = random.Random(seed)
        self.injected_edges: list[tuple[Any, ...]] = []

    def inject_random_relations(self) -> "NoiseInjector":
        nodes = list(self.graph.nodes())
        if len(nodes) < 2:
            return self

        if self.noise_ratio <= 0:
            return self

        target_count = int(max(1, len(self.graph.edges()) * self.noise_ratio))
        fake_relations = ["unrelated_to", "contradicts", "fake_rel"]
        attempts = 0

        while len(self.injected_edges) < target_count and attempts < target_count * 20:
            attempts += 1
            src = self.random.choice(nodes)
            dst = self.random.choice(nodes)
            if src == dst:
                continue

            relation = self.random.choice(fake_relations)
            if self.graph.is_multigraph():
                edge_key = f"noise-random-{len(self.injected_edges)}"
                self.graph.add_edge(
                    src,
                    dst,
                    key=edge_key,
                    relation=relation,
                    conf_score=0.0,
                    is_noise=True,
                )
                self.injected_edges.append((src, dst, edge_key))
                continue

            if self.graph.has_edge(src, dst):
                continue

            self.graph.add_edge(
                src,
                dst,
                relation=relation,
                conf_score=0.0,
                is_noise=True,
            )
            self.injected_edges.append((src, dst))

        return self

    def inject_contradictory_relations(self) -> "NoiseInjector":
        if self.graph.is_multigraph():
            existing = list(self.graph.edges(keys=True, data=True))
        else:
            existing = list(self.graph.edges(data=True))

        if not existing:
            return self

        if self.noise_ratio <= 0:
            return self

        target_count = int(max(1, len(existing) * self.noise_ratio))
        samples = self.random.sample(existing, min(target_count, len(existing)))

        for edge in samples:
            if self.graph.is_multigraph():
                src, dst, _, data = edge
                edge_key = f"noise-contradict-{len(self.injected_edges)}"
                relation = f"NOT_{data.get('relation', 'related_to')}"
                self.graph.add_edge(
                    src,
                    dst,
                    key=edge_key,
                    relation=relation,
                    conf_score=0.0,
                    is_noise=True,
                )
                self.injected_edges.append((src, dst, edge_key))
                continue

            src, dst, data = edge
            relation = f"NOT_{data.get('relation', 'related_to')}"
            if self.graph.is_directed() and not self.graph.has_edge(dst, src):
                self.graph.add_edge(
                    dst,
                    src,
                    relation=relation,
                    conf_score=0.0,
                    is_noise=True,
                )
                self.injected_edges.append((dst, src))
            else:
                self.graph.add_edge(
                    src,
                    dst,
                    relation=relation,
                    conf_score=0.0,
                    is_noise=True,
                )
                self.injected_edges.append((src, dst))

        return self

    def get_ground_truth(self) -> set[tuple[Any, ...]]:
        return set(tuple(edge) for edge in self.injected_edges)


def evaluate_filter(
    graph: nx.Graph,
    injected_edges: set[tuple[Any, ...]] | list[tuple[Any, ...]],
    conf_threshold: float = 0.3,
) -> dict[str, float]:
    threshold = max(0.0, min(1.0, conf_threshold))
    truth = {tuple(edge) for edge in injected_edges}
    filtered: set[tuple[Any, ...]] = set()

    if graph.is_multigraph():
        for src, dst, key, data in graph.edges(keys=True, data=True):
            score = _coerce_float(data.get("conf_score"), 0.5)
            if score < threshold:
                filtered.add((src, dst, key))
    else:
        for src, dst, data in graph.edges(data=True):
            score = _coerce_float(data.get("conf_score"), 0.5)
            if score < threshold:
                filtered.add((src, dst))

    true_positives = len(filtered & truth)
    precision = true_positives / len(filtered) if filtered else 0.0
    recall = true_positives / len(truth) if truth else 0.0
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "filtered_count": float(len(filtered)),
    }
