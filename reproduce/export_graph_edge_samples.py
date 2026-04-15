from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import networkx as nx


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


def edge_record_from_graph(
    src: str,
    dst: str,
    data: dict[str, Any],
    relation_chunks: dict[str, Any],
    text_chunks: dict[str, Any],
    *,
    variant: str,
) -> dict[str, Any]:
    edge_key = "<SEP>".join(sorted([src, dst]))
    relation_chunk_entry = relation_chunks.get(edge_key, {})
    chunk_ids = normalize_chunk_ids(
        relation_chunk_entry.get("chunk_ids", data.get("source_id"))
    )
    source_chunks: list[str] = []
    for chunk_id in chunk_ids[:3]:
        chunk = text_chunks.get(chunk_id, {})
        source_chunks.append(str(chunk.get("content", "")))

    return {
        "variant": variant,
        "src": src,
        "dst": dst,
        "description": str(data.get("description", "")),
        "keywords": str(data.get("keywords", "")),
        "weight": float(data.get("weight", 0.0) or 0.0),
        "conf_score": float(data.get("conf_score", 0.0) or 0.0),
        "conf_freq_score": float(data.get("conf_freq_score", 0.0) or 0.0),
        "conf_consistency_score": float(
            data.get("conf_consistency_score", 0.0) or 0.0
        ),
        "conf_semantic_score": float(data.get("conf_semantic_score", 0.0) or 0.0),
        "conf_support": int(float(data.get("conf_support", 0) or 0)),
        "chunk_ids": chunk_ids,
        "source_chunks": source_chunks,
        "manual_label": "",
        "manual_notes": "",
    }


def sample_edges(
    graph: nx.Graph,
    relation_chunks: dict[str, Any],
    text_chunks: dict[str, Any],
    *,
    variant: str,
    sample_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    randomizer = random.Random(seed)
    edges = list(graph.edges(data=True))
    if not edges:
        return []

    chosen = (
        edges
        if sample_size <= 0 or sample_size >= len(edges)
        else randomizer.sample(edges, sample_size)
    )
    rows = [
        edge_record_from_graph(
            src,
            dst,
            dict(data),
            relation_chunks,
            text_chunks,
            variant=variant,
        )
        for src, dst, data in chosen
    ]
    rows.sort(key=lambda row: (row["src"], row["dst"]))
    return rows


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    serialized_rows: list[dict[str, Any]] = []
    for row in rows:
        serialized_rows.append(
            {
                **row,
                "chunk_ids": json.dumps(row["chunk_ids"], ensure_ascii=False),
                "source_chunks": json.dumps(row["source_chunks"], ensure_ascii=False),
            }
        )

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(serialized_rows[0].keys()))
        writer.writeheader()
        writer.writerows(serialized_rows)


def save_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Graph Edge Samples",
        "",
        "| Variant | Src | Dst | Conf Score | Description |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['src']} | {row['dst']} | {row['conf_score']:.4f} | {row['description'].replace('|', '/')} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export sampled graph edges with source chunks for manual precision annotation.",
    )
    parser.add_argument("--graph-file", required=True)
    parser.add_argument("--relation-chunks-file", required=True)
    parser.add_argument("--text-chunks-file", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = nx.read_graphml(args.graph_file)
    relation_chunks = load_json_dict(args.relation_chunks_file)
    text_chunks = load_json_dict(args.text_chunks_file)
    rows = sample_edges(
        graph,
        relation_chunks,
        text_chunks,
        variant=args.variant,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    output_prefix = Path(args.output_prefix)
    save_json(output_prefix.with_suffix(".json"), rows)
    save_csv(output_prefix.with_suffix(".csv"), rows)
    save_markdown(output_prefix.with_suffix(".md"), rows)

    print("Graph edge sampling completed.")
    print(f"  json: {output_prefix.with_suffix('.json')}")
    print(f"  csv: {output_prefix.with_suffix('.csv')}")
    print(f"  markdown: {output_prefix.with_suffix('.md')}")
    print(f"  sampled_edges: {len(rows)}")


if __name__ == "__main__":
    main()
