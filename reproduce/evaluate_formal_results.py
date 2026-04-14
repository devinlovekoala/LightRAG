from __future__ import annotations

import argparse
import csv
import json
import re
import string
from pathlib import Path
from typing import Any


def normalize_answer(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    lowered = "".join(ch for ch in lowered if ch not in set(string.punctuation))
    lowered = " ".join(lowered.split())
    return lowered


def exact_match_score(prediction: str, ground_truth: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def token_f1_score(prediction: str, ground_truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    overlap: dict[str, int] = {}
    for token in gold_tokens:
        overlap[token] = overlap.get(token, 0) + 1

    matches = 0
    for token in pred_tokens:
        count = overlap.get(token, 0)
        if count > 0:
            matches += 1
            overlap[token] = count - 1

    if matches == 0:
        return 0.0

    precision = matches / len(pred_tokens)
    recall = matches / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_prediction(prediction: str, answers: list[str]) -> dict[str, float]:
    exact_match = max(exact_match_score(prediction, answer) for answer in answers)
    token_f1 = max(token_f1_score(prediction, answer) for answer in answers)
    return {
        "exact_match": exact_match,
        "token_f1": token_f1,
    }


def load_json_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}")
    return payload


def evaluate_rows(
    result_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    row_map: dict[int, dict[str, Any]] = {}
    error_map: dict[int, dict[str, Any]] = {}

    for row in result_rows:
        query_id = row.get("query_id")
        if isinstance(query_id, int):
            row_map[query_id] = row

    for row in error_rows:
        query_id = row.get("query_id")
        if isinstance(query_id, int):
            error_map[query_id] = row

    per_query: list[dict[str, Any]] = []
    for query_id in sorted(row_map):
        row = row_map[query_id]
        answers = row.get("ground_truth_answers", [])
        if not isinstance(answers, list) or not answers:
            continue

        prediction = str(row.get("result", ""))
        metrics = score_prediction(prediction, answers)
        per_query.append(
            {
                "query_id": query_id,
                "query": row.get("query", ""),
                "exact_match": metrics["exact_match"],
                "token_f1": metrics["token_f1"],
                "answers": answers,
                "result": prediction,
            }
        )

    total_queries = len(per_query) + len(error_map)
    avg_exact_match = (
        sum(row["exact_match"] for row in per_query) / len(per_query) if per_query else 0.0
    )
    avg_token_f1 = (
        sum(row["token_f1"] for row in per_query) / len(per_query) if per_query else 0.0
    )

    summary = {
        "total_queries": total_queries,
        "successful_queries": len(per_query),
        "error_queries": len(error_map),
        "avg_exact_match": avg_exact_match,
        "avg_token_f1": avg_token_f1,
    }
    return per_query, summary


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Formal Evaluation Report",
        "",
        "## Summary",
        "",
        f"- total_queries: {summary['total_queries']}",
        f"- successful_queries: {summary['successful_queries']}",
        f"- error_queries: {summary['error_queries']}",
        f"- avg_exact_match: {summary['avg_exact_match']:.4f}",
        f"- avg_token_f1: {summary['avg_token_f1']:.4f}",
        "",
        "## Per Query",
        "",
        "| Query ID | Exact Match | Token F1 | Query |",
        "| --- | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['query_id']} | {row['exact_match']:.4f} | {row['token_f1']:.4f} | {row['query']} |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate formal LightRAG query results against dataset-provided answers.",
    )
    parser.add_argument("--result-file", required=True)
    parser.add_argument("--error-file", required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_rows = load_json_rows(args.result_file)
    error_rows = load_json_rows(args.error_file)
    per_query, summary = evaluate_rows(result_rows, error_rows)

    output_prefix = Path(args.output_prefix)
    save_json(output_prefix.with_suffix(".json"), {"summary": summary, "rows": per_query})
    save_csv(output_prefix.with_suffix(".csv"), per_query)
    output_prefix.with_suffix(".md").write_text(
        build_markdown(summary, per_query),
        encoding="utf-8",
    )

    print("Formal evaluation completed.")
    print(f"  json: {output_prefix.with_suffix('.json')}")
    print(f"  csv: {output_prefix.with_suffix('.csv')}")
    print(f"  markdown: {output_prefix.with_suffix('.md')}")
    print(f"  avg_exact_match: {summary['avg_exact_match']:.4f}")
    print(f"  avg_token_f1: {summary['avg_token_f1']:.4f}")


if __name__ == "__main__":
    main()
