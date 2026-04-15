from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)


CRITERIA = [
    "Comprehensiveness",
    "Diversity",
    "Empowerment",
    "Overall Winner",
]

SYSTEM_PROMPT = """---Role---
You are an expert tasked with evaluating two answers to the same question based on three criteria: Comprehensiveness, Diversity, and Empowerment.

---Instructions---
1. Judge only based on the question and the two answers.
2. Do not reward verbosity by default. Prefer the answer that is more useful, accurate, and directly responsive.
3. If both answers are equally strong or equally weak on a criterion, return "Tie".
4. Respond with valid JSON only.
"""


@dataclass(slots=True)
class PairwiseExample:
    query_id: int
    query: str
    answer_a: str | None
    answer_b: str | None
    label_a: str
    label_b: str


def load_result_rows(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON array in {path}")
    return payload


def build_examples(
    baseline_rows: list[dict[str, Any]],
    noisefilter_rows: list[dict[str, Any]],
) -> list[PairwiseExample]:
    baseline_map = {
        int(row["query_id"]): row
        for row in baseline_rows
        if "query_id" in row and isinstance(row["query_id"], int)
    }
    noisefilter_map = {
        int(row["query_id"]): row
        for row in noisefilter_rows
        if "query_id" in row and isinstance(row["query_id"], int)
    }

    query_ids = sorted(set(baseline_map) & set(noisefilter_map))
    examples: list[PairwiseExample] = []
    for query_id in query_ids:
        baseline_row = baseline_map[query_id]
        noisefilter_row = noisefilter_map[query_id]
        query = str(baseline_row.get("query") or noisefilter_row.get("query") or "")

        # Alternate order to reduce position bias without adding another full pass.
        if query_id % 2 == 0:
            answer_a = noisefilter_row.get("result")
            answer_b = baseline_row.get("result")
            label_a = "noisefilter"
            label_b = "baseline"
        else:
            answer_a = baseline_row.get("result")
            answer_b = noisefilter_row.get("result")
            label_a = "baseline"
            label_b = "noisefilter"

        examples.append(
            PairwiseExample(
                query_id=query_id,
                query=query,
                answer_a=None if answer_a is None else str(answer_a),
                answer_b=None if answer_b is None else str(answer_b),
                label_a=label_a,
                label_b=label_b,
            )
        )
    return examples


def build_user_prompt(example: PairwiseExample) -> str:
    answer_a = example.answer_a if example.answer_a is not None else "None"
    answer_b = example.answer_b if example.answer_b is not None else "None"
    return f"""You will evaluate two answers to the same question based on three criteria: Comprehensiveness, Diversity, and Empowerment.

- Comprehensiveness: How much detail does the answer provide to cover all aspects and details of the question?
- Diversity: How varied and rich is the answer in providing different perspectives and insights on the question?
- Empowerment: How well does the answer help the reader understand and make informed judgments about the topic?

For each criterion, choose the better answer (Answer A, Answer B, or Tie) and explain why. Then, select an overall winner based on these three categories.

Question:
{example.query}

Answer A:
{answer_a}

Answer B:
{answer_b}

Return valid JSON in the following format:
{{
  "Comprehensiveness": {{
    "Winner": "Answer A | Answer B | Tie",
    "Explanation": "..."
  }},
  "Diversity": {{
    "Winner": "Answer A | Answer B | Tie",
    "Explanation": "..."
  }},
  "Empowerment": {{
    "Winner": "Answer A | Answer B | Tie",
    "Explanation": "..."
  }},
  "Overall Winner": {{
    "Winner": "Answer A | Answer B | Tie",
    "Explanation": "..."
  }}
}}"""


def _normalize_winner(raw: str | None) -> str:
    if raw is None:
        return "tie"
    lowered = raw.strip().lower()
    if lowered in {"answer a", "a", "1", "answer 1"}:
        return "answer_a"
    if lowered in {"answer b", "b", "2", "answer 2"}:
        return "answer_b"
    if lowered in {"tie", "draw", "equal", "both"}:
        return "tie"
    return "tie"


def map_winner_to_variant(
    normalized_winner: str,
    example: PairwiseExample,
) -> str:
    if normalized_winner == "answer_a":
        return example.label_a
    if normalized_winner == "answer_b":
        return example.label_b
    return "tie"


def summarize_judgments(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total_queries": len(rows),
        "criteria": {},
    }
    for criterion in CRITERIA:
        counts = {"baseline": 0, "noisefilter": 0, "tie": 0}
        for row in rows:
            winner = row["judgment"][criterion]["mapped_winner"]
            counts[winner] = counts.get(winner, 0) + 1
        total = len(rows) or 1
        summary["criteria"][criterion] = {
            **counts,
            "baseline_win_rate": counts["baseline"] / total,
            "noisefilter_win_rate": counts["noisefilter"] / total,
            "tie_rate": counts["tie"] / total,
        }
    return summary


def build_markdown(summary: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Formal Pairwise Evaluation Report",
        "",
        f"- total_queries: {summary['total_queries']}",
        "",
        "## Win Rates",
        "",
        "| Criterion | Baseline | NoiseFilter | Tie |",
        "| --- | ---: | ---: | ---: |",
    ]
    for criterion in CRITERIA:
        item = summary["criteria"][criterion]
        lines.append(
            f"| {criterion} | {item['baseline_win_rate']:.4f} | {item['noisefilter_win_rate']:.4f} | {item['tie_rate']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Per Query",
            "",
            "| Query ID | Overall | Comprehensiveness | Diversity | Empowerment | Query |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in rows:
        judgment = row["judgment"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["query_id"]),
                    judgment["Overall Winner"]["mapped_winner"],
                    judgment["Comprehensiveness"]["mapped_winner"],
                    judgment["Diversity"]["mapped_winner"],
                    judgment["Empowerment"]["mapped_winner"],
                    str(row["query"]).replace("\n", " "),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    flattened: list[dict[str, Any]] = []
    for row in rows:
        flat = {
            "query_id": row["query_id"],
            "query": row["query"],
        }
        for criterion in CRITERIA:
            item = row["judgment"][criterion]
            prefix = criterion.lower().replace(" ", "_")
            flat[f"{prefix}_winner"] = item["mapped_winner"]
            flat[f"{prefix}_raw_winner"] = item["raw_winner"]
            flat[f"{prefix}_explanation"] = item["explanation"]
        flattened.append(flat)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(flattened[0].keys()))
        writer.writeheader()
        writer.writerows(flattened)


async def evaluate_examples(
    examples: list[PairwiseExample],
    *,
    model: str,
    base_url: str | None,
    api_key: str,
    concurrency: int,
) -> list[dict[str, Any]]:
    from openai import AsyncOpenAI

    semaphore = asyncio.Semaphore(concurrency)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def _judge(example: PairwiseExample) -> dict[str, Any]:
        async with semaphore:
            response = await client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(example)},
                ],
            )
            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)

            judgment: dict[str, Any] = {}
            for criterion in CRITERIA:
                item = parsed.get(criterion, {})
                raw_winner = str(item.get("Winner", "Tie"))
                normalized = _normalize_winner(raw_winner)
                mapped_winner = map_winner_to_variant(normalized, example)
                judgment[criterion] = {
                    "raw_winner": raw_winner,
                    "normalized_winner": normalized,
                    "mapped_winner": mapped_winner,
                    "explanation": str(item.get("Explanation", "")).strip(),
                }

            return {
                "query_id": example.query_id,
                "query": example.query,
                "answer_a_label": example.label_a,
                "answer_b_label": example.label_b,
                "judgment": judgment,
            }

    try:
        rows = await asyncio.gather(*[_judge(example) for example in examples])
        return sorted(rows, key=lambda row: row["query_id"])
    finally:
        await client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pairwise LLM-judge evaluation for baseline vs NoiseFilter formal results.",
    )
    parser.add_argument("--baseline-results", required=True)
    parser.add_argument("--noisefilter-results", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument(
        "--judge-model",
        default=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        help="Judge model name for the OpenAI-compatible API.",
    )
    parser.add_argument(
        "--judge-host",
        default=os.getenv("LLM_BINDING_HOST"),
        help="Judge API base URL.",
    )
    parser.add_argument(
        "--judge-api-key",
        default=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        help="Judge API key. Defaults to current .env compatible binding key.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of concurrent judge requests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.judge_api_key:
        raise EnvironmentError(
            "Missing judge API key. Set LLM_BINDING_API_KEY / OPENAI_API_KEY or pass --judge-api-key."
        )
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be > 0")

    baseline_rows = load_result_rows(args.baseline_results)
    noisefilter_rows = load_result_rows(args.noisefilter_results)
    examples = build_examples(baseline_rows, noisefilter_rows)
    rows = asyncio.run(
        evaluate_examples(
            examples,
            model=args.judge_model,
            base_url=args.judge_host,
            api_key=args.judge_api_key,
            concurrency=args.concurrency,
        )
    )
    summary = summarize_judgments(rows)

    output_prefix = Path(args.output_prefix)
    save_json(output_prefix.with_suffix(".json"), {"summary": summary, "rows": rows})
    save_csv(output_prefix.with_suffix(".csv"), rows)
    output_prefix.with_suffix(".md").write_text(
        build_markdown(summary, rows),
        encoding="utf-8",
    )

    print("Formal pairwise evaluation completed.")
    print(f"  json: {output_prefix.with_suffix('.json')}")
    print(f"  csv: {output_prefix.with_suffix('.csv')}")
    print(f"  markdown: {output_prefix.with_suffix('.md')}")
    print(
        "  overall_noisefilter_win_rate: "
        f"{summary['criteria']['Overall Winner']['noisefilter_win_rate']:.4f}"
    )
    print(
        "  overall_baseline_win_rate: "
        f"{summary['criteria']['Overall Winner']['baseline_win_rate']:.4f}"
    )


if __name__ == "__main__":
    main()
