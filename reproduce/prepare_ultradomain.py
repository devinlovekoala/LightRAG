from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


DEFAULT_REPO_ID = "TommyChien/UltraDomain"


def normalize_domains(raw_domains: str) -> list[str]:
    domains = [domain.strip().lower() for domain in raw_domains.split(",")]
    return [domain for domain in domains if domain]


def load_jsonl_rows(path: str | Path, max_records: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for index, line in enumerate(file):
            if max_records is not None and index >= max_records:
                break
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object rows in {path}")
            rows.append(row)
    return rows


def build_questions_text(qa_records: list[dict[str, Any]]) -> str:
    lines = [
        f"- Question {record['query_id']}: {record['question']}" for record in qa_records
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def extract_ultradomain_artifacts(
    rows: list[dict[str, Any]],
    *,
    source_name: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    unique_contexts: list[str] = []
    seen_contexts: set[str] = set()
    qa_records: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows, start=1):
        question = str(row.get("input", "")).strip()
        context = str(row.get("context", "")).strip()
        answers = row.get("answers", [])

        if not question or not context:
            continue
        if not isinstance(answers, list):
            continue

        clean_answers = [
            str(answer).strip() for answer in answers if str(answer).strip()
        ]
        if not clean_answers:
            continue

        if context not in seen_contexts:
            seen_contexts.add(context)
            unique_contexts.append(context)

        qa_records.append(
            {
                "query_id": len(qa_records) + 1,
                "question": question,
                "answers": clean_answers,
                "metadata": {
                    "source_file": source_name,
                    "row_index": row_index,
                },
            }
        )

    return unique_contexts, qa_records


def save_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def save_text(path: str | Path, content: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def download_domain_file(
    *,
    repo_id: str,
    domain: str,
    raw_dir: Path,
    force_download: bool = False,
) -> Path:
    from huggingface_hub import hf_hub_download

    filename = f"{domain}.jsonl"
    downloaded_path = Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            force_download=force_download,
        )
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_path = raw_dir / filename
    shutil.copy2(downloaded_path, target_path)
    return target_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and prepare UltraDomain files for formal LightRAG reproduction.",
    )
    parser.add_argument(
        "--domains",
        default="mix",
        help="Comma-separated domain names, e.g. mix,agriculture,cs,legal",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--datasets-root", default="datasets")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional cap per domain for smoke tests.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload files even when Hugging Face cache already has them.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets_root = Path(args.datasets_root)
    raw_dir = datasets_root / "raw"
    unique_contexts_dir = datasets_root / "unique_contexts"
    questions_dir = datasets_root / "questions"
    qa_dir = datasets_root / "qa"

    for domain in normalize_domains(args.domains):
        raw_file = download_domain_file(
            repo_id=args.repo_id,
            domain=domain,
            raw_dir=raw_dir,
            force_download=args.force_download,
        )
        rows = load_jsonl_rows(raw_file, max_records=args.max_records)
        unique_contexts, qa_records = extract_ultradomain_artifacts(
            rows,
            source_name=raw_file.name,
        )

        save_json(unique_contexts_dir / f"{domain}_unique_contexts.json", unique_contexts)
        save_json(qa_dir / f"{domain}_qa.json", qa_records)
        save_text(questions_dir / f"{domain}_questions.txt", build_questions_text(qa_records))

        print("Prepared UltraDomain domain.")
        print(f"  domain: {domain}")
        print(f"  raw_file: {raw_file}")
        print(f"  rows: {len(rows)}")
        print(f"  unique_contexts: {len(unique_contexts)}")
        print(f"  questions: {len(qa_records)}")


if __name__ == "__main__":
    main()
