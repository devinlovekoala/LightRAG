from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightrag.noisefilter.reproduction import (
    build_runtime_overrides,
    build_formal_run_paths,
    create_formal_rag,
    finalize_rag,
    load_qa_records,
    resolve_variant_settings,
    run_queries,
    save_json_records,
)


def parse_args(*, fixed_variant: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run formal reproduction queries against a baseline or NoiseFilter LightRAG workspace.",
    )
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. agriculture")
    if fixed_variant is None:
        parser.add_argument(
            "--variant",
            required=True,
            choices=["baseline", "noisefilter"],
            help="Runner variant.",
        )
    else:
        parser.set_defaults(variant=fixed_variant)
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["local", "global", "hybrid", "mix", "naive"],
        help="LightRAG query mode.",
    )
    parser.add_argument(
        "--datasets-root",
        default="datasets",
        help="Root directory containing unique_contexts/ and questions/.",
    )
    parser.add_argument(
        "--working-root",
        default="rag_storage/formal_runs",
        help="Root directory for indexed LightRAG workspaces.",
    )
    parser.add_argument(
        "--results-root",
        default="reproduce/results/formal",
        help="Root directory for query outputs.",
    )
    parser.add_argument(
        "--disable-qa",
        action="store_true",
        help="Ignore datasets/qa/{dataset}_qa.json even when it exists.",
    )
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument(
        "--hard-filter",
        action="store_true",
        help="Use hard filtering for the NoiseFilter variant.",
    )
    parser.add_argument("--w-freq", type=float, default=0.5)
    parser.add_argument("--w-cons", type=float, default=0.3)
    parser.add_argument("--w-sem", type=float, default=0.2)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override LightRAG chunk token size for this run only.",
    )
    parser.add_argument(
        "--chunk-overlap-size",
        type=int,
        default=None,
        help="Override LightRAG chunk overlap token size for this run only.",
    )
    parser.add_argument(
        "--max-async",
        type=int,
        default=None,
        help="Override LightRAG LLM concurrency for this run only.",
    )
    parser.add_argument(
        "--embedding-max-async",
        type=int,
        default=None,
        help="Override embedding concurrency for this run only.",
    )
    parser.add_argument(
        "--max-parallel-insert",
        type=int,
        default=None,
        help="Override document-level insert concurrency for this run only.",
    )
    parser.add_argument(
        "--max-gleaning",
        type=int,
        default=None,
        help="Override extraction gleaning attempts for this run only.",
    )
    parser.add_argument(
        "--max-extract-input-tokens",
        type=int,
        default=None,
        help="Override the maximum extraction input tokens for this run only.",
    )
    parser.add_argument(
        "--llm-timeout",
        type=int,
        default=None,
        help="Override LightRAG LLM timeout in seconds for this run only.",
    )
    parser.add_argument(
        "--embedding-timeout",
        type=int,
        default=None,
        help="Override LightRAG embedding timeout in seconds for this run only.",
    )
    parser.add_argument(
        "--query-concurrency",
        type=int,
        default=4,
        help="Number of concurrent formal queries to execute.",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    paths = build_formal_run_paths(
        dataset=args.dataset,
        variant=args.variant,
        query_mode=args.mode,
        datasets_root=args.datasets_root,
        working_root=args.working_root,
        results_root=args.results_root,
    )
    settings = resolve_variant_settings(
        args.variant,
        conf_threshold=args.conf_threshold,
        soft_mode=not args.hard_filter,
        w_freq=args.w_freq,
        w_cons=args.w_cons,
        w_sem=args.w_sem,
    )
    runtime_overrides = build_runtime_overrides(
        chunk_size=args.chunk_size,
        chunk_overlap_size=args.chunk_overlap_size,
        llm_max_async=args.max_async,
        embedding_max_async=args.embedding_max_async,
        max_parallel_insert=args.max_parallel_insert,
        max_gleaning=args.max_gleaning,
        max_extract_input_tokens=args.max_extract_input_tokens,
        llm_timeout=args.llm_timeout,
        embedding_timeout=args.embedding_timeout,
    )

    rag = None
    try:
        qa_records = None
        if not args.disable_qa and paths.qa_file.exists():
            qa_records = load_qa_records(paths.qa_file)

        rag = await create_formal_rag(
            working_dir=paths.working_dir,
            variant_settings=settings,
            runtime_overrides=runtime_overrides,
        )
        results, errors = await run_queries(
            rag,
            questions_file=paths.questions_file,
            query_mode=paths.query_mode,
            qa_records=qa_records,
            query_concurrency=args.query_concurrency,
        )
        save_json_records(paths.result_file, results)
        save_json_records(paths.error_file, errors)

        print("Formal query run completed.")
        print(f"  dataset: {paths.dataset}")
        print(f"  variant: {paths.variant}")
        print(f"  mode: {paths.query_mode}")
        print(f"  result_file: {paths.result_file}")
        print(f"  error_file: {paths.error_file}")
        print(f"  query_count: {len(results) + len(errors)}")
        print(f"  success_count: {len(results)}")
        print(f"  error_count: {len(errors)}")
        print(
            f"  qa_records: {len(qa_records) if qa_records is not None else 'disabled'}"
        )
        print(f"  query_concurrency: {args.query_concurrency}")
        if runtime_overrides:
            print(f"  runtime_overrides: {runtime_overrides}")
    finally:
        await finalize_rag(rag)


def main(*, fixed_variant: str | None = None) -> None:
    asyncio.run(_run(parse_args(fixed_variant=fixed_variant)))


def run_with_fixed_variant(variant: str) -> None:
    main(fixed_variant=variant)


if __name__ == "__main__":
    main()
