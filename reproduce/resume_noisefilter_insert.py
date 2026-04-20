from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce.run_formal_insert import parse_args
from lightrag.noisefilter.reproduction import (
    build_formal_run_paths,
    build_runtime_overrides,
    create_formal_rag,
    finalize_rag,
    resolve_variant_settings,
    resume_contexts,
)


async def _run() -> None:
    args = parse_args(fixed_variant="noisefilter")
    paths = build_formal_run_paths(
        dataset=args.dataset,
        variant=args.variant,
        query_mode="hybrid",
        datasets_root=args.datasets_root,
        working_root=args.working_root,
    )
    settings = resolve_variant_settings(
        args.variant,
        conf_threshold=args.conf_threshold,
        soft_mode=not args.hard_filter,
        w_freq=args.w_freq,
        w_cons=args.w_cons,
        w_sem=args.w_sem,
    )
    addon_params = {}
    if args.enable_chunk_binding_gate:
        addon_params["enable_relation_chunk_entity_gate"] = True

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
        addon_params=addon_params or None,
    )

    rag = None
    try:
        rag = await create_formal_rag(
            working_dir=paths.working_dir,
            variant_settings=settings,
            runtime_overrides=runtime_overrides,
        )
        summary = await resume_contexts(rag, paths.context_file)
        print("Formal NoiseFilter resume completed.")
        print(f"  dataset: {paths.dataset}")
        print(f"  variant: {paths.variant}")
        print(f"  working_dir: {paths.working_dir}")
        print(f"  dataset_contexts: {summary['dataset_contexts']}")
        print(f"  missing_enqueued: {summary['missing_enqueued']}")
        print(f"  retryable_before: {summary['retryable_before']}")
        print(f"  status_counts_before: {summary['status_counts_before']}")
        print(f"  status_counts_after: {summary['status_counts_after']}")
        if runtime_overrides:
            print(f"  runtime_overrides: {runtime_overrides}")
    finally:
        await finalize_rag(rag)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
