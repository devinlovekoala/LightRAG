from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lightrag.noisefilter.reproduction import (
    build_formal_run_paths,
    create_formal_rag,
    finalize_rag,
    insert_contexts,
    resolve_variant_settings,
)


def parse_args(*, fixed_variant: str | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert a formal reproduction dataset into a baseline or NoiseFilter LightRAG workspace.",
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
        "--datasets-root",
        default="datasets",
        help="Root directory containing unique_contexts/ and questions/.",
    )
    parser.add_argument(
        "--working-root",
        default="rag_storage/formal_runs",
        help="Root directory for indexed LightRAG workspaces.",
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=10.0)
    parser.add_argument("--conf-threshold", type=float, default=0.3)
    parser.add_argument(
        "--hard-filter",
        action="store_true",
        help="Use hard filtering for the NoiseFilter variant.",
    )
    parser.add_argument("--w-freq", type=float, default=0.5)
    parser.add_argument("--w-cons", type=float, default=0.3)
    parser.add_argument("--w-sem", type=float, default=0.2)
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
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

    rag = None
    try:
        rag = await create_formal_rag(
            working_dir=paths.working_dir,
            variant_settings=settings,
        )
        inserted = await insert_contexts(
            rag,
            paths.context_file,
            retries=args.retries,
            retry_delay_seconds=args.retry_delay,
        )
        print("Formal insert completed.")
        print(f"  dataset: {paths.dataset}")
        print(f"  variant: {paths.variant}")
        print(f"  working_dir: {paths.working_dir}")
        print(f"  inserted_contexts: {inserted}")
    finally:
        await finalize_rag(rag)


def main(*, fixed_variant: str | None = None) -> None:
    asyncio.run(_run(parse_args(fixed_variant=fixed_variant)))


def run_with_fixed_variant(variant: str) -> None:
    main(fixed_variant=variant)


if __name__ == "__main__":
    main()
