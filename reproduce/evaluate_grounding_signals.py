from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import fmean
from typing import Any, Callable

import numpy as np


LABELS = ("correct", "wrong", "ambiguous")
SIGNAL_FIELDS = (
    "src_anchor_score",
    "dst_anchor_score",
    "endpoint_anchor_score",
    "cooccurrence_score",
    "nli_support_score",
)
DESCRIPTION_SEP = "<SEP>"


def _normalize_label(value: Any) -> str:
    return str(value or "").strip().lower()


def _parse_json_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(item) for item in raw if str(item).strip()]
    if not isinstance(raw, str):
        return []

    text = raw.strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    return [item for item in text.split("<SEP>") if item]


def load_edge_rows(path: str | Path) -> list[dict[str, Any]]:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows: list[dict[str, Any]] = []
        for row in reader:
            normalized = dict(row)
            normalized["variant"] = str(row.get("variant", "")).strip().lower()
            normalized["manual_label"] = _normalize_label(row.get("manual_label"))
            normalized["chunk_ids"] = _parse_json_list(row.get("chunk_ids"))
            normalized["source_chunks"] = _parse_json_list(row.get("source_chunks"))
            rows.append(normalized)
    return rows


def load_text_chunks(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")

    text_chunks: dict[str, str] = {}
    for chunk_id, data in payload.items():
        if not isinstance(chunk_id, str):
            continue
        if isinstance(data, dict):
            text_chunks[chunk_id] = str(data.get("content", ""))
        else:
            text_chunks[chunk_id] = str(data)
    return text_chunks


def resolve_source_texts(
    row: dict[str, Any], text_chunk_lookup: dict[str, str] | None = None
) -> list[str]:
    chunk_ids = row.get("chunk_ids", [])
    if text_chunk_lookup and isinstance(chunk_ids, list):
        resolved = [
            text_chunk_lookup[chunk_id]
            for chunk_id in chunk_ids
            if isinstance(chunk_id, str) and chunk_id in text_chunk_lookup
        ]
        if resolved:
            return resolved

    source_chunks = row.get("source_chunks", [])
    if isinstance(source_chunks, list):
        return [str(chunk) for chunk in source_chunks if str(chunk).strip()]
    return []


_NON_WORD_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(text: str) -> str:
    return " ".join(_NON_WORD_RE.sub(" ", text.lower()).split())


def _entity_tokens(entity: str) -> list[str]:
    return [token for token in normalize_text(entity).split() if len(token) >= 2]


def _entity_anchor_in_text(entity: str, chunk_text: str) -> float:
    normalized_entity = normalize_text(entity)
    normalized_chunk = normalize_text(chunk_text)
    if not normalized_entity or not normalized_chunk:
        return 0.0
    if normalized_entity in normalized_chunk:
        return 1.0

    chunk_tokens = set(normalized_chunk.split())
    entity_tokens = _entity_tokens(entity)
    if not entity_tokens:
        return 0.0

    overlap = sum(token in chunk_tokens for token in entity_tokens) / len(entity_tokens)
    if overlap >= 1.0:
        return 0.85
    if overlap >= 0.6:
        return 0.5
    return 0.0


def compute_anchor_scores(
    src: str, dst: str, source_texts: list[str]
) -> dict[str, float]:
    if not source_texts:
        return {
            "src_anchor_score": 0.0,
            "dst_anchor_score": 0.0,
            "endpoint_anchor_score": 0.0,
            "cooccurrence_score": 0.0,
        }

    src_hits = []
    dst_hits = []
    cooccurrence_hits = []
    for chunk_text in source_texts:
        src_hit = 1.0 if _entity_anchor_in_text(src, chunk_text) > 0 else 0.0
        dst_hit = 1.0 if _entity_anchor_in_text(dst, chunk_text) > 0 else 0.0
        src_hits.append(src_hit)
        dst_hits.append(dst_hit)
        cooccurrence_hits.append(1.0 if src_hit and dst_hit else 0.0)

    src_anchor = fmean(src_hits)
    dst_anchor = fmean(dst_hits)
    return {
        "src_anchor_score": src_anchor,
        "dst_anchor_score": dst_anchor,
        "endpoint_anchor_score": (src_anchor + dst_anchor) / 2.0,
        "cooccurrence_score": fmean(cooccurrence_hits),
    }


def _clean_hypothesis_text(text: str) -> str:
    cleaned = " ".join(str(text or "").strip().split())
    return cleaned.strip(" .") + "." if cleaned else ""


def build_hypotheses(row: dict[str, Any]) -> list[str]:
    src = str(row.get("src", "")).strip()
    dst = str(row.get("dst", "")).strip()
    keywords = str(row.get("keywords", "")).strip()
    description = str(row.get("description", "")).strip()

    hypotheses: list[str] = []
    seen: set[str] = set()

    def add(text: str) -> None:
        cleaned = _clean_hypothesis_text(text)
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            hypotheses.append(cleaned)

    if description:
        for part in description.split(DESCRIPTION_SEP):
            add(part)

    if keywords:
        for keyword in [item.strip() for item in keywords.split(",") if item.strip()]:
            add(f"{src} {keyword} {dst}")
            add(f"{src} is associated with {dst} via {keyword}")

    if src and dst:
        add(f"{src} is related to {dst}")

    return hypotheses


def score_nli_support(
    source_texts: list[str],
    hypotheses: list[str],
    nli_scorer: Callable[[str, str], float] | None,
) -> float:
    if not source_texts or nli_scorer is None or not hypotheses:
        return 0.0
    return max(
        float(nli_scorer(chunk_text, hypothesis))
        for chunk_text in source_texts
        for hypothesis in hypotheses
    )


def score_nli_support_many(
    rows: list[dict[str, Any]],
    nli_scorer: Any,
) -> list[float]:
    if not rows or nli_scorer is None:
        return [0.0 for _ in rows]

    if not hasattr(nli_scorer, "score_many"):
        return [
            score_nli_support(
                row.get("_source_texts", []),
                row.get("_hypotheses", []),
                nli_scorer,
            )
            for row in rows
        ]

    pair_offsets: list[tuple[int, int]] = []
    pairs: list[tuple[str, str]] = []
    for row in rows:
        row_pairs = [
            (chunk_text, hypothesis)
            for chunk_text in row.get("_source_texts", [])
            for hypothesis in row.get("_hypotheses", [])
        ]
        start = len(pairs)
        pairs.extend(row_pairs)
        pair_offsets.append((start, len(pairs)))

    if not pairs:
        return [0.0 for _ in rows]

    scores = list(nli_scorer.score_many(pairs))
    aggregated: list[float] = []
    for start, end in pair_offsets:
        aggregated.append(max(scores[start:end]) if end > start else 0.0)
    return aggregated


def _mean(values: list[float]) -> float:
    return fmean(values) if values else 0.0


def compute_roc_auc(labels: list[int], scores: list[float]) -> float | None:
    if not labels or len(labels) != len(scores):
        return None
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None

    paired = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    index = 0
    while index < len(paired):
        end = index + 1
        while end < len(paired) and paired[end][0] == paired[index][0]:
            end += 1
        avg_rank = (index + 1 + end) / 2.0
        positive_count = sum(label for _, label in paired[index:end])
        rank_sum += avg_rank * positive_count
        index = end

    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def compute_average_precision(labels: list[int], scores: list[float]) -> float | None:
    if not labels or len(labels) != len(scores):
        return None
    positives = sum(labels)
    if positives == 0:
        return None

    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    hit_count = 0
    precision_sum = 0.0
    for index, (_, label) in enumerate(ranked, start=1):
        if label != 1:
            continue
        hit_count += 1
        precision_sum += hit_count / index
    return precision_sum / positives


def evaluate_variant(
    variant: str,
    rows: list[dict[str, Any]],
    *,
    text_chunk_lookup: dict[str, str] | None = None,
    nli_scorer: Callable[[str, str], float] | None = None,
) -> dict[str, Any]:
    filtered_rows = [
        row for row in rows if _normalize_label(row.get("manual_label")) in LABELS
    ]
    prepared_rows: list[dict[str, Any]] = []

    for row in filtered_rows:
        source_texts = resolve_source_texts(row, text_chunk_lookup)
        anchor_scores = compute_anchor_scores(
            str(row.get("src", "")),
            str(row.get("dst", "")),
            source_texts,
        )
        hypotheses = build_hypotheses(row)
        prepared_row = {
            **row,
            **anchor_scores,
            "resolved_source_chunk_count": len(source_texts),
            "hypothesis": hypotheses[0] if hypotheses else "",
            "hypotheses": hypotheses,
            "_source_texts": source_texts,
            "_hypotheses": hypotheses,
        }
        prepared_rows.append(prepared_row)

    nli_scores = score_nli_support_many(prepared_rows, nli_scorer)
    evaluated_rows: list[dict[str, Any]] = []
    for row, nli_score in zip(prepared_rows, nli_scores):
        evaluated_rows.append(
            {
                key: value
                for key, value in {
                    **row,
                    "nli_support_score": nli_score,
                }.items()
                if key not in {"_source_texts", "_hypotheses"}
            }
        )

    label_counts = {
        label: sum(row["manual_label"] == label for row in evaluated_rows)
        for label in LABELS
    }
    total_edges = len(evaluated_rows)
    strict_precision = label_counts["correct"] / total_edges if total_edges else 0.0
    lenient_precision = (
        (label_counts["correct"] + label_counts["ambiguous"]) / total_edges
        if total_edges
        else 0.0
    )

    signal_means_by_label: dict[str, dict[str, float]] = {}
    for signal in SIGNAL_FIELDS:
        signal_means_by_label[signal] = {
            label: _mean(
                [float(row.get(signal, 0.0)) for row in evaluated_rows if row["manual_label"] == label]
            )
            for label in LABELS
        }

    binary_rows = [row for row in evaluated_rows if row["manual_label"] in {"correct", "wrong"}]
    labels = [1 if row["manual_label"] == "correct" else 0 for row in binary_rows]

    binary_ranking: dict[str, dict[str, float | None]] = {}
    for signal in SIGNAL_FIELDS:
        scores = [float(row.get(signal, 0.0)) for row in binary_rows]
        binary_ranking[signal] = {
            "roc_auc": compute_roc_auc(labels, scores),
            "average_precision": compute_average_precision(labels, scores),
        }

    return {
        "variant": variant,
        "total_edges": total_edges,
        "label_counts": label_counts,
        "strict_precision": strict_precision,
        "lenient_precision": lenient_precision,
        "signal_means_by_label": signal_means_by_label,
        "binary_ranking": binary_ranking,
        "evaluated_rows": evaluated_rows,
        "nli_enabled": nli_scorer is not None,
    }


def build_comparison_summary(
    variant_summaries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    baseline = variant_summaries.get("baseline")
    noisefilter = variant_summaries.get("noisefilter")
    if baseline is None or noisefilter is None:
        return {}

    ranking_delta: dict[str, dict[str, float | None]] = {}
    for signal in SIGNAL_FIELDS:
        baseline_rank = baseline["binary_ranking"].get(signal, {})
        noisefilter_rank = noisefilter["binary_ranking"].get(signal, {})
        ranking_delta[signal] = {}
        for metric in ("roc_auc", "average_precision"):
            base_value = baseline_rank.get(metric)
            nf_value = noisefilter_rank.get(metric)
            ranking_delta[signal][metric] = (
                None if base_value is None or nf_value is None else nf_value - base_value
            )

    return {
        "precision_delta": {
            "strict": noisefilter["strict_precision"] - baseline["strict_precision"],
            "lenient": noisefilter["lenient_precision"] - baseline["lenient_precision"],
        },
        "ranking_delta": ranking_delta,
    }


class LocalNLIScorer:
    def __init__(self, model_name: str, batch_size: int = 8) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError(
                "Local NLI requires sentence-transformers. Install it in .venv first."
            ) from exc

        resolved_model_name = resolve_local_model_path(model_name)
        self._model = CrossEncoder(resolved_model_name, local_files_only=True)
        self._batch_size = max(1, int(batch_size))
        self._label_to_index = self._resolve_label_mapping()

    def _resolve_label_mapping(self) -> dict[str, int]:
        config = getattr(getattr(self._model, "model", None), "config", None)
        id2label = getattr(config, "id2label", {}) or {}
        normalized: dict[str, int] = {}
        for index, label in id2label.items():
            normalized[str(label).strip().lower()] = int(index)
        if not normalized:
            normalized = {"contradiction": 0, "entailment": 1, "neutral": 2}
        return normalized

    def __call__(self, premise: str, hypothesis: str) -> float:
        return float(self.score_many([(premise, hypothesis)])[0])

    def score_many(self, pairs: list[tuple[str, str]]) -> list[float]:
        logits = np.asarray(
            self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        )
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        probs = _softmax(logits)[0]

        entail_index = _match_label_index(self._label_to_index, "entail")
        if entail_index is None:
            raise RuntimeError("Unable to locate entailment label in NLI model config.")
        if logits.shape[0] == 1:
            return [float(probs[entail_index])]
        probs = _softmax(logits)
        return [float(row[entail_index]) for row in probs]


def _match_label_index(label_to_index: dict[str, int], prefix: str) -> int | None:
    for label, index in label_to_index.items():
        if prefix in label:
            return index
    return None


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = np.sum(exp, axis=1, keepdims=True)
    return exp / denom


def resolve_local_model_path(model_name: str) -> str:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return str(candidate)

    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_name, local_files_only=True)
    except Exception:
        return model_name


def build_report(
    variant_files: dict[str, Path],
    *,
    text_chunk_files: dict[str, Path] | None = None,
    nli_scorer: Callable[[str, str], float] | None = None,
) -> dict[str, Any]:
    text_chunk_files = text_chunk_files or {}
    variants: dict[str, dict[str, Any]] = {}
    for variant, path in variant_files.items():
        lookup = (
            load_text_chunks(text_chunk_files[variant])
            if variant in text_chunk_files
            else None
        )
        variants[variant] = evaluate_variant(
            variant,
            load_edge_rows(path),
            text_chunk_lookup=lookup,
            nli_scorer=nli_scorer,
        )
    return {
        "variants": variants,
        "comparison": build_comparison_summary(variants),
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    serializable = {
        "variants": {
            variant: {
                key: value
                for key, value in summary.items()
                if key != "evaluated_rows"
            }
            for variant, summary in payload.get("variants", {}).items()
        },
        "comparison": payload.get("comparison", {}),
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_csv(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for variant, summary in payload.get("variants", {}).items():
        row: dict[str, Any] = {
            "variant": variant,
            "total_edges": summary["total_edges"],
            "correct": summary["label_counts"]["correct"],
            "wrong": summary["label_counts"]["wrong"],
            "ambiguous": summary["label_counts"]["ambiguous"],
            "strict_precision": summary["strict_precision"],
            "lenient_precision": summary["lenient_precision"],
        }
        for signal in SIGNAL_FIELDS:
            ranking = summary["binary_ranking"][signal]
            row[f"{signal}_roc_auc"] = ranking["roc_auc"]
            row[f"{signal}_average_precision"] = ranking["average_precision"]
        rows.append(row)

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_rows_csv(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for variant, summary in payload.get("variants", {}).items():
        for row in summary.get("evaluated_rows", []):
            rows.append(
                {
                    **row,
                    "chunk_ids": json.dumps(row.get("chunk_ids", []), ensure_ascii=False),
                    "source_chunks": json.dumps(
                        row.get("source_chunks", []), ensure_ascii=False
                    ),
                    "hypotheses": json.dumps(
                        row.get("hypotheses", []), ensure_ascii=False
                    ),
                }
            )

    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_markdown(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Grounding Signal Evaluation",
        "",
        "| Variant | Total | Correct | Wrong | Ambiguous | Strict Precision | Lenient Precision | NLI Enabled |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for variant, summary in payload.get("variants", {}).items():
        label_counts = summary["label_counts"]
        lines.append(
            f"| {variant} | {summary['total_edges']} | {label_counts['correct']} | {label_counts['wrong']} | {label_counts['ambiguous']} | {summary['strict_precision']:.4f} | {summary['lenient_precision']:.4f} | {summary['nli_enabled']} |"
        )

    comparison = payload.get("comparison", {})
    if comparison:
        lines.extend(
            [
                "",
                f"Strict precision delta (noisefilter - baseline): {comparison['precision_delta']['strict']:.4f}",
                f"Lenient precision delta (noisefilter - baseline): {comparison['precision_delta']['lenient']:.4f}",
                "",
            ]
        )

    for variant, summary in payload.get("variants", {}).items():
        lines.extend(
            [
                f"## {variant}",
                "",
                "| Signal | Correct Mean | Wrong Mean | Ambiguous Mean | ROC AUC | Average Precision |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for signal in SIGNAL_FIELDS:
            means = summary["signal_means_by_label"][signal]
            ranking = summary["binary_ranking"][signal]
            roc_auc = ranking["roc_auc"]
            avg_precision = ranking["average_precision"]
            lines.append(
                f"| {signal} | {means['correct']:.4f} | {means['wrong']:.4f} | {means['ambiguous']:.4f} | {_format_optional_metric(roc_auc)} | {_format_optional_metric(avg_precision)} |"
            )
        lines.append("")

    output_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _format_optional_metric(value: float | None) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate source-grounded signals on manually labeled edge samples."
    )
    parser.add_argument(
        "--variant-file",
        action="append",
        required=True,
        help="Variant mapping in the format variant=path/to/labeled.csv",
    )
    parser.add_argument(
        "--text-chunks-file",
        action="append",
        default=[],
        help="Optional mapping variant=path/to/kv_store_text_chunks.json",
    )
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--nli-model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--nli-batch-size", type=int, default=8)
    parser.add_argument("--disable-nli", action="store_true")
    return parser.parse_args()


def _parse_mapping(values: list[str]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid mapping value: {raw}. Expected variant=path.")
        variant, path = raw.split("=", 1)
        mapping[variant.strip().lower()] = Path(path).expanduser()
    return mapping


def main() -> None:
    args = parse_args()
    variant_files = _parse_mapping(args.variant_file)
    text_chunk_files = _parse_mapping(args.text_chunks_file)
    nli_scorer = None if args.disable_nli else LocalNLIScorer(
        args.nli_model,
        batch_size=args.nli_batch_size,
    )
    report = build_report(
        variant_files,
        text_chunk_files=text_chunk_files,
        nli_scorer=nli_scorer,
    )

    output_prefix = Path(args.output_prefix)
    save_json(output_prefix.with_suffix(".json"), report)
    save_csv(output_prefix.with_suffix(".csv"), report)
    save_markdown(output_prefix.with_suffix(".md"), report)
    save_rows_csv(output_prefix.with_name(output_prefix.name + ".rows.csv"), report)

    print("Grounding signal evaluation completed.")
    print(f"  json: {output_prefix.with_suffix('.json')}")
    print(f"  csv: {output_prefix.with_suffix('.csv')}")
    print(f"  markdown: {output_prefix.with_suffix('.md')}")
    print(f"  rows: {output_prefix.with_name(output_prefix.name + '.rows.csv')}")
    for variant, summary in report["variants"].items():
        nli_auc = summary["binary_ranking"]["nli_support_score"]["roc_auc"]
        print(
            f"  {variant}: strict_precision={summary['strict_precision']:.4f} "
            f"endpoint_anchor_auc={_format_optional_metric(summary['binary_ranking']['endpoint_anchor_score']['roc_auc'])} "
            f"nli_auc={_format_optional_metric(nli_auc)}"
        )


if __name__ == "__main__":
    main()
