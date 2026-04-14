from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from lightrag.base import BaseGraphStorage, BaseKVStorage
from lightrag.constants import GRAPH_FIELD_SEP
from lightrag.prompt import PROMPTS
from lightrag.utils import (
    logger,
    make_relation_chunk_key,
    sanitize_and_normalize_extracted_text,
    split_string_by_multi_markers,
)


def _normalize_signature(value: str) -> str:
    return " ".join(value.strip().lower().split())


def _coerce_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def compute_freq_score(support_count: int, saturation: int = 3) -> float:
    if saturation <= 0:
        return 1.0
    return max(0.0, min(support_count / float(saturation), 1.0))


def compute_consistency_score(unique_relations: int) -> float:
    if unique_relations <= 1:
        return 1.0
    return 1.0 / float(unique_relations)


@dataclass(slots=True)
class EdgeEvidence:
    src_id: str
    tgt_id: str
    edge_data: dict[str, Any]
    relation_mentions: list[dict[str, Any]]


class ConfidenceScoringEngine:
    def __init__(
        self,
        *,
        embedding_model,
        relation_chunks_storage: BaseKVStorage | None = None,
        text_chunks_storage: BaseKVStorage | None = None,
        llm_response_cache: BaseKVStorage | None = None,
        w_freq: float = 0.5,
        w_cons: float = 0.3,
        w_sem: float = 0.2,
        freq_saturation: int = 3,
        semantic_fallback_score: float = 0.5,
    ) -> None:
        self.embedding_model = embedding_model
        self.relation_chunks_storage = relation_chunks_storage
        self.text_chunks_storage = text_chunks_storage
        self.llm_response_cache = llm_response_cache
        self.w_freq = w_freq
        self.w_cons = w_cons
        self.w_sem = w_sem
        self.freq_saturation = max(1, freq_saturation)
        self.semantic_fallback_score = max(0.0, min(1.0, semantic_fallback_score))

    async def score_and_update_graph(
        self,
        graph_storage: BaseGraphStorage,
        edge_pairs: Iterable[tuple[str, str]] | None = None,
    ) -> dict[tuple[str, str], dict[str, Any]]:
        evidences = await self._collect_edge_evidences(graph_storage, edge_pairs)
        if not evidences:
            return {}

        semantic_scores = await self._compute_semantic_scores(evidences)
        results: dict[tuple[str, str], dict[str, Any]] = {}

        for edge_pair, evidence in evidences.items():
            relation_counter = Counter()
            for mention in evidence.relation_mentions:
                signature = self._relation_signature(mention)
                if signature:
                    relation_counter[signature] += 1

            if relation_counter:
                support_count = sum(relation_counter.values())
                dominant_support = max(relation_counter.values())
                relation_variants = len(relation_counter)
            else:
                support_count = max(1, len(evidence.relation_mentions))
                dominant_support = support_count
                relation_variants = 1 if support_count else 0

            freq_score = compute_freq_score(dominant_support, self.freq_saturation)
            consistency_score = compute_consistency_score(relation_variants)
            semantic_score = semantic_scores.get(
                edge_pair, self.semantic_fallback_score
            )
            conf_score = (
                self.w_freq * freq_score
                + self.w_cons * consistency_score
                + self.w_sem * semantic_score
            )
            conf_score = max(0.0, min(1.0, conf_score))

            persisted_edge = dict(evidence.edge_data)
            persisted_edge.pop("source", None)
            persisted_edge.pop("target", None)
            updated_edge = {
                **persisted_edge,
                "conf_score": conf_score,
                "conf_freq_score": freq_score,
                "conf_consistency_score": consistency_score,
                "conf_semantic_score": semantic_score,
                "conf_support": support_count,
                "conf_relation_variants": relation_variants,
            }
            await graph_storage.upsert_edge(
                evidence.src_id,
                evidence.tgt_id,
                updated_edge,
            )
            result_key = edge_pair if edge_pairs is not None else (
                evidence.src_id,
                evidence.tgt_id,
            )
            results[result_key] = {
                "conf_score": conf_score,
                "freq_score": freq_score,
                "consistency_score": consistency_score,
                "semantic_score": semantic_score,
                "support_count": support_count,
                "relation_variants": relation_variants,
            }

        return results

    async def _collect_edge_evidences(
        self,
        graph_storage: BaseGraphStorage,
        edge_pairs: Iterable[tuple[str, str]] | None,
    ) -> dict[tuple[str, str], EdgeEvidence]:
        selected_pairs = list(edge_pairs) if edge_pairs is not None else None

        evidences: dict[tuple[str, str], EdgeEvidence] = {}

        if selected_pairs is None:
            all_edges = await graph_storage.get_all_edges()
            for edge in all_edges:
                src_id = edge.get("source")
                tgt_id = edge.get("target")
                if not src_id or not tgt_id:
                    continue
                edge_pair = tuple(sorted((src_id, tgt_id)))
                evidences[edge_pair] = EdgeEvidence(
                    src_id=edge_pair[0],
                    tgt_id=edge_pair[1],
                    edge_data=edge,
                    relation_mentions=[],
                )
        else:
            for src_id, tgt_id in selected_pairs:
                edge_data = await graph_storage.get_edge(src_id, tgt_id)
                if edge_data is None and src_id != tgt_id:
                    edge_data = await graph_storage.get_edge(tgt_id, src_id)
                if edge_data is None:
                    continue
                evidences[(src_id, tgt_id)] = EdgeEvidence(
                    src_id=src_id,
                    tgt_id=tgt_id,
                    edge_data=dict(edge_data),
                    relation_mentions=[],
                )

        for edge_pair, evidence in evidences.items():
            evidence.relation_mentions = await self._collect_relation_mentions(
                evidence.src_id,
                evidence.tgt_id,
                evidence.edge_data,
            )
            if evidence.relation_mentions:
                first_mention = evidence.relation_mentions[0]
                mention_src = first_mention.get("src_id")
                mention_tgt = first_mention.get("tgt_id")
                if isinstance(mention_src, str) and isinstance(mention_tgt, str):
                    evidence.src_id = mention_src
                    evidence.tgt_id = mention_tgt

        return evidences

    async def _collect_relation_mentions(
        self,
        src_id: str,
        tgt_id: str,
        edge_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        chunk_ids = await self._get_relation_chunk_ids(src_id, tgt_id, edge_data)
        if (
            not chunk_ids
            or self.text_chunks_storage is None
            or self.llm_response_cache is None
        ):
            return [dict(edge_data)]

        cached_results = await self._get_cached_extraction_results(chunk_ids)
        mentions: list[dict[str, Any]] = []

        for chunk_id, extraction_entries in cached_results.items():
            for extraction_result, timestamp in extraction_entries:
                mentions.extend(
                    self._parse_relation_mentions_from_result(
                        extraction_result, src_id, tgt_id
                    )
                )

        return mentions or [dict(edge_data)]

    async def _get_cached_extraction_results(
        self, chunk_ids: list[str]
    ) -> dict[str, list[tuple[str, int]]]:
        if self.text_chunks_storage is None or self.llm_response_cache is None:
            return {}

        chunk_data_list = await self.text_chunks_storage.get_by_ids(chunk_ids)
        cache_ids: list[str] = []
        cache_to_chunk: dict[str, str] = {}

        for chunk_id, chunk_data in zip(chunk_ids, chunk_data_list):
            if not chunk_data or not isinstance(chunk_data, dict):
                continue
            for cache_id in chunk_data.get("llm_cache_list", []):
                if isinstance(cache_id, str) and cache_id:
                    cache_ids.append(cache_id)
                    cache_to_chunk[cache_id] = chunk_id

        if not cache_ids:
            return {}

        cache_entries = await self.llm_response_cache.get_by_ids(cache_ids)
        grouped: dict[str, list[tuple[str, int]]] = {}

        for cache_id, cache_entry in zip(cache_ids, cache_entries):
            if not cache_entry or not isinstance(cache_entry, dict):
                continue
            if cache_entry.get("cache_type") != "extract":
                continue

            chunk_id = cache_entry.get("chunk_id") or cache_to_chunk.get(cache_id)
            extraction_result = cache_entry.get("return")
            if not chunk_id or not isinstance(extraction_result, str):
                continue

            grouped.setdefault(chunk_id, []).append(
                (
                    extraction_result,
                    int(_coerce_float(cache_entry.get("create_time"), 0)),
                )
            )

        for chunk_id in grouped:
            grouped[chunk_id].sort(key=lambda item: item[1])

        return grouped

    async def _get_relation_chunk_ids(
        self,
        src_id: str,
        tgt_id: str,
        edge_data: dict[str, Any],
    ) -> list[str]:
        if self.relation_chunks_storage is not None:
            storage_key = make_relation_chunk_key(src_id, tgt_id)
            stored = await self.relation_chunks_storage.get_by_id(storage_key)
            if stored and isinstance(stored.get("chunk_ids"), list):
                return [chunk_id for chunk_id in stored["chunk_ids"] if chunk_id]

        source_id = edge_data.get("source_id")
        if isinstance(source_id, str) and source_id:
            if GRAPH_FIELD_SEP in source_id:
                return [
                    chunk_id
                    for chunk_id in source_id.split(GRAPH_FIELD_SEP)
                    if chunk_id
                ]
            return [source_id]
        return []

    def _relation_signature(self, relation_data: dict[str, Any]) -> str:
        keywords = relation_data.get("keywords")
        if isinstance(keywords, str) and keywords.strip():
            return _normalize_signature(keywords)

        description = relation_data.get("description")
        if isinstance(description, str) and description.strip():
            return _normalize_signature(description)

        return ""

    def _relation_text(self, relation_data: dict[str, Any]) -> str:
        keywords = relation_data.get("keywords")
        if isinstance(keywords, str) and keywords.strip():
            return keywords.strip()

        description = relation_data.get("description")
        if isinstance(description, str):
            return description.strip()

        return ""

    def _parse_relation_mentions_from_result(
        self, extraction_result: str, src_id: str, tgt_id: str
    ) -> list[dict[str, Any]]:
        tuple_delimiter = PROMPTS["DEFAULT_TUPLE_DELIMITER"]
        completion_delimiter = PROMPTS["DEFAULT_COMPLETION_DELIMITER"]
        mentions: list[dict[str, Any]] = []

        records = split_string_by_multi_markers(
            extraction_result,
            ["\n", completion_delimiter, completion_delimiter.lower()],
        )

        for record in records:
            record = record.strip()
            if not record:
                continue

            attributes = split_string_by_multi_markers(record, [tuple_delimiter])
            if len(attributes) < 5 or "relation" not in attributes[0]:
                continue

            source = sanitize_and_normalize_extracted_text(
                attributes[1], remove_inner_quotes=True
            )
            target = sanitize_and_normalize_extracted_text(
                attributes[2], remove_inner_quotes=True
            )
            if {source, target} != {src_id, tgt_id}:
                continue

            keywords = sanitize_and_normalize_extracted_text(
                attributes[3], remove_inner_quotes=True
            ).replace("，", ",")
            description = sanitize_and_normalize_extracted_text(attributes[4])
            mentions.append(
                {
                    "src_id": source,
                    "tgt_id": target,
                    "keywords": keywords,
                    "description": description,
                }
            )

        return mentions

    async def _compute_semantic_scores(
        self,
        evidences: dict[tuple[str, str], EdgeEvidence],
    ) -> dict[tuple[str, str], float]:
        if self.embedding_model is None:
            return {
                edge_pair: self.semantic_fallback_score for edge_pair in evidences
            }

        texts: list[str] = []
        valid_pairs: list[tuple[str, str]] = []
        scores = {
            edge_pair: self.semantic_fallback_score for edge_pair in evidences
        }

        for edge_pair, evidence in evidences.items():
            relation_text = self._relation_text(evidence.edge_data)
            if not relation_text:
                continue

            texts.extend([evidence.src_id, evidence.tgt_id, relation_text])
            valid_pairs.append(edge_pair)

        if not valid_pairs:
            return scores

        try:
            embeddings = np.asarray(await self.embedding_model(texts))
        except Exception as exc:
            logger.warning("NoiseFilter semantic scoring fell back to default: %s", exc)
            return scores

        for index, edge_pair in enumerate(valid_pairs):
            src_embedding = np.asarray(embeddings[index * 3], dtype=np.float32)
            tgt_embedding = np.asarray(embeddings[index * 3 + 1], dtype=np.float32)
            relation_embedding = np.asarray(
                embeddings[index * 3 + 2], dtype=np.float32
            )

            direction = tgt_embedding - src_embedding
            denom = float(
                np.linalg.norm(direction) * np.linalg.norm(relation_embedding)
            )
            if denom <= 0:
                scores[edge_pair] = self.semantic_fallback_score
                continue

            cosine = float(np.dot(relation_embedding, direction) / denom)
            scores[edge_pair] = max(cosine, 0.0)

        return scores
