from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


def _coerce_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(slots=True)
class NoiseFilterConfig:
    enabled: bool = False
    conf_threshold: float = 0.3
    soft_mode: bool = True
    default_conf_score: float = 0.5

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any] | None = None,
        *,
        enabled: bool | None = None,
    ) -> "NoiseFilterConfig":
        raw_mapping = dict(mapping or {})

        if (
            "noise_filter_config" in raw_mapping
            or "enable_noise_filter" in raw_mapping
        ):
            nested = raw_mapping.get("noise_filter_config") or {}
            raw_mapping = dict(nested)
            if enabled is None:
                enabled = bool(mapping.get("enable_noise_filter", False))  # type: ignore[arg-type]

        if enabled is None:
            enabled = bool(raw_mapping.get("enabled", False))

        return cls(
            enabled=enabled,
            conf_threshold=max(
                0.0,
                min(1.0, _coerce_float(raw_mapping.get("conf_threshold"), 0.3)),
            ),
            soft_mode=bool(raw_mapping.get("soft_mode", True)),
            default_conf_score=max(
                0.0,
                min(1.0, _coerce_float(raw_mapping.get("default_conf_score"), 0.5)),
            ),
        )


def resolve_noise_filter_config(
    global_config: Mapping[str, Any] | None = None,
) -> NoiseFilterConfig:
    return NoiseFilterConfig.from_mapping(global_config or {})


class NoiseAwareRetriever:
    def __init__(
        self,
        *,
        conf_threshold: float = 0.3,
        soft_mode: bool = True,
        enabled: bool = True,
        default_conf_score: float = 0.5,
    ) -> None:
        self.conf_threshold = max(0.0, min(1.0, conf_threshold))
        self.soft_mode = soft_mode
        self.enabled = enabled
        self.default_conf_score = max(0.0, min(1.0, default_conf_score))

    @classmethod
    def from_global_config(
        cls, global_config: Mapping[str, Any] | None
    ) -> "NoiseAwareRetriever":
        config = resolve_noise_filter_config(global_config)
        return cls(
            conf_threshold=config.conf_threshold,
            soft_mode=config.soft_mode,
            enabled=config.enabled,
            default_conf_score=config.default_conf_score,
        )

    def _extract_confidence(self, edge: Mapping[str, Any]) -> float:
        return _coerce_float(edge.get("conf_score"), self.default_conf_score)

    def _should_keep(self, conf_score: float) -> bool:
        if not self.enabled or self.soft_mode:
            return True
        return conf_score >= self.conf_threshold

    def _finalize_edge(
        self,
        edge: Mapping[str, Any],
        *,
        conf_score: float,
        base_score: float,
    ) -> dict[str, Any]:
        item = dict(edge)
        item["conf_score"] = conf_score
        item["noise_adjusted_score"] = (
            base_score * conf_score if self.enabled and self.soft_mode else base_score
        )
        return item

    def rank_local_edges(self, edges: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []

        for edge in edges:
            conf_score = self._extract_confidence(edge)
            if not self._should_keep(conf_score):
                continue

            rank = _coerce_float(edge.get("rank"), 0.0)
            weight = _coerce_float(edge.get("weight"), 1.0)
            base_score = rank + weight
            ranked.append(
                self._finalize_edge(
                    edge,
                    conf_score=conf_score,
                    base_score=base_score,
                )
            )

        ranked.sort(
            key=lambda item: (
                _coerce_float(item.get("noise_adjusted_score"), 0.0),
                _coerce_float(item.get("conf_score"), self.default_conf_score),
                _coerce_float(item.get("rank"), 0.0),
                _coerce_float(item.get("weight"), 1.0),
            ),
            reverse=True,
        )
        return ranked

    def rank_global_edges(self, edges: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
        ranked: list[dict[str, Any]] = []
        fallback_distance = float(len(edges))

        for index, edge in enumerate(edges):
            conf_score = self._extract_confidence(edge)
            if not self._should_keep(conf_score):
                continue

            distance = _coerce_float(edge.get("distance"), fallback_distance - index)
            ranked.append(
                self._finalize_edge(
                    edge,
                    conf_score=conf_score,
                    base_score=distance,
                )
            )

        ranked.sort(
            key=lambda item: (
                _coerce_float(item.get("noise_adjusted_score"), 0.0),
                _coerce_float(item.get("conf_score"), self.default_conf_score),
                _coerce_float(item.get("distance"), 0.0),
            ),
            reverse=True,
        )
        return ranked
