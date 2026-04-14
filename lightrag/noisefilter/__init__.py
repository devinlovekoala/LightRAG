__all__ = [
    "ConfidenceScoringEngine",
    "NoiseAwareRetriever",
    "NoiseFilterConfig",
    "NoiseInjector",
    "evaluate_filter",
    "resolve_noise_filter_config",
]


def __getattr__(name: str):
    if name == "ConfidenceScoringEngine":
        from .confidence import ConfidenceScoringEngine

        return ConfidenceScoringEngine
    if name in {"NoiseAwareRetriever", "NoiseFilterConfig", "resolve_noise_filter_config"}:
        from .retriever import (
            NoiseAwareRetriever,
            NoiseFilterConfig,
            resolve_noise_filter_config,
        )

        values = {
            "NoiseAwareRetriever": NoiseAwareRetriever,
            "NoiseFilterConfig": NoiseFilterConfig,
            "resolve_noise_filter_config": resolve_noise_filter_config,
        }
        return values[name]
    if name in {"NoiseInjector", "evaluate_filter"}:
        from .benchmark import NoiseInjector, evaluate_filter

        values = {
            "NoiseInjector": NoiseInjector,
            "evaluate_filter": evaluate_filter,
        }
        return values[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
