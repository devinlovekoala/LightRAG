# NoiseFilter-RAG Reproduction on LightRAG

Research-engineering reproduction fork focused on graph noise diagnosis, confidence-aware retrieval, and evidence-backed evaluation for NoiseFilter-RAG.

This repository is built on top of the upstream [HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) project and its paper:
- LightRAG project: <https://github.com/HKUDS/LightRAG>
- LightRAG paper: <https://arxiv.org/abs/2410.05779>

This fork is not just a usage demo. It is a full reproduction workspace with:
- dual-run experiment pipelines for `baseline` and `noisefilter`
- formal dataset ingestion and query runners
- graph-quality sampling and manual labeling support
- pairwise LLM judge evaluation
- source-grounded judge evaluation
- relation directionality and source-binding audit tooling
- stable commit-by-commit milestones for reporting and presentation

## Project Positioning

Problem statement:
- LightRAG relies on LLM-based entity and relation extraction.
- Extraction noise can create wrong edges, redundant neighbors, and polluted retrieval context.
- The original answer-level evaluation does not directly expose graph-noise behavior.

This project targets that gap by turning LightRAG into a reproducible NoiseFilter-RAG study with:
- confidence scoring on extracted relations
- noise-aware retrieval using relation confidence
- formal baseline vs. noisefilter comparisons
- intermediate diagnostics at graph level, judge level, and answer level

## Stable Stage 1

The project has reached its first stable reproduction stage.

Stage-1 scope:
- local environment, Docker/API stack, and formal runners are working
- baseline and noisefilter indexes were both built successfully on the formal `mix` dataset
- answer-level and graph-level evaluation pipelines are in place
- source-grounded judging was stabilized as an operational evaluation track
- failed judge-localization prototype was explicitly diagnosed and reverted

What this means:
- the repo is already suitable for project demo, portfolio presentation, and engineering showcase
- the current results are reproducible
- the remaining work is no longer "make it run", but "push the method further"

## Results Snapshot

### 1. Graph Quality

Manual labeling on 100 sampled edges per variant:

| Metric | Baseline | NoiseFilter | Delta |
| --- | ---: | ---: | ---: |
| Strict precision | 0.15 | 0.22 | +0.07 |
| Lenient precision | 0.50 | 0.57 | +0.07 |

Interpretation:
- NoiseFilter improves graph-level edge precision.
- The method is not a no-op; it creates a measurable graph-quality gain.

Reference files:
- [`reproduce/results/formal/mix/baseline/edge_samples_labeled.csv`](reproduce/results/formal/mix/baseline/edge_samples_labeled.csv)
- [`reproduce/results/formal/mix/noisefilter/nf_edge_samples_labeled.csv`](reproduce/results/formal/mix/noisefilter/nf_edge_samples_labeled.csv)
- [`reproduce/results/formal/mix/labeled_edge_analysis.md`](reproduce/results/formal/mix/labeled_edge_analysis.md)

### 2. Answer-Level Pairwise Evaluation

Formal pairwise LLM judge on baseline vs. noisefilter answers:

| Metric | Value |
| --- | ---: |
| NoiseFilter overall win rate | 0.4615 |
| Baseline overall win rate | 0.4385 |

Interpretation:
- NoiseFilter shows a small but real edge at answer level.
- The gain is weaker than the graph-level improvement, which is exactly why this repo includes intermediate diagnostics.

Reference files:
- [`reproduce/results/formal/mix/pairwise_hybrid_eval.md`](reproduce/results/formal/mix/pairwise_hybrid_eval.md)

### 3. Source-Grounded Judge Status

The source-grounded judge is now a stable evaluation tool operationally, but its scoring remains sensitive to source selection mode and prompt design. That makes it valuable as a diagnostic instrument, while also signaling that judge calibration is still part of the active research work.

Current published report:

| Metric | Baseline | NoiseFilter | Delta |
| --- | ---: | ---: | ---: |
| ROC AUC | 0.5413 | 0.5497 | +0.0084 |
| AP | 0.5488 | 0.4627 | -0.0861 |

Interpretation:
- the judge can separate supported from unsupported edges better than the earliest failed setups
- the metric is not yet robust enough to serve as the single headline proof of improvement
- this repo therefore keeps graph-quality analysis and pairwise answer judging as equally important evidence tracks

Reference files:
- [`reproduce/results/formal/mix/source_grounded_judge_eval_qwen3.md`](reproduce/results/formal/mix/source_grounded_judge_eval_qwen3.md)

### 4. Synthetic Noise Benchmark

Local synthetic benchmark and demo pipeline are working end-to-end.

Representative sanity-check result:
- precision = 1.000
- recall = 1.000
- f1 = 1.000

Interpretation:
- the confidence-aware filtering logic works correctly under controlled injected-noise settings
- this does not replace formal evaluation, but it validates core behavior

## Engineering Deliverables

### Core NoiseFilter Modules

- [`lightrag/noisefilter/confidence.py`](lightrag/noisefilter/confidence.py)
- [`lightrag/noisefilter/retriever.py`](lightrag/noisefilter/retriever.py)
- [`lightrag/noisefilter/benchmark.py`](lightrag/noisefilter/benchmark.py)
- [`lightrag/noisefilter/reproduction.py`](lightrag/noisefilter/reproduction.py)

### Formal Reproduction Runners

- [`reproduce/run_baseline_insert.py`](reproduce/run_baseline_insert.py)
- [`reproduce/run_noisefilter_insert.py`](reproduce/run_noisefilter_insert.py)
- [`reproduce/run_baseline_query.py`](reproduce/run_baseline_query.py)
- [`reproduce/run_noisefilter_query.py`](reproduce/run_noisefilter_query.py)
- [`reproduce/evaluate_formal_results.py`](reproduce/evaluate_formal_results.py)
- [`reproduce/formal_pairwise_eval.py`](reproduce/formal_pairwise_eval.py)

### Graph and Judge Diagnostics

- [`reproduce/export_graph_edge_samples.py`](reproduce/export_graph_edge_samples.py)
- [`reproduce/analyze_labeled_edge_samples.py`](reproduce/analyze_labeled_edge_samples.py)
- [`reproduce/evaluate_grounding_signals.py`](reproduce/evaluate_grounding_signals.py)
- [`reproduce/evaluate_source_grounded_judge.py`](reproduce/evaluate_source_grounded_judge.py)
- [`reproduce/audit_relation_directionality.py`](reproduce/audit_relation_directionality.py)
- [`reproduce/audit_relation_source_binding.py`](reproduce/audit_relation_source_binding.py)

### Demonstration and Verification

- [`examples/noisefilter_demo.py`](examples/noisefilter_demo.py)
- [`tests/test_noise_filter.py`](tests/test_noise_filter.py)
- [`tests/test_noise_filter_experiment.py`](tests/test_noise_filter_experiment.py)
- [`tests/test_noise_filter_formal_reproduction.py`](tests/test_noise_filter_formal_reproduction.py)
- [`tests/test_source_grounded_judge_eval.py`](tests/test_source_grounded_judge_eval.py)
- [`tests/test_relation_directionality_audit.py`](tests/test_relation_directionality_audit.py)
- [`tests/test_relation_source_binding_audit.py`](tests/test_relation_source_binding_audit.py)

## Main Findings So Far

### Confirmed

- NoiseFilter improves graph-edge precision on the formal benchmark.
- NoiseFilter shows a small answer-level win-rate advantage.
- The source-grounded judge is now reproducible enough to support calibration work and ablation analysis.
- The project has already gone beyond "toy demo" into reproducible comparative evaluation.

### Diagnosed

- The original confidence sub-signals were largely non-discriminative on the formal dataset.
- Directionality and source-binding risks exist in the broader architecture and required dedicated audits.
- Evidence-localization heuristics can easily overfit lexical overlap and destroy judge discrimination if inserted directly into the main evaluation path.

### Practical Conclusion

The current Stage-1 result is:
- graph quality improved clearly
- answer quality improved slightly
- judge-based grounding evaluation is now available, but still needs calibration before it becomes a decisive headline metric
- intermediate evaluation tooling is strong enough to support serious iterative research

## Why This Repo Demonstrates Engineering Ability

This fork demonstrates more than model calling or script orchestration.

It shows:
- end-to-end experiment design
- reproducible benchmarking
- quantitative diagnosis instead of intuition-only debugging
- stable milestone management through Git history
- ability to recover from failed directions by auditing, isolating, and reverting safely
- disciplined separation of stable evaluation paths vs. experimental prototypes

In short, this is an engineering-heavy research reproduction, not just a paper reimplementation.

## Quick Start

### Local Validation

```bash
./scripts/test.sh tests/test_noise_filter.py
./scripts/test.sh tests/test_noise_filter_experiment.py
python3 examples/noisefilter_demo.py
python3 reproduce/noisefilter_experiment.py
```

### Formal Reproduction

Use the detailed guide:
- [`docs/NoiseFilterRAG_Reproduction_Guide.md`](docs/NoiseFilterRAG_Reproduction_Guide.md)

Typical workflow:
1. configure `.env`
2. bring up Docker/API services if needed
3. build baseline index
4. build noisefilter index
5. run formal queries
6. run answer evaluation
7. run pairwise judge
8. run graph-quality and source-grounded diagnostics

### Stable Judge Notes

The current recommended source-grounded judge script supports explicit source selection:

```bash
./.venv/bin/python reproduce/evaluate_source_grounded_judge.py \
  --variant-file baseline=... \
  --variant-file noisefilter=... \
  --text-chunks-file baseline=... \
  --text-chunks-file noisefilter=... \
  --source-mode exported
```

Available source modes:
- `auto`
- `exported`
- `resolved`

The evidence-localization prototype was intentionally reverted from the stable path because it reduced judge discrimination.

## Repository Map

| Area | Purpose |
| --- | --- |
| `lightrag/noisefilter/` | Core method implementation |
| `reproduce/` | Reproduction runners, evaluation, and audit scripts |
| `examples/` | Lightweight demos |
| `tests/` | Unit and regression coverage |
| `docs/` | Reproduction and operational guides |
| `rag_storage/` | Local experiment workspaces |
| `reproduce/results/` | Exported experiment results and reports |

## Documentation

- Reproduction guide: [`docs/NoiseFilterRAG_Reproduction_Guide.md`](docs/NoiseFilterRAG_Reproduction_Guide.md)
- Formal experiment outputs: [`reproduce/results/formal/mix/`](reproduce/results/formal/mix/)

## Selected Milestones

Examples of major engineering milestones in this fork:
- formal UltraDomain-style reproduction tooling
- stable dual-run baseline/noisefilter workflow
- pairwise answer evaluation
- labeled graph-edge analysis
- grounding-signal evaluation
- source-grounded judge hardening
- relation directionality audit
- relation source-binding audit
- experimental evidence localization prototype and controlled rollback

This history matters because it shows the project evolved through evidence-backed decisions, not ad hoc tweaking.

## Current Stage and Next Stage

### Current Stage

Stage 1 is complete:
- infrastructure is stable
- formal results exist
- diagnostics exist
- stable judge path exists
- README and docs now support professional presentation

### Next Stage

Stage 2 is focused on method improvement, not plumbing:
- redesign confidence scoring with stronger grounding signals
- improve relation evidence quality without breaking judge discrimination
- tighten graph-noise to answer-quality transfer
- prepare cleaner final result tables and figures

## Acknowledgement

This repository is built on the excellent upstream LightRAG project from HKUDS. The work here focuses on reproduction, diagnosis, and method extension for NoiseFilter-RAG.
