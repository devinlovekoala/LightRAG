# NoiseFilter Full-Pipeline Reproduction Analysis - 2026-04-20

## Scope

This note records the completed `mix` dataset NoiseFilter reproduction round on the
`noisefilter_mix_stage2_qdrant` workspace. The run is treated as a usable
engineering reproduction baseline for this round, with the caveat that an
interrupted `max_gleaning=0` rescue briefly processed one document before it was
reset and rerun with `max_gleaning=1`.

Final resume command:

```bash
./.venv/bin/python -u reproduce/resume_noisefilter_insert.py \
  --dataset mix \
  --working-root rag_storage/formal_runs_stage2_qdrant \
  --enable-chunk-binding-gate \
  --chunk-size 2500 \
  --chunk-overlap-size 100 \
  --max-gleaning 1 \
  --max-async 4 \
  --embedding-max-async 8 \
  --max-parallel-insert 1 \
  --llm-timeout 600
```

The successful log is:

```text
reproduce/results/formal/mix/noisefilter/resume_insert_20260420_g1.log
```

## Completion Evidence

The final resume log reported:

```text
Formal NoiseFilter resume completed.
dataset_contexts: 61
missing_enqueued: 0
retryable_before: 38
status_counts_before: {'pending': 37, 'processed': 23, 'processing': 1}
status_counts_after: {'processed': 61}
entity_extract_max_gleaning: 1
```

Direct PostgreSQL status verification for the real dataset document hashes:

```text
DATASET_STATUS_COUNTS {"processed": 61}
DATASET_RETRYABLE []
```

The workspace still contains 61 `failed` `doc_status` rows, but these are
duplicate tracking records created by earlier repeated insert attempts. They are
not real dataset failures.

## Current Workspace Scale

Read-only workspace statistics after completion:

```text
workspace: noisefilter_mix_stage2_qdrant
real documents: 61
text chunks: 279
graph nodes: 12,638
graph edges: 8,879
edges with conf_score: 8,879 / 8,879
LLM cache rows: 599
```

Confidence score distribution:

```text
conf_score mean:   0.50696
conf_score median: 0.48412
conf_score min:    0.22920
conf_score max:    0.90000

conf_score bins:
  < 0.3:      18
  0.3-0.5: 5,224
  0.5-0.7: 3,550
  >= 0.7:     87
```

Component score behavior:

```text
conf_freq_score median:        0.33333
conf_consistency_score median: 1.00000
conf_semantic_score median:    0.09463
```

Interpretation: confidence scoring is fully populated, but the distribution is
compressed near 0.5. With `conf_threshold=0.3`, hard filtering removes only
18 of 8,879 edges, so the current pipeline mostly behaves as confidence-aware
soft weighting rather than aggressive hard deletion.

## Effectiveness Metrics

### Edge-Level Precision

The manually labeled 100-edge sample shows the clearest positive signal:

```text
baseline strict_precision:    0.15
noisefilter strict_precision: 0.22
delta: +0.07

baseline lenient_precision:    0.50
noisefilter lenient_precision: 0.57
delta: +0.07
```

Label counts:

```text
baseline:    correct 15, wrong 50, ambiguous 35
noisefilter: correct 22, wrong 43, ambiguous 35
```

This supports the claim that NoiseFilter reduced noisy relation edges in the
sample. The sample size is limited, so this is a directional signal rather than
a statistically strong result.

### Pairwise Answer Judging

Pairwise LLM judging on 130 hybrid queries slightly favored NoiseFilter:

```text
Overall Winner:
  baseline:    57
  noisefilter: 60
  tie:         13

Comprehensiveness:
  baseline:    55
  noisefilter: 59
  tie:         16

Diversity:
  baseline:    22
  noisefilter: 29
  tie:         79

Empowerment:
  baseline:    51
  noisefilter: 59
  tie:         20
```

This is a mild positive signal, but not a strong one.

### Automatic QA Token F1

The automatic hybrid QA metric did not improve:

```text
baseline avg_token_f1:    0.20996
noisefilter avg_token_f1: 0.20676
delta: -0.00320
```

Exact match was zero for both variants. This metric is noisy for long-form RAG
answers, but it still shows that the edge-level improvement did not reliably
transfer to automatic QA scoring in this round.

### Source Binding

Source binding moved in the right direction, but only slightly:

```text
baseline weak_endpoint_binding_rate:    1.2723%
noisefilter weak_endpoint_binding_rate: 1.2520%

baseline both_endpoints_missing_rate:    0.0446%
noisefilter both_endpoints_missing_rate: 0.0224%
```

The relation chunk/entity binding gate appears to reduce the worst endpoint
binding cases, but the absolute effect is small.

## Judgment

The full pipeline works end to end:

```text
dataset ingestion -> chunking -> extraction -> merge -> chunk binding gate
-> NoiseFilter confidence scoring -> PostgreSQL graph/KV + Qdrant vector storage
```

The current evidence supports this conservative conclusion:

```text
NoiseFilter shows a measurable edge-level denoising signal, but the current
confidence scoring is not yet calibrated enough to produce stable downstream
QA gains.
```

The strongest positive evidence is the edge sample precision improvement from
15% to 22% strict precision and from 50% to 57% lenient precision. The weakest
area is score calibration: frequency and consistency are mostly saturated or
constant, while semantic scoring provides variation but has not yet produced a
robust ranking signal.

## Caveats

- The final insert run completed all real documents, but one interrupted
  `max_gleaning=0` rescue briefly merged a document before it was reset and
  rerun with `max_gleaning=1`.
- Existing `hybrid_eval` and pairwise result files predate the final 2026-04-20
  resume completion, so they should be interpreted as reproduction evidence
  from the same experimental line rather than as freshly rerun query results
  against the exact final workspace state.
- The `failed=61` rows in `doc_status` are duplicate tracking records, not real
  unprocessed dataset documents.
- Directionality issues remain in the graph/relation tooling and should be
  addressed before making stronger claims about relation quality.

## Next Steps

1. Re-run hybrid query evaluation against the completed
   `noisefilter_mix_stage2_qdrant` workspace if answer-level claims are needed.
2. Calibrate `conf_score` so correct and wrong edges separate more clearly.
3. Raise or sweep `conf_threshold` for hard-filter ablations, since 0.3 filters
   only 18 of 8,879 current edges.
4. Fix relation directionality collapse before treating edge semantics as final.
5. Compare against a clean baseline workspace with matching PostgreSQL/Qdrant
   storage, chunk size, `max_gleaning=1`, and query settings.
