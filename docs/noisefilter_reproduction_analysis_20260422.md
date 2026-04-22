# NoiseFilter Full-Pipeline Reproduction Analysis - 2026-04-22

## Scope

This note records the completed `mix` dataset stage-2 formal reproduction round
under the following workspaces:

- baseline: `baseline_mix_stage2_p1`
- noisefilter: `noisefilter_mix_stage2_p1`

This round includes:

1. baseline insert
2. noisefilter insert
3. baseline hybrid query
4. noisefilter hybrid query
5. automatic QA evaluation
6. pairwise answer evaluation

The goal of this round was to validate whether the current `p1` NoiseFilter
configuration improves answer-level quality after introducing:

- `chunk-binding-gate mode=both`
- `relation-entity-gate`
- stronger extraction grounding constraints
- `workspace`-safe insert/query runners

## Completion Evidence

Direct backend verification for the real dataset document hashes:

```text
baseline dataset_contexts: 61
baseline status_counts: {'processed': 61}

noisefilter dataset_contexts: 61
noisefilter status_counts: {'processed': 61}
```

Formal query outputs are complete for both variants:

```text
baseline hybrid_results.json:    130 rows
baseline hybrid_errors.json:       0 rows
noisefilter hybrid_results.json: 130 rows
noisefilter hybrid_errors.json:    0 rows
```

Therefore this round is a clean, fully queryable reproduction result and can be
used as the formal evaluation basis for this stage.

## Workspace Scale

Read-only backend statistics:

```text
baseline_mix_stage2_p1
  text chunks: 279
  graph nodes: 15,624
  graph edges: 11,306

noisefilter_mix_stage2_p1
  text chunks: 279
  graph nodes: 15,748
  graph edges: 11,157
```

Interpretation:

- Both variants used the same chunking configuration (`279` chunks).
- NoiseFilter did **not** produce a dramatically smaller graph in this round.
- Relative to baseline, NoiseFilter added `124` nodes but reduced `149` edges.
- This suggests the current `p1` pipeline is not acting as a simple aggressive
  graph-pruning layer; instead, it is changing graph composition more subtly.

## Automatic QA Metrics

Automatic QA evaluation:

```text
baseline avg_exact_match: 0.0000
baseline avg_token_f1:    0.2203

noisefilter avg_exact_match: 0.0000
noisefilter avg_token_f1:    0.2249

token_f1 delta: +0.0046
relative gain:  +2.1%
```

Per-query token-F1 comparison:

```text
noisefilter better: 71
baseline better:    58
tie:                 1
```

Interpretation:

- NoiseFilter achieved a real but modest answer-level gain on automatic QA.
- The gain is small, but it is not a one-off artifact: more individual queries
  improved than regressed (`71 > 58`).
- Exact match remains zero for both variants, so token-level overlap is still
  the more useful automatic signal in this setup.

## Pairwise Answer Evaluation

Pairwise LLM-judge results on 130 hybrid answers:

```text
Overall Winner
  baseline:    63
  noisefilter: 55
  tie:         12
```

Criterion breakdown:

```text
Comprehensiveness
  baseline:    60
  noisefilter: 55
  tie:         15

Diversity
  baseline:    50
  noisefilter: 50
  tie:         30

Empowerment
  baseline:    64
  noisefilter: 57
  tie:          9
```

Interpretation:

- Despite the automatic QA gain, pairwise judging still slightly prefers
  baseline overall.
- The gap is concentrated in `Comprehensiveness` and `Empowerment`, not in
  `Diversity`.
- This indicates the current NoiseFilter pipeline is helping answer stability
  and overlap quality, but is still giving up some answer richness or
  explanatory breadth.

## Stability Signal

Output-length statistics provide an important secondary signal:

```text
baseline
  avg chars:    2520.3
  median chars: 2366.0
  min chars:       4
  max chars:   31696

noisefilter
  avg chars:    2236.3
  median chars: 2353.5
  min chars:     357
  max chars:    4729
```

Observed anomalies:

- baseline produced a literal `None` answer on `query_id=24`
- baseline produced a `31,696`-character runaway answer on `query_id=78`
- noisefilter produced neither an empty answer nor an extreme runaway answer in
  this round

This is the clearest positive behavioral signal of the new pipeline:

```text
NoiseFilter substantially improved output stability.
```

That stability likely explains why automatic token-F1 improved even though
pairwise human-style preference did not.

## Main Judgment

The current stage-2 `p1` result supports the following conservative conclusion:

```text
NoiseFilter now produces a small but real answer-level gain on automatic QA and
clearly improves output stability, but it still slightly underperforms baseline
on pairwise judged comprehensiveness and empowerment.
```

Put differently:

- `grounding / stability` improved
- `richness / answer fullness` still lags

This means the project has moved beyond “edge-only improvement” and has started
to influence end-answer quality, but the present version is still better
described as:

```text
more stable, not yet consistently stronger
```

## Why The Metrics Disagree

The disagreement between automatic QA and pairwise judge is itself informative.

The most plausible explanation from this round's evidence is:

1. NoiseFilter reduces runaway or low-quality answer behaviors.
2. NoiseFilter answers are slightly shorter and more constrained.
3. This improves token overlap with short gold answers.
4. But the same constraint can reduce elaboration, context layering, and
   explanatory completeness, which hurts pairwise preference.

Supporting signal:

```text
pairwise baseline-win rows:
  baseline answers are on average 851 characters longer than noisefilter

pairwise noisefilter-win rows:
  noisefilter answers are on average 301 characters longer than baseline
```

This suggests the current bottleneck is no longer pure denoising. It is
recovering grounded richness after denoising.

## Next R&D Direction

The next phase should not undo the current gates. The stability gain is too
valuable to discard. Instead, the follow-up work should target:

1. **Grounded Richness Recovery**
   - keep current anti-noise constraints
   - recover useful detail at query assembly / answer generation time

2. **Shared Retrieval Confusion Audit**
   - identify questions where both baseline and NoiseFilter retrieve the wrong
     topic or wrong document family
   - separate retrieval confusion from graph denoising effects

3. **Graph Composition Audit**
   - explain why `p1` changed node/edge composition without producing strong
     graph shrinkage
   - identify whether pseudo-entity suppression is incomplete or whether merge
     behavior is still expanding the graph through relation-side additions

4. **Targeted Query Error Clusters**
   - review the largest NoiseFilter gains and largest regressions
   - use those samples to guide answer-side improvements instead of only
     graph-side changes

## Immediate Action Items

1. Add a structured `top query delta` audit script for `hybrid_eval.json` and
   `pairwise_hybrid_eval.json`.
2. Inspect shared-failure queries such as `query_id=109` to isolate retrieval
   confusion.
3. Review the largest baseline-only pairwise wins to identify what answer
   richness was lost.
4. Prototype query-side improvements before the next full insert/query rerun:
   - context budgeting
   - answer generation prompt tightening without over-compression
   - grounded expansion after retrieval
