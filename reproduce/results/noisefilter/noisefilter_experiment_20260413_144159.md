# NoiseFilter-RAG Experiment Report

## Configuration

```json
{
  "noise_ratios": [
    0.1,
    0.2,
    0.3
  ],
  "thresholds": [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5
  ],
  "num_nodes": 24,
  "clean_degree": 3,
  "top_k": 5,
  "seed": 7,
  "injectors": [
    "random",
    "contradictory"
  ]
}
```

## Best Hard Filter by F1

| Setting | Threshold | Filter F1 | Retrieval Noise@K |
| --- | ---: | ---: | ---: |
| random@0.10 | 0.1 | 1.000 | 0.000 |
| random@0.20 | 0.1 | 1.000 | 0.000 |
| random@0.30 | 0.1 | 1.000 | 0.000 |
| contradictory@0.10 | 0.1 | 1.000 | 0.000 |
| contradictory@0.20 | 0.1 | 1.000 | 0.000 |
| contradictory@0.30 | 0.1 | 1.000 | 0.000 |

## Best Retrieval Strategy

| Setting | Strategy | Threshold | Retrieval Noise@K | Returned Edges |
| --- | --- | ---: | ---: | ---: |
| random@0.10 | hard | 0.1 | 0.000 | 3.00 |
| random@0.20 | hard | 0.1 | 0.000 | 3.00 |
| random@0.30 | hard | 0.1 | 0.000 | 3.00 |
| contradictory@0.10 | hard | 0.1 | 0.000 | 3.00 |
| contradictory@0.20 | hard | 0.1 | 0.000 | 3.00 |
| contradictory@0.30 | hard | 0.1 | 0.000 | 3.00 |

## Raw Rows

| Injector | Noise Ratio | Strategy | Threshold | Filter F1 | Retrieval Noise@K | Returned Edges |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| random | 0.10 | baseline | - | 0.000 | 0.089 | 3.29 |
| random | 0.10 | soft | - | 0.000 | 0.089 | 3.29 |
| random | 0.10 | hard | 0.1 | 1.000 | 0.000 | 3.00 |
| random | 0.10 | hard | 0.2 | 1.000 | 0.000 | 3.00 |
| random | 0.10 | hard | 0.3 | 1.000 | 0.000 | 3.00 |
| random | 0.10 | hard | 0.4 | 1.000 | 0.000 | 3.00 |
| random | 0.10 | hard | 0.5 | 1.000 | 0.000 | 3.00 |
| random | 0.20 | baseline | - | 0.000 | 0.157 | 3.46 |
| random | 0.20 | soft | - | 0.000 | 0.133 | 3.46 |
| random | 0.20 | hard | 0.1 | 1.000 | 0.000 | 3.00 |
| random | 0.20 | hard | 0.2 | 1.000 | 0.000 | 3.00 |
| random | 0.20 | hard | 0.3 | 1.000 | 0.000 | 3.00 |
| random | 0.20 | hard | 0.4 | 1.000 | 0.000 | 3.00 |
| random | 0.20 | hard | 0.5 | 1.000 | 0.000 | 3.00 |
| random | 0.30 | baseline | - | 0.000 | 0.222 | 3.75 |
| random | 0.30 | soft | - | 0.000 | 0.200 | 3.75 |
| random | 0.30 | hard | 0.1 | 1.000 | 0.000 | 3.00 |
| random | 0.30 | hard | 0.2 | 1.000 | 0.000 | 3.00 |
| random | 0.30 | hard | 0.3 | 1.000 | 0.000 | 3.00 |
| random | 0.30 | hard | 0.4 | 1.000 | 0.000 | 3.00 |
| random | 0.30 | hard | 0.5 | 1.000 | 0.000 | 3.00 |
| contradictory | 0.10 | baseline | - | 0.000 | 0.089 | 3.29 |
| contradictory | 0.10 | soft | - | 0.000 | 0.089 | 3.29 |
| contradictory | 0.10 | hard | 0.1 | 1.000 | 0.000 | 3.00 |
| contradictory | 0.10 | hard | 0.2 | 1.000 | 0.000 | 3.00 |
| contradictory | 0.10 | hard | 0.3 | 1.000 | 0.000 | 3.00 |
| contradictory | 0.10 | hard | 0.4 | 1.000 | 0.000 | 3.00 |
| contradictory | 0.10 | hard | 0.5 | 1.000 | 0.000 | 3.00 |
| contradictory | 0.20 | baseline | - | 0.000 | 0.163 | 3.58 |
| contradictory | 0.20 | soft | - | 0.000 | 0.163 | 3.58 |
| contradictory | 0.20 | hard | 0.1 | 1.000 | 0.000 | 3.00 |
| contradictory | 0.20 | hard | 0.2 | 1.000 | 0.000 | 3.00 |
| contradictory | 0.20 | hard | 0.3 | 1.000 | 0.000 | 3.00 |
| contradictory | 0.20 | hard | 0.4 | 1.000 | 0.000 | 3.00 |
| contradictory | 0.20 | hard | 0.5 | 1.000 | 0.000 | 3.00 |
| contradictory | 0.30 | baseline | - | 0.000 | 0.228 | 3.83 |
| contradictory | 0.30 | soft | - | 0.000 | 0.217 | 3.83 |
| contradictory | 0.30 | hard | 0.1 | 1.000 | 0.000 | 3.00 |
| contradictory | 0.30 | hard | 0.2 | 1.000 | 0.000 | 3.00 |
| contradictory | 0.30 | hard | 0.3 | 1.000 | 0.000 | 3.00 |
| contradictory | 0.30 | hard | 0.4 | 1.000 | 0.000 | 3.00 |
| contradictory | 0.30 | hard | 0.5 | 1.000 | 0.000 | 3.00 |
