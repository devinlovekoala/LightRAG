# Labeled Edge Sample Analysis

| Variant | Total | Correct | Wrong | Ambiguous | Strict Precision | Lenient Precision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 100 | 15 | 50 | 35 | 0.1500 | 0.5000 |
| noisefilter | 100 | 22 | 43 | 35 | 0.2200 | 0.5700 |

Strict precision delta (noisefilter - baseline): 0.0700
Lenient precision delta (noisefilter - baseline): 0.0700

## Signal Diagnostics

### baseline

| Metric | Status | Correct Mean | Wrong Mean | Ambiguous Mean | Correct-Wrong |
| --- | --- | ---: | ---: | ---: | ---: |
| conf_score | dead | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| conf_freq_score | dead | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| conf_consistency_score | dead | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| conf_semantic_score | dead | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

### noisefilter

| Metric | Status | Correct Mean | Wrong Mean | Ambiguous Mean | Correct-Wrong |
| --- | --- | ---: | ---: | ---: | ---: |
| conf_score | inverted | 0.4735 | 0.4780 | 0.4728 | -0.0045 |
| conf_freq_score | dead | 0.3333 | 0.3333 | 0.3333 | 0.0000 |
| conf_consistency_score | dead | 1.0000 | 1.0000 | 0.9857 | 0.0000 |
| conf_semantic_score | inverted | 0.0343 | 0.0566 | 0.0522 | -0.0223 |
