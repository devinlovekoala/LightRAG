# Grounding Signal Evaluation

| Variant | Total | Correct | Wrong | Ambiguous | Strict Precision | Lenient Precision | NLI Enabled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 100 | 15 | 50 | 35 | 0.1500 | 0.5000 | False |
| noisefilter | 100 | 22 | 43 | 35 | 0.2200 | 0.5700 | False |

Strict precision delta (noisefilter - baseline): 0.0700
Lenient precision delta (noisefilter - baseline): 0.0700

## baseline

| Signal | Correct Mean | Wrong Mean | Ambiguous Mean | ROC AUC | Average Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| src_anchor_score | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 0.6448 |
| dst_anchor_score | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 0.6448 |
| endpoint_anchor_score | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 0.6448 |
| cooccurrence_score | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 0.6448 |
| nli_support_score | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.6448 |

## noisefilter

| Signal | Correct Mean | Wrong Mean | Ambiguous Mean | ROC AUC | Average Precision |
| --- | ---: | ---: | ---: | ---: | ---: |
| src_anchor_score | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 0.5560 |
| dst_anchor_score | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 0.5560 |
| endpoint_anchor_score | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 0.5560 |
| cooccurrence_score | 1.0000 | 1.0000 | 1.0000 | 0.5000 | 0.5560 |
| nli_support_score | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.5560 |
