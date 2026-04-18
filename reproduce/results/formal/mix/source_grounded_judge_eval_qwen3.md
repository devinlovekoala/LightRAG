# Source-Grounded Judge Evaluation

| Variant | Total | Correct | Wrong | Ambiguous | Strict Precision | Lenient Precision | ROC AUC | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 100 | 15 | 50 | 35 | 0.1500 | 0.5000 | 0.5413 | 0.5488 |
| noisefilter | 100 | 22 | 43 | 35 | 0.2200 | 0.5700 | 0.5497 | 0.4627 |

Strict precision delta (noisefilter - baseline): 0.0700
Lenient precision delta (noisefilter - baseline): 0.0700
Judge ROC AUC delta: 0.0083
Judge AP delta: -0.0861
