# First model evaluation

## Evaluation design

- Train: June 2022–2023. Validation: 2024. Untouched test: 2025.
- Rows within 60 minutes of a year boundary are excluded so target horizons do not cross splits.
- The target is an NOAA catalog M/X event start in the next hour. Features contain only past/current X-ray measurements.
- The decision threshold is chosen on validation data to maximize F1 and then frozen for 2025.
- Average precision is compared with the 2025 positive-class prevalence baseline.

## Results on 2025 test period

| Model | AP (baseline shown below) | ROC-AUC | Precision | Recall | F1 | Alert rate |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.104 | 0.763 | 0.132 | 0.187 | 0.155 | 5.49% |
| Histogram gradient boosting | 0.106 | 0.760 | 0.127 | 0.297 | 0.178 | 9.01% |

2025 event-window prevalence baseline AP: **0.039**.

![Precision-recall curves](figures/precision_recall_2025.png)

## Limitations

This is retrospective classification of overlapping 5-minute samples, not event-independent operational forecasting. The one-hour target horizon, threshold, alert burden, and event-level scoring need sensitivity analysis before making scientific or operational claims. The test period is a single solar-cycle year.
