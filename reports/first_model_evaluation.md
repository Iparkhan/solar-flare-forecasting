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

## Event-level alarm episodes

For an operationally interpretable complement to overlapping-window scores, threshold-crossing samples are merged into one alarm episode until a 60-minute quiet gap. An episode is a hit when an M/X catalog flare onset occurs in the next 60 minutes. Each flare is counted once; an alarm episode may cover multiple onsets. Median lead time is measured from episode start to its first matched onset.

| Model | Alarm episodes | Hit episodes | Missed M/X events | Event recall | Episode precision | Median lead (min) |
|---|---:|---:|---:|---:|---:|---:|
| Logistic regression | 387 | 48 | 301 | 0.152 | 0.124 | 22.5 |
| Histogram gradient boosting | 686 | 50 | 298 | 0.161 | 0.073 | 26.5 |

Download: [`tables/event_level_test_2025.csv`](tables/event_level_test_2025.csv). These are retrospective episode scores under the stated one-hour matching rule; they are not a validated operational alert protocol.

## Limitations

This is retrospective classification of overlapping five-minute samples, not an event-independent operational forecast. Results depend on the chosen alert threshold and episode matching rule. Only one solar-cycle year is held out; calibration, uncertainty, alternate alert policies, and simple flux-threshold baselines still need study before scientific or operational claims.
