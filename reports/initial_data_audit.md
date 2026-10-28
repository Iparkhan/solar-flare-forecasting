# Initial data audit

- GOES observations read: 1,863,360 one-minute timestamps.
- Quality-screened XRS-B values: 1,773,226.
- NOAA catalog M-class-or-stronger events in downloaded years: 1,867.
- Modeling rows after 5-minute sampling and rolling-window checks: 350,060.
- Positive rows (M/X event starts within 60 minutes): 18,538 (5.30%).

## M/X events by year

| Year | M/X events |
|---:|---:|
| 2022 | 203 |
| 2023 | 371 |
| 2024 | 938 |
| 2025 | 355 |

The target is based on NOAA event **start_time**, not peak time: at time t, a row is positive if an M-class-or-stronger event starts in (t, t + 60 minutes]. Features use only measurements at or before t. Only rows with at least 25 valid XRS-B minutes in the preceding 30-minute window are retained.

This is an initial dataset audit. Adjacent time samples are dependent, and the model evaluation must use chronological year-based splits and event-aware interpretation.
