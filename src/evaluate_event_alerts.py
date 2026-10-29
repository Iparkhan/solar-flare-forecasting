"""Evaluate thresholded forecast alerts as event-level episodes on the 2025 holdout."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def score_episodes(alert_times: pd.DatetimeIndex, event_times: pd.DatetimeIndex) -> dict[str, float | int]:
    """Merge alerts separated by at most one hour; match each episode forward one hour."""
    event_times = event_times.sort_values()
    if len(alert_times) == 0:
        return {"alarm_episodes": 0, "hit_episodes": 0, "missed_events": len(event_times),
                "event_recall": 0.0, "episode_precision": 0.0, "median_lead_minutes": np.nan}

    alert_ns = alert_times.asi8
    gap_ns = 60 * 60 * 1_000_000_000
    episodes = [int(alert_ns[0])]
    last_alert = int(alert_ns[0])
    for timestamp in alert_ns[1:]:
        timestamp = int(timestamp)
        if timestamp - last_alert > gap_ns:
            episodes.append(timestamp)
        last_alert = timestamp

    detected: set[int] = set()
    leads = []
    hits = 0
    event_ns = event_times.asi8
    hour_ns = gap_ns
    for alarm in episodes:
        alarm_ns = alarm
        lo = np.searchsorted(event_ns, alarm_ns, side="right")
        hi = np.searchsorted(event_ns, alarm_ns + hour_ns, side="right")
        if hi > lo:
            hits += 1
            detected.update(range(lo, hi))
            leads.append((event_ns[lo] - alarm_ns) / 60e9)

    return {
        "alarm_episodes": len(episodes),
        "hit_episodes": hits,
        "missed_events": len(event_times) - len(detected),
        "event_recall": len(detected) / len(event_times) if len(event_times) else 0.0,
        "episode_precision": hits / len(episodes),
        "median_lead_minutes": float(np.median(leads)) if leads else np.nan,
    }


def main() -> None:
    report = ROOT / "reports"
    predictions = pd.read_csv(ROOT / "data" / "processed" / "test_predictions_2025.csv",
                              parse_dates=["time_utc"]).set_index("time_utc")
    evaluation = json.loads((report / "model_evaluation.json").read_text(encoding="utf-8"))
    event_files = sorted((ROOT / "data" / "raw" / "flare_report").glob("sci_xrsf-l2-flrpt_geo_y*.csv"))
    if not event_files:
        raise FileNotFoundError("NOAA event catalog missing; run src/download_yearly_data.py first.")
    catalog = pd.concat([pd.read_csv(path) for path in event_files], ignore_index=True)
    catalog["start_time"] = pd.to_datetime(catalog["start_time"], utc=True, errors="coerce")
    catalog["xrsb_irrad"] = pd.to_numeric(catalog["xrsb_irrad"], errors="coerce")
    events = catalog.loc[
        catalog["start_time"].between("2025-01-01", "2025-12-31 23:59:59.999999", inclusive="both")
        & (catalog["xrsb_irrad"] >= 1e-5), "start_time"
    ].dropna().sort_values()
    rows = []
    for model, metrics in evaluation["models"].items():
        col = model.lower().replace(" ", "_")
        alert_times = predictions.index[predictions[col] >= metrics["test_2025"]["threshold"]]
        row = {"model": model, **score_episodes(alert_times, pd.DatetimeIndex(events))}
        rows.append(row)
    table = pd.DataFrame(rows)
    (report / "tables").mkdir(exist_ok=True)
    table.to_csv(report / "tables" / "event_level_test_2025.csv", index=False, float_format="%.4f")
    lines = [
        "", "## Event-level alarm episodes", "",
        "To complement overlapping-window scores, threshold-crossing samples are merged into one alarm episode until a 60-minute quiet gap. An episode is a hit when an M/X catalog flare onset occurs in the next 60 minutes. Each flare is counted once; an alarm episode may cover multiple onsets. Median lead time is measured from episode start to its first matched onset.", "",
        "| Model | Alarm episodes | Hit episodes | Missed M/X events | Event recall | Episode precision | Median lead (min) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['model']} | {row['alarm_episodes']} | {row['hit_episodes']} | {row['missed_events']} | {row['event_recall']:.3f} | {row['episode_precision']:.3f} | {row['median_lead_minutes']:.1f} |")
    lines += ["", "Download: [`tables/event_level_test_2025.csv`](tables/event_level_test_2025.csv). These are retrospective episode scores under the stated one-hour matching rule; they are not a validated operational alert protocol.",
              "", "## Limitations", "",
              "This is retrospective classification of overlapping five-minute samples, not an event-independent operational forecast. The single-feature XRS-B baseline slightly underperforms the ML models on average precision but detects more events under this episode rule. The apparent AP difference is small and the 12-month bootstrap intervals overlap. Results depend on the alert threshold and episode matching rule; calibration, further years, alternate alert policies, and operational validation remain open.", ""]
    path = report / "first_model_evaluation.md"
    text = path.read_text(encoding="utf-8")
    section_starts = [position for heading in ("## Event-level alarm episodes", "## Limitations")
                      if (position := text.find(heading)) >= 0]
    prefix = text[:min(section_starts)] if section_starts else text.rstrip() + "\n\n"
    text = prefix.rstrip() + "\n\n" + "\n".join(lines).lstrip()
    path.write_text(text, encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
