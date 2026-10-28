"""Build quality-screened, leakage-aware GOES-18 flare-forecasting features."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
HORIZON_NS = 60 * 60 * 1_000_000_000
SAMPLE_EVERY = 5  # minutes; reduces highly redundant adjacent examples
M_CLASS_THRESHOLD = 1e-5  # W/m^2, NOAA GOES XRS-B definition


def load_measurements() -> pd.DataFrame:
    pieces = []
    for path in sorted((RAW / "goes18_yearly").glob("*.nc")):
        with xr.open_dataset(path) as ds:
            part = ds[["xrsb_flux", "xrsa_flux", "xrsb_flag", "xrsa_flag"]].to_dataframe().reset_index()
        part["source_file"] = path.name
        pieces.append(part)
        print(f"Read {path.name}: {len(part):,} minutes", flush=True)
    if not pieces:
        raise FileNotFoundError("No yearly NetCDF files found. Run src/download_yearly_data.py first.")
    frame = pd.concat(pieces, ignore_index=True).sort_values("time").drop_duplicates("time")
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    valid = (frame["xrsb_flag"] == 0) & (frame["xrsa_flag"] == 0)
    for col in ["xrsb_flux", "xrsa_flux"]:
        frame.loc[~valid | (frame[col] <= 0), col] = np.nan
    frame = frame.set_index("time")
    return frame


def load_events() -> pd.DataFrame:
    paths = sorted((RAW / "flare_report").glob("sci_xrsf-l2-flrpt_geo_y*.csv"))
    if not paths:
        raise FileNotFoundError("No NOAA flare catalog found. Run src/download_yearly_data.py first.")
    events = pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)
    events["start_time"] = pd.to_datetime(events["start_time"], utc=True, errors="coerce")
    events["xrsb_irrad"] = pd.to_numeric(events["xrsb_irrad"], errors="coerce")
    events = events.dropna(subset=["start_time", "xrsb_irrad"])
    events = events[events["xrsb_irrad"] >= M_CLASS_THRESHOLD].sort_values("start_time")
    return events


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    measurements = load_measurements()
    events = load_events()
    b = measurements["xrsb_flux"]
    a = measurements["xrsa_flux"]
    log_b, log_a = np.log10(b), np.log10(a)

    features = pd.DataFrame(index=measurements.index)
    features["log10_xrsb_now"] = log_b
    features["log10_xrsa_now"] = log_a
    features["log10_hardness_ratio"] = log_a - log_b
    features["log10_xrsb_median_10m"] = log_b.rolling("10min", min_periods=8).median()
    features["log10_xrsb_max_30m"] = log_b.rolling("30min", min_periods=25).max()
    features["log10_xrsb_std_30m"] = log_b.rolling("30min", min_periods=25).std()
    features["log10_xrsb_change_10m"] = log_b - log_b.shift(10)
    features["valid_minutes_30m"] = b.rolling("30min", min_periods=1).count()

    sampled = features.iloc[::SAMPLE_EVERY].copy()
    times = sampled.index
    event_starts = events["start_time"].astype("int64").to_numpy()
    t_ns = times.astype("int64").to_numpy()
    next_event_index = np.searchsorted(event_starts, t_ns, side="right")
    has_next = next_event_index < len(event_starts)
    target = np.zeros(len(times), dtype=np.int8)
    target[has_next] = (event_starts[next_event_index[has_next]] <= t_ns[has_next] + HORIZON_NS).astype(np.int8)
    sampled["target_m_or_x_start_next_60m"] = target
    sampled = sampled[sampled["valid_minutes_30m"] >= 25].dropna()
    sampled.index.name = "time_utc"
    sampled.to_csv(OUT / "goes18_forecast_features.csv")

    event_counts = events.assign(year=events["start_time"].dt.year).groupby("year").size()
    summary = [
        "# Initial data audit",
        "",
        f"- GOES observations read: {len(measurements):,} one-minute timestamps.",
        f"- Quality-screened XRS-B values: {measurements['xrsb_flux'].notna().sum():,}.",
        f"- NOAA catalog M-class-or-stronger events in downloaded years: {len(events):,}.",
        f"- Modeling rows after 5-minute sampling and rolling-window checks: {len(sampled):,}.",
        f"- Positive rows (M/X event starts within 60 minutes): {int(sampled['target_m_or_x_start_next_60m'].sum()):,} ({sampled['target_m_or_x_start_next_60m'].mean():.2%}).",
        "",
        "## M/X events by year",
        "",
        "| Year | M/X events |",
        "|---:|---:|",
    ]
    summary += [f"| {year} | {count:,} |" for year, count in event_counts.items()]
    summary += [
        "",
        "The target is based on NOAA event **start_time**, not peak time: at time t, a row is positive if an M-class-or-stronger event starts in (t, t + 60 minutes]. Features use only measurements at or before t. Only rows with at least 25 valid XRS-B minutes in the preceding 30-minute window are retained.",
        "",
        "This is an initial dataset audit. Adjacent time samples are dependent, and the model evaluation must use chronological year-based splits and event-aware interpretation.",
    ]
    (ROOT / "reports" / "initial_data_audit.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary), flush=True)
    print(f"Saved {OUT / 'goes18_forecast_features.csv'}", flush=True)


if __name__ == "__main__":
    main()

