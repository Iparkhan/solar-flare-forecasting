"""Create portfolio-ready plots and tables from the audited NOAA data."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"
FIG = REPORTS / "figures"
TABLES = REPORTS / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

# 1. A real ten-day X-ray time series with NOAA M/X flare onset annotations.
flux_parts = []
for path in sorted((RAW / "goes18_daily" / "2024" / "10").glob("*.nc")):
    with xr.open_dataset(path) as ds:
        part = ds[["xrsb_flux", "xrsb_flag"]].to_dataframe().reset_index()
    part["time"] = pd.to_datetime(part["time"], utc=True)
    part.loc[part["xrsb_flag"] != 0, "xrsb_flux"] = np.nan
    flux_parts.append(part[["time", "xrsb_flux"]])
if not flux_parts:
    raise FileNotFoundError("October 2024 daily sample missing; run src/download_data.py first.")
flux = pd.concat(flux_parts, ignore_index=True).drop_duplicates("time").sort_values("time")
start, end = pd.Timestamp("2024-10-01", tz="UTC"), pd.Timestamp("2024-10-11", tz="UTC")
flux = flux[(flux.time >= start) & (flux.time < end)]
events = pd.read_csv(RAW / "flare_report" / "sci_xrsf-l2-flrpt_geo_y2024_v1-0-1.csv")
events["start_time"] = pd.to_datetime(events["start_time"], utc=True, errors="coerce")
events["xrsb_irrad"] = pd.to_numeric(events["xrsb_irrad"], errors="coerce")
events = events[(events.start_time >= start) & (events.start_time < end) & (events.xrsb_irrad >= 1e-5)].copy()
fig, ax = plt.subplots(figsize=(13, 5.5))
ax.plot(flux.time, flux.xrsb_flux, color="#245a9a", lw=0.85, label="GOES-18 XRS-B (quality-screened)")
ax.set_yscale("log")
ax.set_ylabel("X-ray irradiance (W/m²)")
ax.set_xlabel("Time (UTC)")
ax.set_title("GOES-18 solar X-ray flux with M/X flare starts · 1–10 Oct 2024")
for row in events.itertuples():
    is_x = row.xrsb_irrad >= 1e-4
    ax.axvline(row.start_time, color="#d34a3a" if is_x else "#e69f00", alpha=0.45 if is_x else 0.2, lw=1.0 if is_x else 0.7)
    if is_x:
        ax.annotate(row.flare_class, (row.start_time, max(flux.xrsb_flux.dropna()) * 0.72), rotation=90,
                    va="top", ha="right", fontsize=8, color="#b33128")
ax.plot([], [], color="#e69f00", lw=1.5, label="M-class flare start")
ax.plot([], [], color="#d34a3a", lw=1.5, label="X-class flare start")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1, tz=mdates.UTC))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d", tz=mdates.UTC))
ax.grid(which="both", alpha=0.2)
ax.legend(loc="upper left", ncol=3, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "goes18_xray_and_flare_onsets_oct2024.png", dpi=180)
plt.close(fig)

# 2. Catalog event counts by year, restricting 2022 to the GOES-18 coverage interval.
year_frames = []
for path in sorted(RAW.joinpath("flare_report").glob("sci_xrsf-l2-flrpt_geo_y*.csv")):
    year = int(path.name.split("_y")[1][:4])
    d = pd.read_csv(path)
    d["start_time"] = pd.to_datetime(d["start_time"], utc=True, errors="coerce")
    d["xrsb_irrad"] = pd.to_numeric(d["xrsb_irrad"], errors="coerce")
    if year == 2022:
        d = d[d.start_time >= pd.Timestamp("2022-06-17", tz="UTC")]
    d = d.dropna(subset=["start_time", "xrsb_irrad"])
    d["year"] = year
    d["class_group"] = np.select(
        [d.xrsb_irrad >= 1e-4, d.xrsb_irrad >= 1e-5, d.xrsb_irrad >= 1e-6],
        ["X", "M", "C"], default="B")
    year_frames.append(d)
catalog = pd.concat(year_frames, ignore_index=True)
counts = catalog.groupby(["year", "class_group"]).size().unstack(fill_value=0).reindex(columns=["C", "M", "X"], fill_value=0)
counts.to_csv(TABLES / "flare_counts_by_year.csv")
fig, ax = plt.subplots(figsize=(9, 5))
counts.plot(kind="bar", ax=ax, color=["#56b4e9", "#e69f00", "#d34a3a"], width=0.78)
ax.set_title("GOES-18 era: cataloged solar flares by year and class")
ax.set_xlabel("Year (2022 begins 17 June; later years are complete)")
ax.set_ylabel("Number of flares")
ax.tick_params(axis="x", rotation=0)
ax.grid(axis="y", alpha=0.22)
ax.legend(title="GOES class group")
fig.tight_layout()
fig.savefig(FIG / "flare_counts_by_year.png", dpi=180)
plt.close(fig)

# 3. Month-by-month 2025 test-set alert burden and precision using the frozen validation threshold.
pred = pd.read_csv(ROOT / "data" / "processed" / "test_predictions_2025.csv", parse_dates=["time_utc"]).set_index("time_utc")
y = pred["observed_m_or_x_start_next_60m"].astype(int)
model_col = "histogram_gradient_boosting"
with (REPORTS / "model_evaluation.json").open(encoding="utf-8") as f:
    evaluation = json.load(f)
threshold = evaluation["models"]["Histogram gradient boosting"]["test_2025"]["threshold"]
yhat = (pred[model_col] >= threshold).astype(int)
pred["month"] = pred.index.month
pred["actual_positive"] = y
pred["alert"] = yhat
monthly = pred.groupby("month").agg(
    rows=("actual_positive", "size"),
    positive_windows=("actual_positive", "sum"),
    alerts=("alert", "sum"),
)
monthly["positive_rate"] = monthly.positive_windows / monthly.rows
monthly["alert_rate"] = monthly.alerts / monthly.rows
monthly["precision"] = [
    float(y[(pred.month == m) & (yhat == 1)].mean()) if int(((pred.month == m) & (yhat == 1)).sum()) else 0.0
    for m in monthly.index
]
monthly["recall"] = [
    float(y[(pred.month == m) & (y == 1) & (yhat == 1)].sum() / max(int(((pred.month == m) & (y == 1)).sum()), 1))
    for m in monthly.index
]
monthly.index.name = "month"
monthly.to_csv(TABLES / "monthly_test_metrics_2025.csv", float_format="%.5f")
fig, ax = plt.subplots(figsize=(11, 5.5))
months = monthly.index.to_numpy()
width = 0.36
ax.bar(months - width / 2, monthly.positive_rate * 100, width, color="#8ab6d6", label="Observed positive windows")
ax.bar(months + width / 2, monthly.alert_rate * 100, width, color="#ef9b62", label="Model alert rate")
ax.set_xticks(months, [pd.Timestamp(2025, m, 1).strftime("%b") for m in months])
ax.set_ylabel("Share of valid forecast windows (%)")
ax.set_title("2025 holdout: monthly flare-window prevalence and model alerts")
ax.grid(axis="y", alpha=0.22)
ax2 = ax.twinx()
ax2.plot(months, monthly.precision, color="#633b7a", marker="o", lw=2, label="Precision")
ax2.set_ylabel("Precision among alerts")
ax2.set_ylim(0, min(1.0, max(0.3, monthly.precision.max() * 1.4)))
handles, labels = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax.legend(handles + handles2, labels + labels2, loc="upper left", ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "monthly_alert_performance_2025.png", dpi=180)
plt.close(fig)

summary = [
    "# Solar Flare Forecasting: visual project summary",
    "",
    "## The question",
    "",
    "Can recent GOES-18 soft X-ray measurements help estimate whether an M-class-or-stronger flare will begin within the next 60 minutes? This is a retrospective portfolio analysis, not an operational warning system.",
    "",
    "## Dataset and scope",
    "",
    "- Official NOAA science-quality GOES-18 XRS measurements at one-minute cadence, June 2022–December 2025.",
    "- NOAA event catalog used to label flare start times and peak irradiance classes.",
    "- 1,863,360 timestamps; 1,773,226 quality-screened XRS-B values; 350,060 five-minute modeling rows.",
    "- Target: M/X flare start in `(t, t + 60 minutes]`; features use only information at or before `t`.",
    "",
    "## 1. What a flare looks like in the measured signal",
    "",
    "![GOES-18 X-ray flux and flare starts](figures/goes18_xray_and_flare_onsets_oct2024.png)",
    "",
    "This active interval illustrates the scale and timing of real XRS-B measurements. Vertical lines mark cataloged M- and X-class flare starts; X-class events are labeled. It is an example interval, not the model test set.",
    "",
    "## 2. How activity varied across the study years",
    "",
    "![Flare counts by year](figures/flare_counts_by_year.png)",
    "",
    "2022 begins on 17 June because GOES-18 observations start then. Annual counts are descriptive and reflect changing solar activity; they are not directly comparable as rates without accounting for exposure and solar-cycle variation.",
    "",
    "## 3. How the models performed on the untouched 2025 period",
    "",
    "![Precision-recall curves](figures/precision_recall_2025.png)",
    "",
    "Average precision was 0.104 for logistic regression and 0.106 for histogram gradient boosting, versus a 0.039 prevalence baseline. The thresholds were chosen using 2024 validation data. On the overlapping five-minute windows, gradient boosting reached 0.30 recall and 0.13 precision at a 9.01% alert rate.",
    "",
    "## 4. How alert rates changed by month in the holdout year",
    "",
    "![Monthly 2025 alert rates and precision](figures/monthly_alert_performance_2025.png)",
    "",
    "The threshold remains fixed across 2025. Month-to-month variation highlights why a single summary score is not enough for a rare-event forecasting problem.",
    "",
    "## Evaluation design and limitations",
    "",
    "- Train: June 2022–2023; validation: 2024; final test: 2025.",
    "- Year boundaries are kept outside the 60-minute target horizon.",
    "- Features include recent log flux, short-term variability and trend, and the A/B channel hardness ratio.",
    "- Samples overlap in time, so window-level recall is not the percentage of distinct flares successfully forecast.",
    "- False alerts remain frequent; event-level alert grouping, lead-time scoring, calibration, uncertainty, and simple flux-threshold comparisons are next.",
    "",
    "## Detailed artifacts",
    "",
    "- [Initial data audit](initial_data_audit.md)",
    "- [First model evaluation](first_model_evaluation.md)",
    "- [Annual event-count table](tables/flare_counts_by_year.csv)",
    "- [Monthly 2025 test table](tables/monthly_test_metrics_2025.csv)",
]
(REPORTS / "portfolio_summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
print(f"Saved 3 figures and 2 tables under {REPORTS}")
print(monthly.round(3).to_string())
