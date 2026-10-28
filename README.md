# Solar Flare Forecasting with NOAA GOES Data

A reproducible physics-data and machine-learning project exploring whether recent solar X-ray measurements contain useful warning signals for upcoming solar flares.

## Project question

Can GOES-18 X-ray time series help estimate the probability of an M-class-or-stronger solar flare beginning in the next 60 minutes? The project compares transparent statistical baselines with machine-learning models and evaluates them on later years that were held out chronologically.

This is an exploratory research portfolio project, not an operational space-weather warning system.

## Visual project summary

See the [visual report](reports/portfolio_summary.md) for the signal-vs-flare plot, flare counts by year, model precision-recall curves, and month-by-month 2025 alert analysis.

## Current results

The first end-to-end run uses science-quality GOES-18 XRS data from June 2022 through 2025 and NOAA's event catalog. It contains 1.86 million one-minute timestamps, 1,867 cataloged M/X events within the measurement interval, and 350,060 five-minute modeling rows after data-quality and feature-window checks.

On the untouched 2025 test year, average precision was 0.104 for logistic regression and 0.106 for histogram gradient boosting, compared with a 0.039 positive-rate baseline. At thresholds selected on 2024 validation data, gradient boosting reached 0.30 recall and 0.13 precision on overlapping time windows. These are preliminary retrospective metrics; they do not establish operational forecast skill. See [`reports/first_model_evaluation.md`](reports/first_model_evaluation.md) and [`reports/initial_data_audit.md`](reports/initial_data_audit.md).

## Methods

- NOAA GOES-18 XRS science-quality, 1-minute averages in two X-ray passbands.
- NOAA GOES Flare Report for event onset and irradiance class.
- Quality-screened observations using product quality flags.
- Target: an M/X event **start** in `(t, t + 60 minutes]`.
- Features use observations at or before `t`; rolling windows require at least 25 valid minutes in the preceding 30 minutes.
- Five-minute sampling reduces highly redundant rows.
- Chronological split: train on 2022–2023, choose thresholds on 2024, report test metrics on 2025. Label horizons are kept inside their split.
- Metrics include average precision, ROC-AUC, precision, recall, F1, alert rate, and a prevalence baseline.

## Quick start on Windows PowerShell

From this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe src\download_yearly_data.py
.\.venv\Scripts\python.exe src\build_features.py
.\.venv\Scripts\python.exe src\train_models.py
.\.venv\Scripts\jupyter-lab.exe notebooks\01_goes_flare_forecasting.ipynb
```

The downloaded raw data and generated processed tables are excluded from Git. `src/download_yearly_data.py` can be rerun safely; existing files are skipped. By default it downloads annual GOES-18 measurements and NOAA flare catalogs for 2022–2025.

## Repository layout

```text
data/
  raw/          Original NOAA files (not committed)
  processed/    Generated modeling tables (not committed)
notebooks/      Exploration and results notebook
reports/        Data audit, model evaluation, figures
src/            Download, feature-building, and modeling scripts
```

## Data sources

- [NOAA NCEI GOES-18 XRS science-quality data](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/xrsf-l2-avg1m_science/)
- [NOAA NCEI GOES Flare Report annual CSVs](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv/)
- [GOES Flare Report data guide](https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES_Flare_Report_ReadMe.pdf)

Exact file names and data caveats are documented in [`data/README.md`](data/README.md).

## Next improvements

1. Evaluate alert quality at event level, grouping adjacent positive windows into alert episodes.
2. Test multiple forecast horizons and flare thresholds without touching the final test year.
3. Add calibration and uncertainty analysis; compare against simple flux-threshold and persistence baselines.
4. Examine detector gaps, solar-cycle drift, and sensitivity to quality-flag policy.
5. Expand the scientific angle with active-region/context data only after the XRS-only baseline is stable.



