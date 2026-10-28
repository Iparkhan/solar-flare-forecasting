# Data directory

Raw and processed data are intentionally excluded from Git.

## Data used in the first analysis

- GOES-18 XRS science-quality 1-minute averages for 2022–2025, NOAA NCEI, product `xrsf-l2-avg1m_science`. GOES-18 measurements begin in June 2022. Annual original NetCDF files are stored in `raw/goes18_yearly/`.
- NOAA GOES Flare Report annual CSV files for 2022–2025, stored in `raw/flare_report/`.
- October 2024 daily GOES-18 files are also kept in `raw/goes18_daily/` for the first visual comparison of time series and event onsets.

The downloader `src/download_yearly_data.py` fetches annual measurement files and catalogs from these official NOAA directories:

- https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/xrsf-l2-avg1m_science/
- https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv/

Documentation: https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes16/l2/docs/GOES_Flare_Report_ReadMe.pdf

The event catalog is event-based (one row per flare, indexed by peak time); labels here use the catalog's `start_time`. GOES XRS measurements are a one-minute time series in two passbands. Fluxes are in W/m². The first feature-building script retains only records where both XRS-A and XRS-B quality flags equal `0` (the product's `good_data` value) and requires at least 25 valid XRS-B minutes in a trailing 30-minute feature window.

The prediction target is an M-class-or-stronger flare onset in the following 60 minutes. The class threshold is XRS-B irradiance of at least `1e-5 W/m²`. These are retrospective labels; the preliminary evaluation uses chronological year splits and does not yet measure distinct alert episodes or event-level warning lead time.
