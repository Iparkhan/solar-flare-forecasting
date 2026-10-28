"""Download annual science-quality GOES-18 XRS data and NOAA flare catalogs."""
from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
GOES_BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/xrsf-l2-avg1m_science"
FLARE_BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv"


def download(url: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size:
        print(f"Already downloaded: {destination.name}", flush=True)
        return True
    try:
        with urlopen(url, timeout=120) as response, destination.open("wb") as output:
            output.write(response.read())
        print(f"Downloaded: {destination.name} ({destination.stat().st_size:,} bytes)", flush=True)
        return True
    except (HTTPError, URLError, TimeoutError) as exc:
        if destination.exists():
            destination.unlink()
        print(f"Unavailable: {url} ({exc})", flush=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-year", type=int, default=2022)
    parser.add_argument("--last-year", type=int, default=2025)
    args = parser.parse_args()
    if args.last_year < args.first_year:
        parser.error("--last-year must be >= --first-year")

    raw = ROOT / "data" / "raw"
    successes = 0
    for year in range(args.first_year, args.last_year + 1):
        goes_name = f"sci_xrsf-l2-avg1m_g18_y{year}_v2-2-2.nc"
        goes_url = f"{GOES_BASE}/{goes_name}"
        goes_path = raw / "goes18_yearly" / goes_name
        successes += download(goes_url, goes_path)

        flare_name = f"sci_xrsf-l2-flrpt_geo_y{year}_v1-0-1.csv"
        flare_url = f"{FLARE_BASE}/{flare_name}"
        flare_path = raw / "flare_report" / flare_name
        download(flare_url, flare_path)

    if successes == 0:
        raise SystemExit("No annual GOES files available; check the NOAA source or requested years.")


if __name__ == "__main__":
    main()
