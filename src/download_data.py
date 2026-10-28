"""Download daily science-quality GOES-18 XRS files and NOAA annual flare reports."""
from __future__ import annotations

import argparse
from datetime import date, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
GOES_BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/goes18/l2/data/xrsf-l2-avg1m_science"
FLARE_BASE = "https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes/multi/l2/data/xrsf-l2-flrpt_science/csv"
VERSION = "v2-2-2"


def download(url: str, destination: Path) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already downloaded: {destination.name}")
        return True
    try:
        with urlopen(url, timeout=60) as response, destination.open("wb") as output:
            output.write(response.read())
        print(f"Downloaded: {destination.name} ({destination.stat().st_size:,} bytes)")
        return True
    except (HTTPError, URLError, TimeoutError) as exc:
        if destination.exists():
            destination.unlink()
        print(f"Unavailable: {url} ({exc})")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, default=date(2024, 10, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 10, 31))
    args = parser.parse_args()
    if args.end < args.start:
        parser.error("--end must not be earlier than --start")

    raw = ROOT / "data" / "raw"
    day, days_ok, days_missing = args.start, 0, 0
    while day <= args.end:
        stamp = day.strftime("%Y%m%d")
        filename = f"sci_xrsf-l2-avg1m_g18_d{stamp}_{VERSION}.nc"
        url = f"{GOES_BASE}/{day:%Y/%m}/{filename}"
        target = raw / "goes18_daily" / str(day.year) / f"{day.month:02d}" / filename
        if download(url, target):
            days_ok += 1
        else:
            days_missing += 1
        day += timedelta(days=1)

    for year in range(args.start.year, args.end.year + 1):
        filename = f"sci_xrsf-l2-flrpt_geo_y{year}_v1-0-1.csv"
        url = f"{FLARE_BASE}/{filename}"
        target = raw / "flare_report" / filename
        download(url, target)

    print(f"Daily GOES files available: {days_ok}; unavailable: {days_missing}")
    if days_ok == 0:
        raise SystemExit("No daily GOES files were downloaded. Check the source URL or connection.")


if __name__ == "__main__":
    main()
