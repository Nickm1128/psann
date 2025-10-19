#!/usr/bin/env python
"""Download the trimmed portfolio price series used by HISSO benchmarks.

The default configuration mirrors the repository's example dataset:
- Ticker: AAPL (Apple Inc.)
- Range: 2015-02-17 through 2017-02-16 (inclusive)
- Columns: date, open, close
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import StringIO
from pathlib import Path
from typing import Iterable, List
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


@dataclass
class DownloadConfig:
    symbol: str
    start: datetime
    end: datetime
    interval: str = "1d"

    @property
    def period1(self) -> int:
        return int(self.start.replace(tzinfo=timezone.utc).timestamp())

    @property
    def period2(self) -> int:
        # Yahoo treats period2 as an exclusive bound; bump by one day.
        exclusive_end = self.end + timedelta(days=1)
        return int(exclusive_end.replace(tzinfo=timezone.utc).timestamp())


def _download_csv(config: DownloadConfig) -> str:
    params = {
        "period1": config.period1,
        "period2": config.period2,
        "interval": config.interval,
        "events": "history",
        "includeAdjustedClose": "false",
    }
    query = urlencode(params)
    url = f"https://query1.finance.yahoo.com/v7/finance/download/{config.symbol}?{query}"
    try:
        with urlopen(url, timeout=30) as response:
            payload = response.read().decode("utf-8")
    except (HTTPError, URLError) as exc:
        raise SystemExit(
            f"Failed to download data for {config.symbol}: {exc}. "
            "You may need to rerun with a shorter range or try again later."
        ) from exc
    if "Date,Open,High,Low,Close" not in payload:
        raise SystemExit("Unexpected payload received from Yahoo Finance; verify the symbol/range.")
    return payload


def _trim_columns(csv_text: str) -> List[List[str]]:
    reader = csv.DictReader(StringIO(csv_text))
    rows: List[List[str]] = []
    for row in reader:
        if not row:
            continue
        date = row.get("Date")
        open_price = row.get("Open")
        close_price = row.get("Close")
        if not (date and open_price and close_price):
            continue
        rows.append([date, open_price, close_price])
    return rows


def write_dataset(rows: Iterable[Iterable[str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "open", "close"])
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the HISSO benchmark price dataset from Yahoo Finance."
    )
    parser.add_argument("--symbol", default="AAPL", help="Ticker symbol to download.")
    parser.add_argument(
        "--start",
        default="2015-02-17",
        help="Start date (inclusive, YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end",
        default="2017-02-16",
        help="End date (inclusive, YYYY-MM-DD).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmarks") / "hisso_portfolio_prices.csv",
        help="Destination CSV path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.out.exists() and not args.overwrite:
        raise SystemExit(f"{args.out} already exists. Pass --overwrite to replace it.")

    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    if end_dt < start_dt:
        raise SystemExit("End date must be on or after the start date.")

    config = DownloadConfig(symbol=args.symbol, start=start_dt, end=end_dt)
    csv_text = _download_csv(config)
    trimmed_rows = _trim_columns(csv_text)
    if not trimmed_rows:
        raise SystemExit("No rows were parsed from the downloaded dataset.")
    write_dataset(trimmed_rows, args.out)
    print(f"Wrote {len(trimmed_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
