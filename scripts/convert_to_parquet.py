#!/usr/bin/env python3
"""
Convert downloaded CSV files to Parquet format.

Handles multiple CSV column-naming conventions found across different
Kaggle futures datasets and normalises them to a standard schema:

    timestamp | open | high | low | close | volume

Usage
-----
    # Auto-detect all CSVs in data/raw/es/ and convert as 1min bars
    python -m scripts.convert_to_parquet --instrument es

    # Specify custom paths and timeframe
    python -m scripts.convert_to_parquet --instrument nq \\
        --input-dir data/raw/nq --output-dir data/parquet --timeframe 1min

    # Show help
    python -m scripts.convert_to_parquet --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
from data.store import DataStore

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Column-name mappings for auto-detection
# Each dict maps *source* column names (lower-cased) to the canonical
# names used in our system.
# -------------------------------------------------------------------
_COLUMN_MAPS: list[dict[str, str]] = [
    # Standard OHLCV headers
    {
        "timestamp": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    },
    # Date instead of timestamp
    {
        "date": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    },
    # Datetime
    {
        "datetime": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    },
    # TradingView-style
    {
        "time": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    },
    # CQG / some brokers use <DTYYYYMMDD> style
    {
        "<dtyyyymmdd>": "timestamp",
        "<open>": "open",
        "<high>": "high",
        "<low>": "low",
        "<close>": "close",
        "<vol>": "volume",
    },
    # Separate date + time columns (combined later)
    {
        "date": "_date",
        "time": "_time",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    },
]

# Date formats to try in order when pandas' default parser fails
_DATE_FORMATS: list[str] = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%SZ",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%Y%m%d",
    "%Y%m%d %H:%M",
    "%d/%m/%Y %H:%M:%S",
]


def _detect_column_mapping(columns: list[str]) -> dict[str, str] | None:
    """Find the first column map that matches the CSV's columns.

    Returns a mapping from source column name to canonical name, or
    ``None`` if no mapping matched.
    """
    lower_cols = {c.lower().strip(): c for c in columns}

    for cmap in _COLUMN_MAPS:
        if all(src in lower_cols for src in cmap):
            # Build a mapping using the *original* column names from the CSV
            return {lower_cols[src]: dest for src, dest in cmap.items()}

    return None


def _parse_timestamp(series: pd.Series) -> pd.Series:
    """Best-effort timestamp parsing, trying multiple formats."""
    # Try pandas' automatic parser first (handles most ISO formats)
    try:
        return pd.to_datetime(series, infer_datetime_format=True)
    except (ValueError, TypeError):
        pass

    for fmt in _DATE_FORMATS:
        try:
            return pd.to_datetime(series, format=fmt)
        except (ValueError, TypeError):
            continue

    # Last resort: coerce and let NaT propagate
    return pd.to_datetime(series, errors="coerce")


def convert_csv_to_parquet(
    csv_path: Path,
    instrument: str,
    timeframe: str,
    store: DataStore,
) -> int:
    """Read a single CSV, normalise columns, and write to Parquet.

    Returns the number of rows written.
    """
    print(f"  Reading {csv_path.name} ...")
    df = pd.read_csv(csv_path, low_memory=False)

    if df.empty:
        print(f"  [skip] {csv_path.name} is empty.")
        return 0

    # --- Detect column mapping ---
    col_map = _detect_column_mapping(list(df.columns))
    if col_map is None:
        print(
            f"  [error] Unable to auto-detect columns in {csv_path.name}.\n"
            f"          Found columns: {list(df.columns)}\n"
            f"          Skipping file."
        )
        return 0

    df.rename(columns=col_map, inplace=True)

    # Handle separate date + time columns
    if "_date" in df.columns and "_time" in df.columns:
        df["timestamp"] = df["_date"].astype(str) + " " + df["_time"].astype(str)
        df.drop(columns=["_date", "_time"], inplace=True)

    # --- Validate presence of required columns ---
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        print(f"  [error] Missing columns after mapping: {missing}. Skipping.")
        return 0

    # --- Normalise types ---
    df["timestamp"] = _parse_timestamp(df["timestamp"])
    bad_ts = df["timestamp"].isna().sum()
    if bad_ts > 0:
        print(f"  [warn] Dropping {bad_ts} rows with unparseable timestamps.")
        df.dropna(subset=["timestamp"], inplace=True)

    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)

    # Drop rows where OHLC is entirely NaN
    df.dropna(subset=["open", "high", "low", "close"], how="all", inplace=True)

    # Keep only the standard columns
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        print(f"  [skip] No valid rows after cleaning.")
        return 0

    store.save_bars(instrument, timeframe, df)
    print(f"  [ok] {len(df):,} bars written for {instrument.upper()}/{timeframe}")
    return len(df)


def _auto_detect_timeframe(csv_path: Path) -> str:
    """Peek at the first few rows to guess the timeframe from timestamps."""
    try:
        df = pd.read_csv(csv_path, nrows=50)
        col_map = _detect_column_mapping(list(df.columns))
        if col_map is None:
            return "1min"
        df.rename(columns=col_map, inplace=True)

        if "_date" in df.columns and "_time" in df.columns:
            df["timestamp"] = df["_date"].astype(str) + " " + df["_time"].astype(str)

        if "timestamp" not in df.columns:
            return "1min"

        ts = _parse_timestamp(df["timestamp"]).dropna()
        if len(ts) < 2:
            return "1min"

        diffs = ts.diff().dropna()
        median_seconds = diffs.dt.total_seconds().median()

        if median_seconds < 2:
            return "tick"
        elif median_seconds < 120:
            return "1min"
        elif median_seconds < 600:
            return "5min"
        elif median_seconds < 1800:
            return "15min"
        elif median_seconds < 7200:
            return "1h"
        else:
            return "daily"
    except Exception:
        return "1min"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert downloaded CSV files to Parquet format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m scripts.convert_to_parquet --instrument es\n"
            "  python -m scripts.convert_to_parquet --instrument nq --timeframe 1min\n"
            "  python -m scripts.convert_to_parquet --instrument es "
            "--input-dir data/raw/es --output-dir data/parquet\n"
        ),
    )
    parser.add_argument(
        "--instrument",
        type=str,
        required=True,
        help="Instrument symbol (e.g. es, nq, mes, mnq).",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default=None,
        help=(
            "Target timeframe label (e.g. 1min, 5min, daily). "
            "If omitted the script auto-detects from timestamp spacing."
        ),
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help=(
            "Directory containing CSV files. "
            "Defaults to data/raw/{instrument}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/parquet",
        help="Root output directory for Parquet files (default: data/parquet).",
    )
    args = parser.parse_args()

    instrument = args.instrument.strip().lower()

    # Resolve input dir
    if args.input_dir:
        input_dir = Path(args.input_dir)
    else:
        input_dir = Path("data/raw") / instrument
    if not input_dir.is_absolute():
        input_dir = PROJECT_ROOT / input_dir
    input_dir = input_dir.resolve()

    # Resolve output dir
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()

    store = DataStore(base_path=str(output_dir))

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        print("Run the download script first:")
        print(f"  python -m scripts.download_kaggle --instrument {instrument}")
        sys.exit(1)

    print(f"Found {len(csv_files)} CSV file(s) in {input_dir}")
    total_rows = 0

    for csv_path in csv_files:
        timeframe = args.timeframe or _auto_detect_timeframe(csv_path)
        rows = convert_csv_to_parquet(csv_path, instrument, timeframe, store)
        total_rows += rows

    print(f"\nConversion complete. Total rows written: {total_rows:,}")


if __name__ == "__main__":
    main()
