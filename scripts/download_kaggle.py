#!/usr/bin/env python3
"""
Download futures / orderbook data from Kaggle.

Attempts the Kaggle CLI first, then falls back to ``requests`` with a
``tqdm`` progress bar for a direct HTTP download.

Usage
-----
    # List all available datasets
    python -m scripts.download_kaggle --list

    # Download a specific dataset
    python -m scripts.download_kaggle --dataset es

    # Download all futures OHLCV datasets
    python -m scripts.download_kaggle --dataset all-futures

    # Download all LOB datasets
    python -m scripts.download_kaggle --dataset all-lob

    # Download everything
    python -m scripts.download_kaggle --dataset all

    # Show help
    python -m scripts.download_kaggle --help
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


def _load_kaggle_credentials() -> tuple[str, str] | None:
    """Load Kaggle credentials from .env or ~/.kaggle/kaggle.json.

    Returns (username, key) or None if not found.
    """
    # 1. Try .env file in project root
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        username = None
        key = None
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            if k == "KAGGLE_USERNAME":
                username = v
            elif k == "KAGGLE_KEY":
                key = v
        if username and key and username != "your_kaggle_username":
            return username, key

    # 2. Try environment variables
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        return username, key

    # 3. Try ~/.kaggle/kaggle.json
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        import json
        data = json.loads(kaggle_json.read_text())
        if "username" in data and "key" in data:
            return data["username"], data["key"]

    return None

# -------------------------------------------------------------------
# All Kaggle datasets organised by category
# -------------------------------------------------------------------

KAGGLE_DATASETS: dict[str, dict] = {
    # ===== Futures OHLCV =====
    "es": {
        "dataset": "jorijnsmit/emini-sp-500-futures",
        "description": "E-mini S&P 500 (ES) minute bars",
        "category": "futures",
        "manual_url": "https://www.kaggle.com/datasets/jorijnsmit/emini-sp-500-futures",
    },
    "es-cme": {
        "dataset": "choweric/cme-es",
        "description": "E-mini S&P 500 (ES) 2000-2022 minute OHLCV",
        "category": "futures",
        "manual_url": "https://www.kaggle.com/datasets/choweric/cme-es",
    },
    "nq": {
        "dataset": "jorijnsmit/emini-nasdaq-100-futures",
        "description": "E-mini Nasdaq 100 (NQ) minute bars",
        "category": "futures",
        "manual_url": "https://www.kaggle.com/datasets/jorijnsmit/emini-nasdaq-100-futures",
    },
    "nq-cme": {
        "dataset": "youneseloiarm/nasdaq-cme-future-nq",
        "description": "E-mini Nasdaq 100 (NQ) CME minute OHLCV",
        "category": "futures",
        "manual_url": "https://www.kaggle.com/datasets/youneseloiarm/nasdaq-cme-future-nq",
    },
    "sp-tick": {
        "dataset": "finnhub/sp-500-futures-tick-data-sp",
        "description": "S&P 500 Futures tick data (Finnhub)",
        "category": "futures",
        "manual_url": "https://www.kaggle.com/datasets/finnhub/sp-500-futures-tick-data-sp",
    },
    "sp-nq-5yr": {
        "dataset": "salaheddineelkhirani/5-year-data-for-s-and-p-500-and-nasdaq-100",
        "description": "S&P 500 + Nasdaq 100 -- 5 year daily/minute bars",
        "category": "futures",
        "manual_url": "https://www.kaggle.com/datasets/salaheddineelkhirani/5-year-data-for-s-and-p-500-and-nasdaq-100",
    },

    # ===== Limit Order Book (LOB) =====
    "lob": {
        "dataset": "praanj/limit-orderbook-data",
        "description": "Limit Order Book (L2) data",
        "category": "lob",
        "manual_url": "https://www.kaggle.com/datasets/praanj/limit-orderbook-data",
    },
    "crypto-lob": {
        "dataset": "martinsn/high-frequency-crypto-limit-order-book-data",
        "description": "High-frequency crypto L2 order book",
        "category": "lob",
        "manual_url": "https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data",
    },
    "btc-lob": {
        "dataset": "siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data",
        "description": "BTC Perpetual (BTCUSDT.P) L2 order book",
        "category": "lob",
        "manual_url": "https://www.kaggle.com/datasets/siavashraz/bitcoin-perpetualbtcusdtp-limit-order-book-data",
    },
    "trade-orderbook": {
        "dataset": "farhankardan/trade-and-order-book",
        "description": "Trade + Order Book combined dataset",
        "category": "lob",
        "manual_url": "https://www.kaggle.com/datasets/farhankardan/trade-and-order-book",
    },
}

# Group aliases for batch download
DATASET_GROUPS: dict[str, list[str]] = {
    "all-futures": [k for k, v in KAGGLE_DATASETS.items() if v["category"] == "futures"],
    "all-lob": [k for k, v in KAGGLE_DATASETS.items() if v["category"] == "lob"],
    "all": list(KAGGLE_DATASETS.keys()),
}


def _try_kaggle_cli(dataset_slug: str, output_dir: Path) -> bool:
    """Attempt to download via the ``kaggle`` CLI.  Returns True on success."""
    if shutil.which("kaggle") is None:
        return False

    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        dataset_slug,
        "-p",
        str(output_dir),
        "--unzip",
    ]
    print(f"[kaggle-cli] Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[kaggle-cli] Command failed (rc={exc.returncode}).")
        return False


def _download_http(url: str, dest_zip: Path, auth: tuple[str, str] | None = None) -> bool:
    """Download a file via HTTP with a progress bar.  Returns True on success."""
    try:
        import requests
        from tqdm import tqdm
    except ImportError as exc:
        print(
            f"[http] Missing dependency: {exc.name}.  "
            "Install with: pip install requests tqdm"
        )
        return False

    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"[http] Downloading {url}")
    print(f"[http] Destination: {dest_zip}")
    if auth:
        print(f"[http] Authenticated as: {auth[0]}")

    try:
        resp = requests.get(url, stream=True, timeout=120, auth=auth)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[http] Request failed: {exc}")
        return False

    total = int(resp.headers.get("content-length", 0))

    with (
        open(dest_zip, "wb") as fh,
        tqdm(
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest_zip.name,
        ) as bar,
    ):
        for chunk in resp.iter_content(chunk_size=1024 * 64):
            fh.write(chunk)
            bar.update(len(chunk))

    print(f"[http] Saved {dest_zip} ({dest_zip.stat().st_size:,} bytes)")
    return True


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    """Extract a ZIP archive into *dest_dir*."""
    print(f"[extract] Unzipping {zip_path} -> {dest_dir}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print(f"[extract] Done.  Contents: {[p.name for p in dest_dir.iterdir()]}")


def download_dataset(key: str, output_dir: Path, credentials: tuple[str, str] | None = None) -> None:
    """Download data for a single dataset key.

    Tries the Kaggle CLI first, falls back to authenticated HTTP download.
    Provides manual download instructions if both methods fail.
    """
    if key not in KAGGLE_DATASETS:
        print(f"[error] Unknown dataset: {key!r}")
        print(f"[error] Available: {sorted(KAGGLE_DATASETS.keys())}")
        return

    info = KAGGLE_DATASETS[key]
    dest_dir = output_dir / key
    slug = info["dataset"]
    api_url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"

    print(f"\n{'='*60}")
    print(f"  [{key}] {info['description']}")
    print(f"  {slug}")
    print(f"{'='*60}\n")

    # --- Strategy 1: Kaggle CLI ---
    if _try_kaggle_cli(slug, dest_dir):
        print(f"[ok] Downloaded via kaggle CLI -> {dest_dir}")
        return

    # --- Strategy 2: Authenticated HTTP ---
    zip_name = f"{key}.zip"
    zip_path = output_dir / zip_name
    if _download_http(api_url, zip_path, auth=credentials):
        _extract_zip(zip_path, dest_dir)
        zip_path.unlink(missing_ok=True)
        print(f"[ok] Downloaded via HTTP -> {dest_dir}")
        return

    # --- Fallback: Manual instructions ---
    print(f"\n[warning] Automatic download failed for [{key}].")
    print("  Download manually:")
    print(f"  1. Visit: {info['manual_url']}")
    print("  2. Click 'Download' (needs Kaggle account).")
    print(f"  3. Extract the ZIP into: {dest_dir}")
    print()


def list_datasets() -> None:
    """Print all available datasets grouped by category."""
    print("\n  Available Datasets")
    print("  " + "=" * 56)

    categories = {}
    for key, info in KAGGLE_DATASETS.items():
        cat = info["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((key, info))

    for cat, items in categories.items():
        label = {"futures": "Futures OHLCV", "lob": "Limit Order Book (LOB)"}.get(cat, cat)
        print(f"\n  {label}")
        print("  " + "-" * 56)
        for key, info in items:
            print(f"    {key:<18} {info['description']}")

    print(f"\n  Batch groups:")
    print(f"    {'all-futures':<18} All futures datasets ({len(DATASET_GROUPS['all-futures'])})")
    print(f"    {'all-lob':<18} All LOB datasets ({len(DATASET_GROUPS['all-lob'])})")
    print(f"    {'all':<18} Everything ({len(DATASET_GROUPS['all'])})")
    print()


def main() -> None:
    all_choices = sorted(KAGGLE_DATASETS.keys()) + sorted(DATASET_GROUPS.keys())

    parser = argparse.ArgumentParser(
        description="Download futures / orderbook data from Kaggle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m scripts.download_kaggle --list\n"
            "  python -m scripts.download_kaggle --dataset es-cme\n"
            "  python -m scripts.download_kaggle --dataset all-futures\n"
            "  python -m scripts.download_kaggle --dataset lob\n"
            "  python -m scripts.download_kaggle --dataset all\n"
        ),
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available datasets and exit.",
    )
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help=f"Dataset key or group to download. Use --list to see options.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory for downloaded files (default: data/raw).",
    )

    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    if args.dataset is None:
        parser.error("--dataset is required (use --list to see available datasets)")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()

    # Resolve dataset key(s)
    key = args.dataset.lower().strip()
    if key in DATASET_GROUPS:
        keys = DATASET_GROUPS[key]
    elif key in KAGGLE_DATASETS:
        keys = [key]
    else:
        print(f"[error] Unknown dataset: {key!r}")
        print(f"[hint] Use --list to see available datasets.")
        sys.exit(1)

    # Load Kaggle credentials
    creds = _load_kaggle_credentials()
    if creds:
        print(f"[auth] Kaggle credentials loaded for: {creds[0]}")
    else:
        print("[auth] No Kaggle credentials found. Will try unauthenticated download.")
        print("[hint] Set KAGGLE_USERNAME and KAGGLE_KEY in .env")

    print(f"Downloading {len(keys)} dataset(s) to {output_dir}\n")

    for k in keys:
        download_dataset(k, output_dir, credentials=creds)

    print("\nDone.")


if __name__ == "__main__":
    main()
