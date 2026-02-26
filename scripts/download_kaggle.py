#!/usr/bin/env python3
"""
Download ES / NQ futures minute-bar data from Kaggle.

Attempts the Kaggle CLI first, then falls back to ``requests`` with a
``tqdm`` progress bar for a direct HTTP download.

Usage
-----
    # Download both ES and NQ
    python -m scripts.download_kaggle --instrument all

    # Download only ES to a custom directory
    python -m scripts.download_kaggle --instrument es --output-dir data/raw

    # Show help
    python -m scripts.download_kaggle --help
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# -------------------------------------------------------------------
# Known Kaggle datasets for futures data
# Each entry contains the Kaggle dataset slug and a description.
# -------------------------------------------------------------------
KAGGLE_DATASETS: dict[str, dict] = {
    "es": {
        "dataset": "jorijnsmit/emini-sp-500-futures",
        "description": "E-mini S&P 500 Futures (ES) -- minute bars",
        "url": "https://www.kaggle.com/api/v1/datasets/download/jorijnsmit/emini-sp-500-futures",
        "manual_url": "https://www.kaggle.com/datasets/jorijnsmit/emini-sp-500-futures",
    },
    "nq": {
        "dataset": "jorijnsmit/emini-nasdaq-100-futures",
        "description": "E-mini Nasdaq 100 Futures (NQ) -- minute bars",
        "url": "https://www.kaggle.com/api/v1/datasets/download/jorijnsmit/emini-nasdaq-100-futures",
        "manual_url": "https://www.kaggle.com/datasets/jorijnsmit/emini-nasdaq-100-futures",
    },
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


def _download_http(url: str, dest_zip: Path) -> bool:
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

    try:
        resp = requests.get(url, stream=True, timeout=60)
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


def download_instrument(instrument: str, output_dir: Path) -> None:
    """Download data for a single instrument.

    Tries the Kaggle CLI first, falls back to direct HTTP download.
    Provides manual download instructions if both methods fail.
    """
    key = instrument.lower()
    if key not in KAGGLE_DATASETS:
        print(f"[error] Unknown instrument: {instrument!r}")
        print(f"[error] Available: {list(KAGGLE_DATASETS.keys())}")
        return

    info = KAGGLE_DATASETS[key]
    inst_dir = output_dir / key
    print(f"\n{'='*60}")
    print(f"  {info['description']}")
    print(f"{'='*60}\n")

    # --- Strategy 1: Kaggle CLI ---
    if _try_kaggle_cli(info["dataset"], inst_dir):
        print(f"[ok] Data downloaded via kaggle CLI -> {inst_dir}")
        return

    # --- Strategy 2: Direct HTTP ---
    zip_name = f"{key}_futures.zip"
    zip_path = output_dir / zip_name
    if _download_http(info["url"], zip_path):
        _extract_zip(zip_path, inst_dir)
        # Clean up zip
        zip_path.unlink(missing_ok=True)
        print(f"[ok] Data downloaded via HTTP -> {inst_dir}")
        return

    # --- Fallback: Manual instructions ---
    print(f"\n[warning] Automatic download failed for {instrument.upper()}.")
    print("  You can download the data manually:")
    print(f"  1. Visit: {info['manual_url']}")
    print("  2. Click 'Download' (you may need a Kaggle account).")
    print(f"  3. Extract the ZIP into: {inst_dir}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ES/NQ futures minute-bar data from Kaggle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m scripts.download_kaggle --instrument all\n"
            "  python -m scripts.download_kaggle --instrument es\n"
            "  python -m scripts.download_kaggle --instrument nq --output-dir ./my_data\n"
        ),
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default="all",
        choices=["es", "nq", "all"],
        help="Instrument to download: es, nq, or all (default: all).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory for downloaded files (default: data/raw).",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()

    instruments = list(KAGGLE_DATASETS.keys()) if args.instrument == "all" else [args.instrument]

    for inst in instruments:
        download_instrument(inst, output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
