"""
DataStore -- Parquet-backed storage for OHLCV bar data.

Manages reading and writing of market data as Parquet files organised
in a directory tree: ``{base_path}/{instrument}/{timeframe}/data.parquet``.

All path handling uses ``pathlib.Path`` for cross-platform compatibility.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Project root is two levels up from this file (data/store.py -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class DataStore:
    """Manages Parquet file storage for OHLCV bar data.

    Parameters
    ----------
    base_path : str
        Root directory for parquet files.  If the path is relative it is
        resolved from the project root directory.
    """

    REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}

    def __init__(self, base_path: str = "data/parquet") -> None:
        path = Path(base_path)
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        self._base_path: Path = path.resolve()
        logger.debug("DataStore initialised with base_path=%s", self._base_path)

    @property
    def base_path(self) -> Path:
        return self._base_path

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def save_bars(
        self,
        instrument: str,
        timeframe: str,
        df: pd.DataFrame,
    ) -> Path:
        """Persist a DataFrame of OHLCV bars to Parquet.

        Parameters
        ----------
        instrument : str
            Instrument symbol (e.g. ``"ES"``).  Stored in lower-case directory.
        timeframe : str
            Timeframe label (e.g. ``"1min"``, ``"5min"``).
        df : pd.DataFrame
            Must contain at minimum the columns: timestamp, open, high, low,
            close, volume.

        Returns
        -------
        Path
            Absolute path to the written parquet file.

        Raises
        ------
        ValueError
            If required columns are missing from *df*.
        """
        missing = self.REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"DataFrame is missing required columns: {sorted(missing)}"
            )

        dest_dir = self._base_path / instrument.lower() / timeframe
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_file = dest_dir / "data.parquet"

        # Ensure timestamp is datetime and sort
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        df.to_parquet(dest_file, engine="pyarrow", index=False)
        logger.info(
            "Saved %d bars -> %s (%s / %s)",
            len(df),
            dest_file,
            instrument.upper(),
            timeframe,
        )
        return dest_file

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def load_bars(
        self,
        instrument: str,
        timeframe: str,
        start_date: Optional[datetime | str] = None,
        end_date: Optional[datetime | str] = None,
    ) -> pd.DataFrame:
        """Load OHLCV bars from Parquet, with optional date filtering.

        Parameters
        ----------
        instrument : str
            Instrument symbol.
        timeframe : str
            Timeframe label.
        start_date, end_date : datetime or str, optional
            Inclusive date boundaries.  Strings are parsed with
            ``pd.to_datetime``.

        Returns
        -------
        pd.DataFrame
            Sorted by timestamp ascending with a reset integer index.

        Raises
        ------
        FileNotFoundError
            If no parquet file exists for the given instrument/timeframe.
        """
        parquet_path = self._parquet_path(instrument, timeframe)
        if not parquet_path.exists():
            raise FileNotFoundError(
                f"No data found for {instrument.upper()}/{timeframe} "
                f"at {parquet_path}"
            )

        df = pd.read_parquet(parquet_path, engine="pyarrow")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.sort_values("timestamp", inplace=True)

        if start_date is not None:
            start_dt = pd.to_datetime(start_date)
            df = df[df["timestamp"] >= start_dt]

        if end_date is not None:
            end_dt = pd.to_datetime(end_date)
            df = df[df["timestamp"] <= end_dt]

        df.reset_index(drop=True, inplace=True)
        logger.debug(
            "Loaded %d bars for %s/%s%s",
            len(df),
            instrument.upper(),
            timeframe,
            f" ({start_date} -> {end_date})" if start_date or end_date else "",
        )
        return df

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def list_instruments(self) -> list[str]:
        """Return a sorted list of instrument symbols that have stored data.

        Scans immediate subdirectories of ``base_path``.  Directory names
        are returned in UPPER case to match :class:`Instrument` enum values.
        """
        if not self._base_path.exists():
            return []
        return sorted(
            d.name.upper()
            for d in self._base_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def list_timeframes(self, instrument: str) -> list[str]:
        """Return available timeframes for *instrument*.

        Scans sub-directories under ``{base_path}/{instrument}/`` that
        contain a ``data.parquet`` file.
        """
        inst_dir = self._base_path / instrument.lower()
        if not inst_dir.exists():
            return []
        return sorted(
            d.name
            for d in inst_dir.iterdir()
            if d.is_dir() and (d / "data.parquet").exists()
        )

    def get_date_range(
        self, instrument: str, timeframe: str
    ) -> tuple[datetime, datetime, int]:
        """Return ``(first_timestamp, last_timestamp, bar_count)`` for a dataset.

        Raises
        ------
        FileNotFoundError
            If no data exists for the given instrument/timeframe.
        """
        df = self.load_bars(instrument, timeframe)
        if df.empty:
            raise ValueError(
                f"Parquet file for {instrument.upper()}/{timeframe} is empty"
            )
        first_ts: datetime = df["timestamp"].iloc[0].to_pydatetime()
        last_ts: datetime = df["timestamp"].iloc[-1].to_pydatetime()
        return first_ts, last_ts, len(df)

    def has_data(self, instrument: str, timeframe: str) -> bool:
        """Check whether a parquet file exists for the given instrument/timeframe."""
        return self._parquet_path(instrument, timeframe).exists()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parquet_path(self, instrument: str, timeframe: str) -> Path:
        """Build the canonical parquet file path."""
        return self._base_path / instrument.lower() / timeframe / "data.parquet"

    def __repr__(self) -> str:
        return f"DataStore(base_path={str(self._base_path)!r})"
