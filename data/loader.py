"""
DataLoader -- high-level facade for accessing market data.

Provides a simplified interface over :class:`DataStore` with input
validation and sensible defaults.  This is the primary entry point
that strategies and backtest engines should use to obtain bar data.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from data.models import INSTRUMENT_SPECS, Timeframe
from data.store import DataStore

logger = logging.getLogger(__name__)

# Accepted timeframe aliases (user-facing string -> canonical value)
_TIMEFRAME_ALIASES: dict[str, str] = {}
for _tf in Timeframe:
    _TIMEFRAME_ALIASES[_tf.value] = _tf.value
    _TIMEFRAME_ALIASES[_tf.name.lower()] = _tf.value
# Extra convenience aliases
_TIMEFRAME_ALIASES.update(
    {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1hour": "1h",
        "60min": "1h",
        "d": "daily",
        "1d": "daily",
    }
)


def _normalise_timeframe(raw: str) -> str:
    """Resolve a user-supplied timeframe string to a canonical value.

    Raises ``ValueError`` if *raw* cannot be mapped.
    """
    key = raw.strip().lower()
    canonical = _TIMEFRAME_ALIASES.get(key)
    if canonical is None:
        valid = sorted(set(_TIMEFRAME_ALIASES.values()))
        raise ValueError(
            f"Unknown timeframe {raw!r}. Valid timeframes: {valid}"
        )
    return canonical


def _normalise_instrument(raw: str) -> str:
    """Resolve a user-supplied instrument string to a canonical upper-case symbol.

    Raises ``ValueError`` if *raw* is not a known instrument.
    """
    symbol = raw.strip().upper()
    if symbol not in INSTRUMENT_SPECS:
        valid = sorted(INSTRUMENT_SPECS.keys())
        raise ValueError(
            f"Unknown instrument {raw!r}. Valid instruments: {valid}"
        )
    return symbol


class DataLoader:
    """High-level facade for loading OHLCV bar data.

    Parameters
    ----------
    store : DataStore, optional
        Backing data store.  If *None* a default :class:`DataStore` is created
        with the standard ``data/parquet`` base path.
    """

    def __init__(self, store: Optional[DataStore] = None) -> None:
        self._store = store if store is not None else DataStore()

    @property
    def store(self) -> DataStore:
        return self._store

    def get_bars(
        self,
        instrument: str,
        timeframe: str,
        start_date: Optional[datetime | str] = None,
        end_date: Optional[datetime | str] = None,
    ) -> pd.DataFrame:
        """Load OHLCV bars for a given instrument and timeframe.

        Input strings are normalised: instruments are uppercased and common
        timeframe aliases (e.g. ``"1m"``, ``"5m"``, ``"1hour"``) are
        resolved to their canonical forms.

        Parameters
        ----------
        instrument : str
            Symbol such as ``"ES"``, ``"NQ"``, ``"MES"``, ``"MNQ"``.
        timeframe : str
            Timeframe label, e.g. ``"1min"``, ``"5m"``, ``"1h"``, ``"daily"``.
        start_date, end_date : datetime or str, optional
            Inclusive date boundaries for filtering.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp``, ``open``, ``high``, ``low``, ``close``,
            ``volume``.  Sorted by ``timestamp`` ascending.

        Raises
        ------
        ValueError
            If *instrument* or *timeframe* is not recognised.
        FileNotFoundError
            If no Parquet data has been stored for the given combination.
        """
        inst = _normalise_instrument(instrument)
        tf = _normalise_timeframe(timeframe)

        logger.info(
            "Loading bars: %s / %s (start=%s, end=%s)",
            inst,
            tf,
            start_date,
            end_date,
        )

        df = self._store.load_bars(inst, tf, start_date=start_date, end_date=end_date)

        # Guarantee a consistent column order
        column_order = ["timestamp", "open", "high", "low", "close", "volume"]
        extra_cols = [c for c in df.columns if c not in column_order]
        df = df[column_order + extra_cols]

        return df

    def get_available_data(self) -> dict[str, list[str]]:
        """Return a mapping of instruments to their available timeframes.

        Returns
        -------
        dict[str, list[str]]
            ``{ "ES": ["1min", "5min", ...], "NQ": [...] }``
        """
        result: dict[str, list[str]] = {}
        for inst in self._store.list_instruments():
            timeframes = self._store.list_timeframes(inst)
            if timeframes:
                result[inst] = timeframes
        return result

    def __repr__(self) -> str:
        return f"DataLoader(store={self._store!r})"
