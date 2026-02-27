"""
Data models for the ancserFX quantitative futures trading system.

Defines enums for timeframes and instruments, instrument specifications,
and dataclasses for bars, trades, and equity tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Timeframe(str, Enum):
    """Supported bar timeframes for market data."""

    TICK = "tick"
    MIN_1 = "1min"
    MIN_5 = "5min"
    MIN_15 = "15min"
    HOUR_1 = "1h"
    DAILY = "daily"

    @property
    def pandas_freq(self) -> Optional[str]:
        """Return the pandas-compatible frequency string for resampling.

        Returns None for TICK since ticks cannot be resampled by time alone.
        """
        mapping = {
            Timeframe.MIN_1: "1min",
            Timeframe.MIN_5: "5min",
            Timeframe.MIN_15: "15min",
            Timeframe.HOUR_1: "1h",
            Timeframe.DAILY: "1D",
        }
        return mapping.get(self)

    @property
    def minutes(self) -> Optional[int]:
        """Return the number of minutes per bar, or None for non-time-based."""
        mapping = {
            Timeframe.MIN_1: 1,
            Timeframe.MIN_5: 5,
            Timeframe.MIN_15: 15,
            Timeframe.HOUR_1: 60,
            Timeframe.DAILY: 1440,
        }
        return mapping.get(self)


# ---------------------------------------------------------------------------
# Instrument specifications
# ---------------------------------------------------------------------------

INSTRUMENT_SPECS: dict[str, dict] = {
    "ES": {
        "tick_size": 0.25,
        "tick_value": 12.50,
        "name": "E-mini S&P 500",
        "exchange": "CME",
    },
    "NQ": {
        "tick_size": 0.25,
        "tick_value": 5.00,
        "name": "E-mini Nasdaq 100",
        "exchange": "CME",
    },
    "MNQ": {
        "tick_size": 0.25,
        "tick_value": 0.50,
        "name": "Micro E-mini Nasdaq",
        "exchange": "CME",
    },
    "MES": {
        "tick_size": 0.25,
        "tick_value": 1.25,
        "name": "Micro E-mini S&P",
        "exchange": "CME",
    },
}


class Instrument(str, Enum):
    """Supported futures instruments."""

    ES = "ES"
    NQ = "NQ"
    MNQ = "MNQ"
    MES = "MES"

    @property
    def tick_size(self) -> float:
        """Minimum price increment for this instrument."""
        return INSTRUMENT_SPECS[self.value]["tick_size"]

    @property
    def tick_value(self) -> float:
        """Dollar value of one tick move for a single contract."""
        return INSTRUMENT_SPECS[self.value]["tick_value"]

    @property
    def full_name(self) -> str:
        """Human-readable contract name."""
        return INSTRUMENT_SPECS[self.value]["name"]

    @property
    def exchange(self) -> str:
        """Exchange where this instrument is listed."""
        return INSTRUMENT_SPECS[self.value]["exchange"]

    @property
    def point_value(self) -> float:
        """Dollar value of a one-point move (tick_value / tick_size)."""
        return self.tick_value / self.tick_size


# ---------------------------------------------------------------------------
# Core data structures
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class Bar:
    """A single OHLCV bar.

    Immutable by design -- bars represent historical facts that should
    not be mutated after creation.
    """

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(
                f"Bar high ({self.high}) cannot be less than low ({self.low})"
            )
        if self.volume < 0:
            raise ValueError(f"Bar volume ({self.volume}) cannot be negative")

    @property
    def range(self) -> float:
        """High minus low."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Absolute difference between open and close."""
        return abs(self.close - self.open)

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2.0


@dataclass(slots=True)
class Trade:
    """Record of a completed round-trip trade.

    Attributes:
        entry_time:   Timestamp when the position was opened.
        exit_time:    Timestamp when the position was closed.
        instrument:   Instrument symbol (e.g. "ES", "NQ").
        direction:    1 for long, -1 for short.
        quantity:     Number of contracts.
        entry_price:  Average fill price at entry.
        exit_price:   Average fill price at exit.
        pnl:          Realized profit/loss in dollars (after commissions).
        commission:   Total round-trip commission cost.
        bars_held:    Number of bars the position was open.
    """

    entry_time: datetime
    exit_time: datetime
    instrument: str
    direction: int  # 1 = long, -1 = short
    quantity: int
    entry_price: float
    exit_price: float
    pnl: float
    commission: float = 0.0
    bars_held: int = 0

    def __post_init__(self) -> None:
        if self.direction not in (1, -1):
            raise ValueError(
                f"Trade direction must be 1 (long) or -1 (short), got {self.direction}"
            )
        if self.quantity <= 0:
            raise ValueError(f"Trade quantity must be positive, got {self.quantity}")

    @property
    def gross_pnl(self) -> float:
        """PnL before commissions."""
        return self.pnl + self.commission

    @property
    def pnl_per_contract(self) -> float:
        """Net PnL per contract."""
        return self.pnl / self.quantity if self.quantity else 0.0

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0.0

    @property
    def price_change(self) -> float:
        """Directional price change (positive when trade is profitable before costs)."""
        return (self.exit_price - self.entry_price) * self.direction

    @property
    def duration(self):
        """Wall-clock duration of the trade as a timedelta."""
        return self.exit_time - self.entry_time


@dataclass(slots=True, frozen=True)
class EquityPoint:
    """A snapshot of account equity at a specific point in time.

    Used to build an equity curve during backtesting.
    """

    timestamp: datetime
    bar_index: int
    equity: float
    cash: float
    unrealized_pnl: float = 0.0

    @property
    def total_value(self) -> float:
        """Total account value (cash + unrealized)."""
        return self.cash + self.unrealized_pnl
