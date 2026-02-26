"""
SMA Crossover strategy.

Goes long when the fast SMA crosses above the slow SMA, and short when
the fast SMA crosses below.  An optional ATR-based stop-loss and
take-profit can be attached to each order.
"""

from __future__ import annotations

import pandas as pd

from backtest.events import SignalDirection, SignalEvent
from strategies.base import BaseStrategy, StrategyParam
from strategies.indicators import atr, sma


class SMACrossover(BaseStrategy):
    name = "SMA Crossover"
    description = "Dual SMA crossover with optional ATR-based SL/TP."
    category = "basic"

    params = {
        "fast_period": StrategyParam(
            name="Fast SMA Period",
            param_type="int",
            default=10,
            min_val=2,
            max_val=200,
            step=1,
            description="Period for the fast (short-term) SMA.",
        ),
        "slow_period": StrategyParam(
            name="Slow SMA Period",
            param_type="int",
            default=30,
            min_val=5,
            max_val=500,
            step=1,
            description="Period for the slow (long-term) SMA.",
        ),
        "atr_period": StrategyParam(
            name="ATR Period",
            param_type="int",
            default=14,
            min_val=2,
            max_val=100,
            step=1,
            description="ATR lookback for volatility measurement.",
        ),
        "atr_sl_mult": StrategyParam(
            name="ATR SL Multiplier",
            param_type="float",
            default=2.0,
            min_val=0.0,
            max_val=10.0,
            step=0.1,
            description="Stop loss distance as ATR multiple (0 = disabled).",
        ),
        "atr_tp_mult": StrategyParam(
            name="ATR TP Multiplier",
            param_type="float",
            default=3.0,
            min_val=0.0,
            max_val=10.0,
            step=0.1,
            description="Take profit distance as ATR multiple (0 = disabled).",
        ),
    }

    def on_init(self, data: pd.DataFrame) -> None:
        fast = self.get_param("fast_period")
        slow = self.get_param("slow_period")
        atr_p = self.get_param("atr_period")

        self._fast_sma = sma(data["close"], fast)
        self._slow_sma = sma(data["close"], slow)
        self._atr = atr(data["high"], data["low"], data["close"], atr_p)

    def on_bar(self, bar: dict, history: pd.DataFrame) -> SignalEvent | None:
        idx = len(history) - 1
        slow = self.get_param("slow_period")

        # Need enough bars for the slow SMA to be valid
        if idx < slow:
            return None

        fast_now = self._fast_sma.iloc[idx]
        fast_prev = self._fast_sma.iloc[idx - 1]
        slow_now = self._slow_sma.iloc[idx]
        slow_prev = self._slow_sma.iloc[idx - 1]

        if pd.isna(fast_now) or pd.isna(slow_now) or pd.isna(fast_prev) or pd.isna(slow_prev):
            return None

        # Detect crossover
        crossed_above = fast_prev <= slow_prev and fast_now > slow_now
        crossed_below = fast_prev >= slow_prev and fast_now < slow_now

        if crossed_above:
            return SignalEvent(
                timestamp=bar["timestamp"],
                direction=SignalDirection.LONG,
                strength=1.0,
            )
        elif crossed_below:
            return SignalEvent(
                timestamp=bar["timestamp"],
                direction=SignalDirection.SHORT,
                strength=1.0,
            )

        return None
