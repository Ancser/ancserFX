"""
RSI Mean Reversion strategy.

Enters long when RSI crosses below the oversold threshold and reverts back
above it (bottom reversal), and enters short when RSI crosses above the
overbought threshold and drops back below (top reversal).

Exits on the opposite signal or when RSI returns to the neutral zone.
"""

from __future__ import annotations

import pandas as pd

from backtest.events import SignalDirection, SignalEvent
from strategies.base import BaseStrategy, StrategyParam
from strategies.indicators import ema, rsi


class RSIMeanReversion(BaseStrategy):
    name = "RSI Mean Reversion"
    description = "Mean reversion on RSI oversold/overbought with EMA trend filter."
    category = "basic"

    params = {
        "rsi_period": StrategyParam(
            name="RSI Period",
            param_type="int",
            default=14,
            min_val=2,
            max_val=100,
            step=1,
            description="Lookback period for RSI calculation.",
        ),
        "oversold": StrategyParam(
            name="Oversold Level",
            param_type="float",
            default=30.0,
            min_val=5.0,
            max_val=50.0,
            step=1.0,
            description="RSI level below which the market is considered oversold.",
        ),
        "overbought": StrategyParam(
            name="Overbought Level",
            param_type="float",
            default=70.0,
            min_val=50.0,
            max_val=95.0,
            step=1.0,
            description="RSI level above which the market is considered overbought.",
        ),
        "exit_level": StrategyParam(
            name="Exit Level",
            param_type="float",
            default=50.0,
            min_val=30.0,
            max_val=70.0,
            step=1.0,
            description="RSI level at which to close the position (neutral zone).",
        ),
        "ema_filter_period": StrategyParam(
            name="EMA Filter Period",
            param_type="int",
            default=0,
            min_val=0,
            max_val=500,
            step=1,
            description="EMA period for trend filter (0 = disabled). Only take longs above EMA, shorts below.",
        ),
    }

    def on_init(self, data: pd.DataFrame) -> None:
        self._rsi = rsi(data["close"], self.get_param("rsi_period"))

        ema_period = self.get_param("ema_filter_period")
        if ema_period > 0:
            self._ema = ema(data["close"], ema_period)
        else:
            self._ema = None

        self._in_position = False
        self._position_dir = 0  # 1 = long, -1 = short

    def on_bar(self, bar: dict, history: pd.DataFrame) -> SignalEvent | None:
        idx = len(history) - 1
        rsi_period = self.get_param("rsi_period")

        if idx < rsi_period + 1:
            return None

        rsi_now = self._rsi.iloc[idx]
        rsi_prev = self._rsi.iloc[idx - 1]

        if pd.isna(rsi_now) or pd.isna(rsi_prev):
            return None

        oversold = self.get_param("oversold")
        overbought = self.get_param("overbought")
        exit_level = self.get_param("exit_level")
        close_price = bar["close"]

        # Trend filter
        allow_long = True
        allow_short = True
        if self._ema is not None and not pd.isna(self._ema.iloc[idx]):
            ema_val = self._ema.iloc[idx]
            allow_long = close_price > ema_val
            allow_short = close_price < ema_val

        # Exit logic: RSI returns to neutral
        if self._in_position:
            if self._position_dir == 1 and rsi_now >= exit_level:
                self._in_position = False
                self._position_dir = 0
                return SignalEvent(
                    timestamp=bar["timestamp"],
                    direction=SignalDirection.EXIT,
                )
            elif self._position_dir == -1 and rsi_now <= exit_level:
                self._in_position = False
                self._position_dir = 0
                return SignalEvent(
                    timestamp=bar["timestamp"],
                    direction=SignalDirection.EXIT,
                )
            return None

        # Entry logic: RSI reversal from extreme
        # Long: RSI was below oversold, now crosses back above
        if rsi_prev < oversold and rsi_now >= oversold and allow_long:
            self._in_position = True
            self._position_dir = 1
            return SignalEvent(
                timestamp=bar["timestamp"],
                direction=SignalDirection.LONG,
                strength=min(1.0, (oversold - rsi_prev) / oversold),
            )

        # Short: RSI was above overbought, now crosses back below
        if rsi_prev > overbought and rsi_now <= overbought and allow_short:
            self._in_position = True
            self._position_dir = -1
            return SignalEvent(
                timestamp=bar["timestamp"],
                direction=SignalDirection.SHORT,
                strength=min(1.0, (rsi_prev - overbought) / (100 - overbought)),
            )

        return None
