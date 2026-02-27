"""
Delta Momentum strategy (orderflow-based).

Uses cumulative delta and volume imbalance (approximated from OHLCV) to
detect institutional buying/selling pressure.  Enters in the direction
of the delta momentum when confirmed by volume imbalance.

This is a simplified orderflow proxy strategy -- true delta requires
tick-level trade data, but the OHLCV approximation still captures
directional volume pressure.
"""

from __future__ import annotations

import pandas as pd

from backtest.events import SignalDirection, SignalEvent
from strategies.base import BaseStrategy, StrategyParam
from strategies.indicators import cumulative_delta, ema, volume_imbalance


class DeltaMomentum(BaseStrategy):
    name = "Delta Momentum"
    description = "Orderflow momentum using cumulative delta and volume imbalance."
    category = "orderflow"

    params = {
        "delta_ema_period": StrategyParam(
            name="Delta EMA Period",
            param_type="int",
            default=20,
            min_val=5,
            max_val=100,
            step=1,
            description="EMA smoothing period for cumulative delta.",
        ),
        "imbalance_lookback": StrategyParam(
            name="Imbalance Lookback",
            param_type="int",
            default=10,
            min_val=3,
            max_val=50,
            step=1,
            description="Rolling window for volume imbalance ratio.",
        ),
        "imbalance_threshold": StrategyParam(
            name="Imbalance Threshold",
            param_type="float",
            default=0.3,
            min_val=0.05,
            max_val=0.9,
            step=0.05,
            description="Minimum absolute imbalance to confirm a signal (0-1 scale).",
        ),
        "exit_imbalance": StrategyParam(
            name="Exit Imbalance",
            param_type="float",
            default=0.0,
            min_val=-0.5,
            max_val=0.5,
            step=0.05,
            description="Imbalance level at which to exit (0 = neutral, exit when flow reverses).",
        ),
        "trend_ema_period": StrategyParam(
            name="Trend EMA Period",
            param_type="int",
            default=50,
            min_val=0,
            max_val=200,
            step=1,
            description="EMA for trend filter (0 = disabled).",
        ),
    }

    def on_init(self, data: pd.DataFrame) -> None:
        # Compute cumulative delta
        self._cum_delta = cumulative_delta(
            volume=data["volume"],
            close=data["close"],
            open_price=data["open"],
        )

        # Smooth the delta with EMA
        delta_ema_period = self.get_param("delta_ema_period")
        self._delta_ema = ema(self._cum_delta, delta_ema_period)

        # Volume imbalance
        imb_lookback = self.get_param("imbalance_lookback")
        self._vol_imbalance = volume_imbalance(
            volume=data["volume"],
            close=data["close"],
            open_price=data["open"],
            lookback=imb_lookback,
        )

        # Optional trend filter
        trend_period = self.get_param("trend_ema_period")
        if trend_period > 0:
            self._trend_ema = ema(data["close"], trend_period)
        else:
            self._trend_ema = None

        self._in_position = False
        self._position_dir = 0

    def on_bar(self, bar: dict, history: pd.DataFrame) -> SignalEvent | None:
        idx = len(history) - 1
        warmup = max(
            self.get_param("delta_ema_period"),
            self.get_param("imbalance_lookback"),
        )
        if idx < warmup + 1:
            return None

        delta_now = self._delta_ema.iloc[idx]
        delta_prev = self._delta_ema.iloc[idx - 1]
        imbalance = self._vol_imbalance.iloc[idx]

        if pd.isna(delta_now) or pd.isna(delta_prev) or pd.isna(imbalance):
            return None

        threshold = self.get_param("imbalance_threshold")
        exit_imb = self.get_param("exit_imbalance")
        close_price = bar["close"]

        # Delta momentum direction: rising = bullish, falling = bearish
        delta_rising = delta_now > delta_prev
        delta_falling = delta_now < delta_prev

        # Trend filter
        allow_long = True
        allow_short = True
        if self._trend_ema is not None and not pd.isna(self._trend_ema.iloc[idx]):
            trend_val = self._trend_ema.iloc[idx]
            allow_long = close_price > trend_val
            allow_short = close_price < trend_val

        # Exit logic
        if self._in_position:
            should_exit = False

            if self._position_dir == 1:
                # Exit long when imbalance flips bearish or delta reverses
                should_exit = imbalance < exit_imb or delta_falling
            elif self._position_dir == -1:
                # Exit short when imbalance flips bullish or delta reverses
                should_exit = imbalance > -exit_imb or delta_rising

            if should_exit:
                self._in_position = False
                self._position_dir = 0
                return SignalEvent(
                    timestamp=bar["timestamp"],
                    direction=SignalDirection.EXIT,
                )
            return None

        # Entry logic: delta momentum + volume imbalance confirmation
        if delta_rising and imbalance > threshold and allow_long:
            self._in_position = True
            self._position_dir = 1
            return SignalEvent(
                timestamp=bar["timestamp"],
                direction=SignalDirection.LONG,
                strength=min(1.0, abs(imbalance)),
            )

        if delta_falling and imbalance < -threshold and allow_short:
            self._in_position = True
            self._position_dir = -1
            return SignalEvent(
                timestamp=bar["timestamp"],
                direction=SignalDirection.SHORT,
                strength=min(1.0, abs(imbalance)),
            )

        return None
