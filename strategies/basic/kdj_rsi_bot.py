"""
KDJ + RSI Bot strategy (嚶嚶bot).

Converted from TradingView Pine Script. Enters long when KDJ D-line
crosses under 50 with RSI > 50, and short when D-line crosses over 50
with RSI < 50.

Features multi-level take-profit: splits the total position across
three TP levels at configurable tick offsets, with a shared stop-loss.
"""

from __future__ import annotations

import pandas as pd

from backtest.events import OrderEvent, SignalDirection, SignalEvent
from strategies.base import BaseStrategy, StrategyParam
from strategies.indicators import kdj, rsi


class KDJRSIBot(BaseStrategy):
    name = "KDJ RSI Bot"
    description = "KDJ D-line cross + RSI filter with multi-level take-profit (YingYing bot)."
    category = "basic"

    params = {
        "kdj_period": StrategyParam(
            name="KDJ Period",
            param_type="int",
            default=9,
            min_val=2, max_val=50, step=1,
            description="Lookback for KDJ highest-high / lowest-low.",
        ),
        "kdj_signal": StrategyParam(
            name="KDJ Signal",
            param_type="int",
            default=3,
            min_val=2, max_val=20, step=1,
            description="Smoothing period for KDJ K and D lines.",
        ),
        "rsi_period": StrategyParam(
            name="RSI Period",
            param_type="int",
            default=14,
            min_val=2, max_val=100, step=1,
            description="RSI lookback period.",
        ),
        "rsi_threshold": StrategyParam(
            name="RSI Threshold",
            param_type="float",
            default=50.0,
            min_val=30.0, max_val=70.0, step=1.0,
            description="RSI level for buy/sell filter.",
        ),
        "d_cross_level": StrategyParam(
            name="D Cross Level",
            param_type="float",
            default=50.0,
            min_val=20.0, max_val=80.0, step=1.0,
            description="D-line level for crossover signals.",
        ),
        "sl_points": StrategyParam(
            name="Stop Loss Points",
            param_type="float",
            default=15.0,
            min_val=1.0, max_val=100.0, step=0.5,
            description="Stop loss distance in points from entry.",
        ),
        "tp1_ticks": StrategyParam(
            name="TP1 Ticks",
            param_type="int",
            default=200,
            min_val=1, max_val=5000, step=10,
            description="Take-profit level 1 in ticks from entry.",
        ),
        "tp1_pct": StrategyParam(
            name="TP1 Quantity %",
            param_type="int",
            default=60,
            min_val=1, max_val=100, step=1,
            description="Percent of total contracts to close at TP1.",
        ),
        "tp2_ticks": StrategyParam(
            name="TP2 Ticks",
            param_type="int",
            default=300,
            min_val=1, max_val=5000, step=10,
            description="Take-profit level 2 in ticks from entry.",
        ),
        "tp2_pct": StrategyParam(
            name="TP2 Quantity %",
            param_type="int",
            default=20,
            min_val=1, max_val=100, step=1,
            description="Percent of total contracts to close at TP2.",
        ),
        "tp3_ticks": StrategyParam(
            name="TP3 Ticks",
            param_type="int",
            default=500,
            min_val=1, max_val=5000, step=10,
            description="Take-profit level 3 in ticks from entry.",
        ),
    }

    def on_init(self, data: pd.DataFrame) -> None:
        kdj_p = self.get_param("kdj_period")
        kdj_s = self.get_param("kdj_signal")
        rsi_p = self.get_param("rsi_period")

        self._k, self._d, self._j = kdj(
            data["high"], data["low"], data["close"],
            ilong=kdj_p, isig=kdj_s,
        )
        self._rsi = rsi(data["close"], rsi_p)
        self._in_position = False

    def on_bar(self, bar: dict, history: pd.DataFrame) -> SignalEvent | None:
        idx = len(history) - 1
        warmup = max(self.get_param("kdj_period"), self.get_param("rsi_period")) + 1
        if idx < warmup:
            return None

        d_now = self._d.iloc[idx]
        d_prev = self._d.iloc[idx - 1]
        rsi_now = self._rsi.iloc[idx]

        if pd.isna(d_now) or pd.isna(d_prev) or pd.isna(rsi_now):
            return None

        cross_level = self.get_param("d_cross_level")
        rsi_thresh = self.get_param("rsi_threshold")

        # No new signals while in a position — rely on SL/TP orders
        if self._in_position:
            return None

        # BUY: D crosses under cross_level AND RSI > threshold
        if d_prev >= cross_level and d_now < cross_level and rsi_now > rsi_thresh:
            self._in_position = True
            return SignalEvent(
                timestamp=bar["timestamp"],
                direction=SignalDirection.LONG,
            )

        # SELL: D crosses over cross_level AND RSI < threshold
        if d_prev <= cross_level and d_now > cross_level and rsi_now < rsi_thresh:
            self._in_position = True
            return SignalEvent(
                timestamp=bar["timestamp"],
                direction=SignalDirection.SHORT,
            )

        return None

    def build_order(
        self,
        signal: SignalEvent,
        bar: dict,
        quantity: int,
        tick_size: float,
    ) -> OrderEvent | None:
        """Create multi-level TP order based on Pine Script logic."""
        sl_points = self.get_param("sl_points")
        tp1_ticks = self.get_param("tp1_ticks")
        tp2_ticks = self.get_param("tp2_ticks")
        tp3_ticks = self.get_param("tp3_ticks")
        tp1_pct = self.get_param("tp1_pct")
        tp2_pct = self.get_param("tp2_pct")

        # Compute quantities from percentages
        tp1_qty = max(1, round(quantity * tp1_pct / 100))
        tp2_qty = max(1, round(quantity * tp2_pct / 100))
        tp3_qty = quantity - tp1_qty - tp2_qty
        if tp3_qty <= 0:
            tp3_qty = 1
            tp2_qty = quantity - tp1_qty - tp3_qty

        entry_price = bar["close"]

        if signal.direction == SignalDirection.LONG:
            sl_price = entry_price - sl_points
            tp_levels = [
                (entry_price + tp1_ticks * tick_size, tp1_qty),
                (entry_price + tp2_ticks * tick_size, tp2_qty),
                (entry_price + tp3_ticks * tick_size, tp3_qty),
            ]
        elif signal.direction == SignalDirection.SHORT:
            sl_price = entry_price + sl_points
            tp_levels = [
                (entry_price - tp1_ticks * tick_size, tp1_qty),
                (entry_price - tp2_ticks * tick_size, tp2_qty),
                (entry_price - tp3_ticks * tick_size, tp3_qty),
            ]
        else:
            return None

        return OrderEvent(
            timestamp=signal.timestamp,
            direction=signal.direction,
            quantity=quantity,
            stop_loss=sl_price,
            take_profit_levels=tp_levels,
        )

    def on_position_closed(self) -> None:
        self._in_position = False
