"""
Portfolio management for the backtesting engine.

Tracks positions, cash, equity, and converts signals into orders.
Records completed round-trip trades and builds the equity curve.
"""

from __future__ import annotations

import logging
from datetime import datetime

from data.models import EquityPoint, INSTRUMENT_SPECS, Trade

from backtest.events import (
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalDirection,
    SignalEvent,
)

logger = logging.getLogger(__name__)


class Portfolio:
    """Simulated portfolio that manages a single-instrument, single-position account.

    The portfolio converts incoming signals to orders, processes fills, and
    maintains a full equity curve and trade log.

    Attributes:
        initial_capital: Starting cash balance.
        cash:            Current cash balance (changes on fills).
        position:        Current position size (positive=long, negative=short, 0=flat).
        entry_price:     Average entry price of the current position, or None.
        entry_time:      Timestamp when the current position was opened, or None.
        entry_bar_index: Bar index at which the current position was opened.
        equity_curve:    List of EquityPoint snapshots, one per bar.
        trades:          List of completed round-trip Trade records.
    """

    def __init__(
        self,
        initial_capital: float,
        commission_per_contract: float,
        instrument_specs: dict,
    ) -> None:
        self.initial_capital = initial_capital
        self.commission_per_contract = commission_per_contract
        self.tick_size: float = instrument_specs["tick_size"]
        self.tick_value: float = instrument_specs["tick_value"]
        self.point_value: float = self.tick_value / self.tick_size
        self.instrument_name: str = instrument_specs.get("name", "Unknown")

        # Position state
        self.cash: float = initial_capital
        self.position: int = 0
        self.entry_price: float | None = None
        self.entry_time: datetime | None = None
        self.entry_bar_index: int = 0

        # Current market price for unrealized PnL
        self._current_price: float = 0.0

        # Records
        self.equity_curve: list[EquityPoint] = []
        self.trades: list[Trade] = []

    # ------------------------------------------------------------------
    # Signal -> Order
    # ------------------------------------------------------------------

    def on_signal(self, signal: SignalEvent, quantity: int) -> OrderEvent | None:
        """Convert a signal event into an order event.

        Rules:
            - EXIT with no position -> None (nothing to close).
            - LONG when already long -> None (already positioned).
            - SHORT when already short -> None (already positioned).
            - LONG/SHORT when positioned opposite -> close first via EXIT.
            - Otherwise -> create order in signal direction.

        Args:
            signal:   The signal to process.
            quantity: Number of contracts to trade.

        Returns:
            An OrderEvent to submit, or None if no action is needed.
        """
        direction = signal.direction

        # EXIT signal
        if direction == SignalDirection.EXIT:
            if self.position == 0:
                return None
            # Close existing position
            close_dir = (
                SignalDirection.SHORT if self.position > 0 else SignalDirection.LONG
            )
            return OrderEvent(
                timestamp=signal.timestamp,
                direction=close_dir,
                quantity=abs(self.position),
            )

        # LONG signal
        if direction == SignalDirection.LONG:
            if self.position > 0:
                # Already long, skip
                return None
            if self.position < 0:
                # Currently short -- close first by going long to cover
                return OrderEvent(
                    timestamp=signal.timestamp,
                    direction=SignalDirection.LONG,
                    quantity=abs(self.position),
                )
            # Flat -> open long
            return OrderEvent(
                timestamp=signal.timestamp,
                direction=SignalDirection.LONG,
                quantity=quantity,
            )

        # SHORT signal
        if direction == SignalDirection.SHORT:
            if self.position < 0:
                # Already short, skip
                return None
            if self.position > 0:
                # Currently long -- close first by going short to sell
                return OrderEvent(
                    timestamp=signal.timestamp,
                    direction=SignalDirection.SHORT,
                    quantity=abs(self.position),
                )
            # Flat -> open short
            return OrderEvent(
                timestamp=signal.timestamp,
                direction=SignalDirection.SHORT,
                quantity=quantity,
            )

        return None

    # ------------------------------------------------------------------
    # Fill processing
    # ------------------------------------------------------------------

    def on_fill(self, fill: FillEvent, bar_index: int) -> None:
        """Process a fill event, updating position and recording trades.

        If the fill closes (or partially closes) a position, a Trade record
        is appended to self.trades.

        Args:
            fill:      The fill event from the broker.
            bar_index: Current bar index for trade record keeping.
        """
        fill_direction_int = fill.direction.value  # 1 for LONG, -1 for SHORT
        fill_quantity = fill.quantity
        new_position = self.position + fill_direction_int * fill_quantity

        # Determine if this fill is opening or closing
        is_closing = (
            (self.position > 0 and fill_direction_int < 0)
            or (self.position < 0 and fill_direction_int > 0)
        )

        if is_closing and self.position != 0:
            # Closing (fully or partially) an existing position
            close_qty = min(abs(self.position), fill_quantity)
            trade_direction = 1 if self.position > 0 else -1

            # PnL = (exit - entry) * direction * quantity * point_value
            gross_pnl = (
                (fill.fill_price - self.entry_price)
                * trade_direction
                * close_qty
                * self.point_value
            )
            total_commission = fill.commission  # broker already computed round-trip
            net_pnl = gross_pnl - total_commission

            trade = Trade(
                entry_time=self.entry_time,
                exit_time=fill.timestamp,
                instrument=self.instrument_name,
                direction=trade_direction,
                quantity=close_qty,
                entry_price=self.entry_price,
                exit_price=fill.fill_price,
                pnl=net_pnl,
                commission=total_commission,
                bars_held=bar_index - self.entry_bar_index,
            )
            self.trades.append(trade)
            self.cash += gross_pnl - total_commission

            logger.debug(
                "Trade closed: dir=%d qty=%d entry=%.2f exit=%.2f pnl=%.2f",
                trade_direction,
                close_qty,
                self.entry_price,
                fill.fill_price,
                net_pnl,
            )
        elif not is_closing:
            # Opening a new position
            self.entry_price = fill.fill_price
            self.entry_time = fill.timestamp
            self.entry_bar_index = bar_index
            # Deduct commission from cash (entry side)
            self.cash -= fill.commission

            logger.debug(
                "Position opened: dir=%d qty=%d price=%.2f",
                fill_direction_int,
                fill_quantity,
                fill.fill_price,
            )

        self.position = new_position

        # If now flat, reset entry tracking
        if self.position == 0:
            self.entry_price = None
            self.entry_time = None
            self.entry_bar_index = 0

    # ------------------------------------------------------------------
    # Market update
    # ------------------------------------------------------------------

    def update_market(self, bar: MarketEvent) -> None:
        """Update unrealized PnL with current market data and record equity point.

        Called once per bar after all fills and risk checks.

        Args:
            bar: The current market event.
        """
        self._current_price = bar.close

        equity = self.get_equity()
        unrealized = self._unrealized_pnl()

        point = EquityPoint(
            timestamp=bar.timestamp,
            bar_index=bar.bar_index,
            equity=equity,
            cash=self.cash,
            unrealized_pnl=unrealized,
        )
        self.equity_curve.append(point)

    # ------------------------------------------------------------------
    # Equity helpers
    # ------------------------------------------------------------------

    def get_equity(self) -> float:
        """Return total account equity: cash + unrealized PnL."""
        return self.cash + self._unrealized_pnl()

    def _unrealized_pnl(self) -> float:
        """Compute unrealized PnL on the open position."""
        if self.position == 0 or self.entry_price is None:
            return 0.0

        direction = 1 if self.position > 0 else -1
        return (
            (self._current_price - self.entry_price)
            * direction
            * abs(self.position)
            * self.point_value
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def force_close_order(self, timestamp: datetime) -> OrderEvent | None:
        """Generate an order to flatten the current position.

        Used at end-of-backtest or when risk manager forces a close.

        Returns:
            An OrderEvent to close the position, or None if already flat.
        """
        if self.position == 0:
            return None

        close_dir = (
            SignalDirection.SHORT if self.position > 0 else SignalDirection.LONG
        )
        return OrderEvent(
            timestamp=timestamp,
            direction=close_dir,
            quantity=abs(self.position),
        )
