"""
Simulated broker for the backtesting engine.

Handles order execution with configurable slippage and commission,
and manages stop-loss / take-profit orders attached to positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from backtest.events import (
    FillEvent,
    MarketEvent,
    OrderEvent,
    SignalDirection,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _SLTPOrder:
    """Internal representation of a stop-loss or take-profit order.

    Attributes:
        position_direction: Direction of the position being protected (1=long, -1=short).
        stop_loss:          Absolute stop-loss price, or None.
        take_profit:        Absolute take-profit price, or None.
        quantity:           Number of contracts.
    """

    position_direction: int  # 1 = long position, -1 = short position
    stop_loss: float | None
    take_profit: float | None
    quantity: int


class SimulatedBroker:
    """Simulates order execution against historical bar data.

    Market orders are filled at the next bar's open price with slippage applied.
    Stop-loss and take-profit orders are checked against each bar's high/low
    and filled at the trigger price (or worse on a gap).

    Attributes:
        slippage_ticks: Number of ticks of adverse slippage per fill.
        tick_size:      Minimum price increment for the instrument.
        commission:     Commission per contract per side.
        pending_orders: Queue of orders waiting to be filled on the next bar.
        sl_tp_orders:   Active stop-loss / take-profit orders protecting positions.
    """

    def __init__(
        self,
        slippage_ticks: int = 1,
        tick_size: float = 0.25,
        commission: float = 2.50,
    ) -> None:
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        self.commission = commission

        self.pending_orders: list[OrderEvent] = []
        self.sl_tp_orders: list[_SLTPOrder] = []

    def submit_order(self, order: OrderEvent) -> None:
        """Queue an order for execution on the next bar.

        If the order has stop_loss or take_profit set, a corresponding
        SL/TP order is registered to protect the new position.

        Args:
            order: The order event to submit.
        """
        self.pending_orders.append(order)

        # Register SL/TP if the order opens a new position (LONG or SHORT, not EXIT-close)
        if order.direction in (SignalDirection.LONG, SignalDirection.SHORT):
            if order.stop_loss is not None or order.take_profit is not None:
                position_dir = order.direction.value  # 1 for LONG, -1 for SHORT
                sl_tp = _SLTPOrder(
                    position_direction=position_dir,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    quantity=order.quantity,
                )
                self.sl_tp_orders.append(sl_tp)

                logger.debug(
                    "SL/TP registered: dir=%d SL=%s TP=%s qty=%d",
                    position_dir,
                    order.stop_loss,
                    order.take_profit,
                    order.quantity,
                )

    def process_bar(self, bar: MarketEvent) -> list[FillEvent]:
        """Execute pending orders and check SL/TP orders against the current bar.

        Processing order:
            1. Pending market/limit orders are filled first (at bar open + slippage).
            2. SL/TP orders are checked against bar high/low.
               - Stop-loss is checked before take-profit (conservative assumption).

        Args:
            bar: The current market event (new bar data).

        Returns:
            List of fill events generated during this bar.
        """
        fills: list[FillEvent] = []

        # ---- 1. Process pending orders ----
        orders_to_process = self.pending_orders[:]
        self.pending_orders.clear()

        for order in orders_to_process:
            fill = self._execute_order(order, bar)
            if fill is not None:
                fills.append(fill)

        # ---- 2. Check SL/TP orders ----
        sl_tp_to_remove: list[int] = []

        for i, sl_tp in enumerate(self.sl_tp_orders):
            fill = self._check_sl_tp(sl_tp, bar)
            if fill is not None:
                fills.append(fill)
                sl_tp_to_remove.append(i)

        # Remove triggered SL/TP orders (reverse order to preserve indices)
        for idx in reversed(sl_tp_to_remove):
            self.sl_tp_orders.pop(idx)

        return fills

    def clear_sl_tp(self) -> None:
        """Remove all active SL/TP orders.

        Called when a position is closed by a signal EXIT or by the engine
        at end-of-backtest, so stale SL/TP orders do not trigger.
        """
        self.sl_tp_orders.clear()

    # ------------------------------------------------------------------
    # Internal execution logic
    # ------------------------------------------------------------------

    def _execute_order(self, order: OrderEvent, bar: MarketEvent) -> FillEvent | None:
        """Execute a single pending order against the current bar.

        Market orders fill at bar.open with slippage applied adversely.
        Limit orders are not yet fully implemented (filled at limit price
        if the bar's range covers the limit).

        Returns:
            A FillEvent, or None if a limit order was not triggered.
        """
        if order.order_type == "MARKET":
            slippage_amount = self.slippage_ticks * self.tick_size

            if order.direction == SignalDirection.LONG:
                fill_price = bar.open + slippage_amount  # buy at worse (higher) price
            elif order.direction == SignalDirection.SHORT:
                fill_price = bar.open - slippage_amount  # sell at worse (lower) price
            else:
                # EXIT direction should have been converted to LONG/SHORT by portfolio
                fill_price = bar.open

            commission = self.commission * order.quantity
            slippage_cost = slippage_amount * order.quantity

            logger.debug(
                "MARKET fill: dir=%s qty=%d price=%.2f (open=%.2f slip=%.4f)",
                order.direction.name,
                order.quantity,
                fill_price,
                bar.open,
                slippage_amount,
            )

            return FillEvent(
                timestamp=bar.timestamp,
                direction=order.direction,
                quantity=order.quantity,
                fill_price=fill_price,
                commission=commission,
                slippage=slippage_cost,
            )

        elif order.order_type == "LIMIT":
            if order.limit_price is None:
                logger.warning("LIMIT order missing limit_price, skipping")
                return None

            # Check if the bar's range covers the limit price
            if bar.low <= order.limit_price <= bar.high:
                commission = self.commission * order.quantity
                return FillEvent(
                    timestamp=bar.timestamp,
                    direction=order.direction,
                    quantity=order.quantity,
                    fill_price=order.limit_price,
                    commission=commission,
                    slippage=0.0,
                )
            else:
                # Re-queue the limit order for the next bar
                self.pending_orders.append(order)
                return None

        return None

    def _check_sl_tp(
        self, sl_tp: _SLTPOrder, bar: MarketEvent
    ) -> FillEvent | None:
        """Check whether a SL/TP order is triggered by the current bar.

        For LONG positions:
            - SL triggers if bar.low <= stop_loss   -> fill at stop_loss (or bar.open if gap)
            - TP triggers if bar.high >= take_profit -> fill at take_profit (or bar.open if gap)

        For SHORT positions:
            - SL triggers if bar.high >= stop_loss   -> fill at stop_loss (or bar.open if gap)
            - TP triggers if bar.low <= take_profit   -> fill at take_profit (or bar.open if gap)

        Stop-loss is always checked first (conservative: assume worst case).

        Returns:
            A FillEvent if triggered, otherwise None.
        """
        if sl_tp.position_direction == 1:
            # LONG position
            # Check SL first
            if sl_tp.stop_loss is not None and bar.low <= sl_tp.stop_loss:
                # Gap check: if bar opens below SL, fill at open (worse)
                fill_price = min(sl_tp.stop_loss, bar.open)
                commission = self.commission * sl_tp.quantity
                logger.debug(
                    "LONG SL triggered at %.2f (bar low=%.2f)", fill_price, bar.low
                )
                return FillEvent(
                    timestamp=bar.timestamp,
                    direction=SignalDirection.SHORT,  # closing long -> sell
                    quantity=sl_tp.quantity,
                    fill_price=fill_price,
                    commission=commission,
                    slippage=0.0,
                )

            # Check TP
            if sl_tp.take_profit is not None and bar.high >= sl_tp.take_profit:
                # Gap check: if bar opens above TP, fill at open (better)
                fill_price = max(sl_tp.take_profit, bar.open)
                commission = self.commission * sl_tp.quantity
                logger.debug(
                    "LONG TP triggered at %.2f (bar high=%.2f)", fill_price, bar.high
                )
                return FillEvent(
                    timestamp=bar.timestamp,
                    direction=SignalDirection.SHORT,  # closing long -> sell
                    quantity=sl_tp.quantity,
                    fill_price=fill_price,
                    commission=commission,
                    slippage=0.0,
                )

        elif sl_tp.position_direction == -1:
            # SHORT position
            # Check SL first
            if sl_tp.stop_loss is not None and bar.high >= sl_tp.stop_loss:
                # Gap check: if bar opens above SL, fill at open (worse)
                fill_price = max(sl_tp.stop_loss, bar.open)
                commission = self.commission * sl_tp.quantity
                logger.debug(
                    "SHORT SL triggered at %.2f (bar high=%.2f)", fill_price, bar.high
                )
                return FillEvent(
                    timestamp=bar.timestamp,
                    direction=SignalDirection.LONG,  # closing short -> buy
                    quantity=sl_tp.quantity,
                    fill_price=fill_price,
                    commission=commission,
                    slippage=0.0,
                )

            # Check TP
            if sl_tp.take_profit is not None and bar.low <= sl_tp.take_profit:
                # Gap check: if bar opens below TP, fill at open (better)
                fill_price = min(sl_tp.take_profit, bar.open)
                commission = self.commission * sl_tp.quantity
                logger.debug(
                    "SHORT TP triggered at %.2f (bar low=%.2f)", fill_price, bar.low
                )
                return FillEvent(
                    timestamp=bar.timestamp,
                    direction=SignalDirection.LONG,  # closing short -> buy
                    quantity=sl_tp.quantity,
                    fill_price=fill_price,
                    commission=commission,
                    slippage=0.0,
                )

        return None
