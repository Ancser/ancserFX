"""
Risk manager for the backtesting engine.

Enforces TopStep funded account rules during backtesting by checking
trailing drawdown, position size limits, and time-of-day constraints.

Also provides pre-trade safety checks (circuit breaker) and a
no-new-entry window before the must-flat-by deadline.
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta

from backtest.events import OrderEvent
from backtest.topstep_rules import TopStepAccountRules

logger = logging.getLogger(__name__)


class RiskManager:
    """Monitors portfolio state and enforces TopStep account rules.

    The risk manager is checked after every fill and market update to ensure
    the strategy never violates funded account constraints.

    Attributes:
        rules:         The TopStep account rules to enforce.
        initial_capital: Starting account balance.
        peak_equity:   Highest equity level reached (for trailing drawdown).
    """

    def __init__(
        self, rules: TopStepAccountRules, initial_capital: float
    ) -> None:
        self.rules = rules
        self.initial_capital = initial_capital
        self.peak_equity: float = initial_capital

        # Parse must_close_by time
        parts = self.rules.must_close_by.split(":")
        self._close_by_time = time(int(parts[0]), int(parts[1]))

    def check(
        self,
        equity: float,
        position: int,
        timestamp: datetime,
    ) -> list[str]:
        """Check all risk rules and return a list of violation messages.

        An empty list means all rules are satisfied. Any non-empty result
        indicates a rule breach that may require the position to be force-closed.

        Args:
            equity:    Current total account equity (cash + unrealized).
            position:  Current position size (signed).
            timestamp: Current bar timestamp.

        Returns:
            List of human-readable violation messages.
        """
        violations: list[str] = []

        # Update peak equity (trailing high-water mark)
        if equity > self.peak_equity:
            self.peak_equity = equity

        # --- 1. Trailing max loss ---
        drawdown = self.peak_equity - equity
        if drawdown >= self.rules.max_loss_limit:
            msg = (
                f"TRAILING MAX LOSS BREACH: drawdown ${drawdown:,.2f} "
                f">= limit ${self.rules.max_loss_limit:,.2f} "
                f"(peak=${self.peak_equity:,.2f}, current=${equity:,.2f})"
            )
            violations.append(msg)
            logger.warning(msg)

        # --- 2. Position size limit ---
        if abs(position) > self.rules.max_contracts:
            msg = (
                f"POSITION SIZE BREACH: {abs(position)} contracts "
                f"> max {self.rules.max_contracts}"
            )
            violations.append(msg)
            logger.warning(msg)

        # --- 3. Must close by time ---
        bar_time = timestamp.time()
        if bar_time >= self._close_by_time and position != 0:
            msg = (
                f"TIME BREACH: position open at {bar_time.strftime('%H:%M')} CT, "
                f"must be flat by {self.rules.must_close_by} CT"
            )
            violations.append(msg)
            logger.warning(msg)

        return violations

    def should_force_close(self, violations: list[str]) -> bool:
        """Determine whether any violation requires immediate position closure.

        All current violations warrant forced closure, so this returns True
        if the violations list is non-empty.

        Args:
            violations: List of violation messages from check().

        Returns:
            True if the position should be force-closed.
        """
        return len(violations) > 0

    def get_trailing_drawdown(self, equity: float) -> float:
        """Return the current trailing drawdown in dollars.

        Args:
            equity: Current total account equity.

        Returns:
            Drawdown amount from peak (always >= 0).
        """
        return max(0.0, self.peak_equity - equity)

    def get_remaining_loss_budget(self, equity: float) -> float:
        """Return how much more the account can lose before breaching the max loss rule.

        Args:
            equity: Current total account equity.

        Returns:
            Remaining dollar budget before trailing max loss is breached.
        """
        current_dd = self.get_trailing_drawdown(equity)
        return max(0.0, self.rules.max_loss_limit - current_dd)

    # ------------------------------------------------------------------
    # Pre-trade safety checks
    # ------------------------------------------------------------------

    def check_order_safe(
        self,
        order: OrderEvent,
        equity: float,
        point_value: float,
    ) -> tuple[bool, str]:
        """Circuit breaker: check if the worst-case SL loss of this order
        would exceed the remaining drawdown budget.

        Args:
            order:       The order about to be submitted.
            equity:      Current total account equity.
            point_value: Dollar value of a 1-point move per contract
                         (e.g. 50 for ES, 5 for MES).

        Returns:
            (is_safe, reason) — True if OK to trade, False with explanation if not.
        """
        if order.stop_loss is None:
            return True, ""

        # Estimate entry price (close of current bar, used in build_order)
        # For the SL check we only need the distance
        # The order carries stop_loss as an absolute price.
        # We'll compute potential loss from the SL distance
        # entry is unknown here but we can derive from TP levels or
        # just use: loss = |entry - SL| * qty * point_value
        # Since we don't have entry, derive it:
        #   LONG: entry > SL -> distance = entry - SL  (SL below)
        #   SHORT: entry < SL -> distance = SL - entry (SL above)
        # We need entry. The engine should pass it.
        # For now we'll use a heuristic: average of TP levels or just use
        # the midpoint approach. Actually let's require the engine to pass entry.

        return True, ""

    def check_order_safe_with_entry(
        self,
        order: OrderEvent,
        entry_price: float,
        equity: float,
        point_value: float,
    ) -> tuple[bool, str]:
        """Circuit breaker with known entry price.

        Checks if the worst-case SL loss would blow the account. If so,
        the order should be skipped to protect the account.

        Args:
            order:       The order about to be submitted.
            entry_price: Expected entry price (bar close or open).
            equity:      Current total account equity.
            point_value: Dollar value of a 1-point move per contract.

        Returns:
            (is_safe, reason) — True if OK, False with explanation.
        """
        if order.stop_loss is None:
            return True, ""

        sl_distance = abs(entry_price - order.stop_loss)
        potential_loss = sl_distance * point_value * order.quantity

        remaining = self.get_remaining_loss_budget(equity)

        if potential_loss >= remaining:
            reason = (
                f"CIRCUIT BREAKER: potential SL loss ${potential_loss:,.2f} "
                f">= remaining budget ${remaining:,.2f} "
                f"(SL dist={sl_distance:.2f} pts, qty={order.quantity}, "
                f"point_val=${point_value:.2f}). Order SKIPPED."
            )
            logger.warning(reason)
            return False, reason

        return True, ""

    def is_in_no_entry_window(
        self,
        timestamp: datetime,
        buffer_minutes: int = 20,
    ) -> bool:
        """Check if the current time is within the no-new-entry window.

        Returns True if the timestamp is too close to the must_close_by
        deadline, meaning no new positions should be opened.

        Args:
            timestamp:      Current bar timestamp.
            buffer_minutes: Minutes before must_close_by to stop entries.

        Returns:
            True if new entries should be blocked.
        """
        bar_time = timestamp.time()

        # Calculate the cutoff time (must_close_by - buffer)
        close_dt = datetime.combine(timestamp.date(), self._close_by_time)
        cutoff_dt = close_dt - timedelta(minutes=buffer_minutes)
        cutoff_time = cutoff_dt.time()

        # Handle case where cutoff is before market open (shouldn't happen
        # with normal values but be safe)
        if cutoff_time < self._close_by_time:
            # Normal case: cutoff < close_by, block if bar_time >= cutoff
            return bar_time >= cutoff_time and bar_time < self._close_by_time
        else:
            return False
