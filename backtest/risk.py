"""
Risk manager for the backtesting engine.

Enforces TopStep funded account rules during backtesting by checking
trailing drawdown, position size limits, and time-of-day constraints.
"""

from __future__ import annotations

import logging
from datetime import datetime, time

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
