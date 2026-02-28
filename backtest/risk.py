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

    Supports evaluation-phase pass detection and trailing drawdown
    floor lock-in after passing (floor rises to account baseline).
    """

    def __init__(
        self,
        rules: TopStepAccountRules,
        initial_capital: float,
        best_day_limit: float = 0.0,
    ) -> None:
        self.rules = rules
        self.initial_capital = initial_capital
        self.peak_equity: float = initial_capital

        # Pass detection
        self.passed: bool = False
        self.pass_timestamp: datetime | None = None
        self.pass_bar_index: int | None = None

        # Trailing DD floor: starts at (capital - max_loss).
        # After passing, the floor rises to account baseline and locks.
        self._dd_floor: float = initial_capital - rules.max_loss_limit

        # Parse must_close_by time
        parts = self.rules.must_close_by.split(":")
        self._close_by_time = time(int(parts[0]), int(parts[1]))

        # Best Day tracking — pause trading if single-day profit exceeds limit
        self.best_day_limit: float = best_day_limit
        self._day_start_equity: float = initial_capital
        self._current_day: str = ""
        self._day_paused: bool = False  # True when today's profit >= limit
        self.best_day_pauses: list[str] = []  # dates when trading was paused

        # Circuit breaker tracking
        self.circuit_breaker_blocks: int = 0

    def check(
        self,
        equity: float,
        position: int,
        timestamp: datetime,
        bar_index: int = -1,
    ) -> list[str]:
        """Check all risk rules and return a list of violation messages."""
        violations: list[str] = []

        # Update peak equity (trailing high-water mark)
        if equity > self.peak_equity:
            self.peak_equity = equity

        # Update trailing DD floor (it trails UP with peak equity)
        new_floor = self.peak_equity - self.rules.max_loss_limit
        if new_floor > self._dd_floor:
            if self.passed and new_floor > self.initial_capital:
                # After passing, floor locks at account baseline
                self._dd_floor = self.initial_capital
            else:
                self._dd_floor = new_floor

        # Check if profit target reached (evaluation pass)
        if not self.passed and self.rules.profit_target > 0:
            profit = equity - self.initial_capital
            if profit >= self.rules.profit_target:
                self.passed = True
                self.pass_timestamp = timestamp
                self.pass_bar_index = bar_index
                # Lock DD floor at account baseline
                self._dd_floor = max(self._dd_floor, self.initial_capital)
                logger.info(
                    "PASSED evaluation at bar %d, equity=$%.2f, "
                    "DD floor locked at $%.2f",
                    bar_index, equity, self._dd_floor,
                )

        # --- 1. Trailing max loss (using floor) ---
        if equity < self._dd_floor:
            drawdown = self.peak_equity - equity
            msg = (
                f"TRAILING MAX LOSS BREACH: equity ${equity:,.2f} "
                f"< floor ${self._dd_floor:,.2f} "
                f"(peak=${self.peak_equity:,.2f}, dd=${drawdown:,.2f})"
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

        # --- 3. Must close by time (only check, engine does force-close) ---
        # This is now mainly a diagnostic — the engine should proactively
        # force-close BEFORE this bar via should_force_flat_now().

        return violations

    def should_force_flat_now(self, timestamp: datetime, position: int) -> bool:
        """Return True if the engine should force-close all positions NOW.

        This is checked BEFORE the strategy signal, so the force-close
        happens proactively and no time violation is logged.

        Only triggers between must_close_by (15:10) and the evening
        session open (17:00).  After 17:00 the Globex session is active
        and positions are allowed again.
        """
        if position == 0:
            return False
        bar_time = timestamp.time()
        evening_open = time(17, 0)
        return self._close_by_time <= bar_time < evening_open

    def should_force_close(self, violations: list[str]) -> bool:
        """Determine whether any violation requires immediate position closure."""
        return len(violations) > 0

    def get_trailing_drawdown(self, equity: float) -> float:
        """Return the current trailing drawdown in dollars."""
        return max(0.0, self.peak_equity - equity)

    def get_remaining_loss_budget(self, equity: float) -> float:
        """Return how much more the account can lose before breaching."""
        return max(0.0, equity - self._dd_floor)

    # ------------------------------------------------------------------
    # Pre-trade safety checks
    # ------------------------------------------------------------------

    def check_order_safe(
        self, order: OrderEvent, equity: float, point_value: float,
    ) -> tuple[bool, str]:
        """Circuit breaker stub (use check_order_safe_with_entry instead)."""
        if order.stop_loss is None:
            return True, ""
        return True, ""

    def check_order_safe_with_entry(
        self,
        order: OrderEvent,
        entry_price: float,
        equity: float,
        point_value: float,
    ) -> tuple[bool, str]:
        """Circuit breaker with known entry price."""
        if order.stop_loss is None:
            return True, ""

        sl_distance = abs(entry_price - order.stop_loss)
        potential_loss = sl_distance * point_value * order.quantity

        remaining = self.get_remaining_loss_budget(equity)

        if potential_loss >= remaining:
            self.circuit_breaker_blocks += 1
            reason = (
                f"CIRCUIT BREAKER: potential SL loss ${potential_loss:,.2f} "
                f">= remaining budget ${remaining:,.2f} "
                f"(SL dist={sl_distance:.2f} pts, qty={order.quantity}, "
                f"point_val=${point_value:.2f}). Order SKIPPED."
            )
            logger.warning(reason)
            return False, reason

        return True, ""

    def update_day(self, timestamp: datetime, equity: float) -> None:
        """Track daily P&L for Best Day rule. Call at start of each bar."""
        day_str = timestamp.strftime("%Y-%m-%d")
        if day_str != self._current_day:
            # New trading day — reset
            self._current_day = day_str
            self._day_start_equity = equity
            self._day_paused = False

    def is_best_day_exceeded(self, equity: float) -> bool:
        """Return True if today's realized+unrealized profit exceeds the limit."""
        if self.best_day_limit <= 0:
            return False
        if self._day_paused:
            return True
        day_profit = equity - self._day_start_equity
        if day_profit >= self.best_day_limit:
            self._day_paused = True
            self.best_day_pauses.append(self._current_day)
            logger.info(
                "BEST DAY LIMIT: day %s profit $%.2f >= limit $%.2f. "
                "Trading paused for rest of day.",
                self._current_day, day_profit, self.best_day_limit,
            )
            return True
        return False

    def is_in_no_entry_window(
        self,
        timestamp: datetime,
        buffer_minutes: int = 20,
    ) -> bool:
        """Check if the current time is within the no-new-entry window."""
        bar_time = timestamp.time()

        close_dt = datetime.combine(timestamp.date(), self._close_by_time)
        cutoff_dt = close_dt - timedelta(minutes=buffer_minutes)
        cutoff_time = cutoff_dt.time()

        if cutoff_time < self._close_by_time:
            # Only block entries in the pre-close window (e.g. 14:50–15:10).
            # Evening session (17:00+) must NOT be blocked.
            return cutoff_time <= bar_time < self._close_by_time
        else:
            return False
