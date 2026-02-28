"""
TopStep funded account rules and constraints.

These rules are enforced by the RiskManager during backtesting to ensure
strategies comply with TopStep's requirements before going live.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TopStepAccountRules:
    """Constraints for a specific TopStep account tier.

    Attributes:
        account_size:    Starting account balance in dollars.
        max_loss_limit:  Trailing maximum drawdown allowed before account breach.
        max_contracts:   Maximum number of contracts that can be held simultaneously.
        must_close_by:   Time (CT / Central Time) by which all positions must be flat.
    """

    account_size: int
    max_loss_limit: float
    max_contracts: int
    must_close_by: str = "15:10"
    profit_target: float = 0.0  # profit needed to pass evaluation


TOPSTEP_ACCOUNTS: dict[str, TopStepAccountRules] = {
    "50K": TopStepAccountRules(
        account_size=50_000,
        max_loss_limit=2_000.0,
        max_contracts=5,
        profit_target=3_000.0,
    ),
    "100K": TopStepAccountRules(
        account_size=100_000,
        max_loss_limit=3_000.0,
        max_contracts=10,
        profit_target=6_000.0,
    ),
    "150K": TopStepAccountRules(
        account_size=150_000,
        max_loss_limit=4_500.0,
        max_contracts=15,
        profit_target=9_000.0,
    ),
}
