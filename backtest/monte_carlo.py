"""
Monte Carlo simulation for backtest trade sequence analysis.

Shuffles the order of completed trades N times, rebuilding equity curves
each time to assess strategy robustness and estimate ruin probability.

Usage:
    from backtest.monte_carlo import run_monte_carlo
    mc = run_monte_carlo(trades_pnl=[100, -50, 200, ...], initial_capital=50000)
    print(mc.ruin_probability)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloResult:
    """Results from a Monte Carlo trade-shuffle simulation.

    Attributes:
        n_simulations:      Number of shuffle iterations performed.
        n_trades:           Number of trades in the original sequence.
        initial_capital:    Starting account balance.
        loss_limit:         Account ruin threshold (trailing max loss).
        percentile_curves:  Equity curves at key percentiles.
                            Keys: 5, 25, 50, 75, 95.
        final_equity_stats: Distribution stats for final equity values.
        max_drawdown_stats: Distribution stats for max drawdown per sim.
        ruin_probability:   Fraction of simulations that hit the loss limit.
        max_consec_loss_stats: Distribution stats for max consecutive losses.
        original_pnl:       Original trade PnL sequence (for reference).
    """

    n_simulations: int
    n_trades: int
    initial_capital: float
    loss_limit: float
    percentile_curves: dict[int, list[float]]
    final_equity_stats: dict
    max_drawdown_stats: dict
    ruin_probability: float
    max_consec_loss_stats: dict
    original_pnl: list[float] = field(default_factory=list)


def run_monte_carlo(
    trades_pnl: list[float],
    initial_capital: float = 50_000.0,
    loss_limit: float = 2_000.0,
    n_simulations: int = 1000,
    seed: int | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo trade-order simulation.

    Takes a sequence of trade P&L values, shuffles the order N times,
    and reconstructs equity curves to measure:
    - Best/worst/median equity paths
    - Probability of account ruin (trailing drawdown hitting loss limit)
    - Distribution of max drawdowns and consecutive losses

    Args:
        trades_pnl:     List of net P&L values per trade (positive = win).
        initial_capital: Starting account balance.
        loss_limit:     Maximum trailing drawdown before account is blown.
        n_simulations:  Number of random shuffles to perform.
        seed:           Random seed for reproducibility.

    Returns:
        MonteCarloResult with all statistics and percentile curves.
    """
    if not trades_pnl:
        logger.warning("No trades provided for Monte Carlo simulation.")
        return MonteCarloResult(
            n_simulations=0,
            n_trades=0,
            initial_capital=initial_capital,
            loss_limit=loss_limit,
            percentile_curves={p: [] for p in (5, 25, 50, 75, 95)},
            final_equity_stats=_empty_stats(),
            max_drawdown_stats=_empty_stats(),
            ruin_probability=0.0,
            max_consec_loss_stats=_empty_stats(),
        )

    rng = np.random.default_rng(seed)
    pnl_array = np.array(trades_pnl, dtype=np.float64)
    n_trades = len(pnl_array)

    logger.info(
        "Monte Carlo: %d simulations x %d trades (capital=$%.0f, loss_limit=$%.0f)",
        n_simulations, n_trades, initial_capital, loss_limit,
    )

    # Pre-allocate arrays for all simulations
    # Each row = one simulation's equity curve (n_trades + 1 points including start)
    all_equity = np.zeros((n_simulations, n_trades + 1), dtype=np.float64)
    all_equity[:, 0] = initial_capital

    final_equities = np.zeros(n_simulations, dtype=np.float64)
    max_drawdowns = np.zeros(n_simulations, dtype=np.float64)
    max_consec_losses = np.zeros(n_simulations, dtype=np.int64)
    ruin_count = 0

    for sim in range(n_simulations):
        # Shuffle trade order
        shuffled = pnl_array.copy()
        rng.shuffle(shuffled)

        # Build equity curve
        equity = initial_capital
        peak = initial_capital
        max_dd = 0.0
        consec_loss = 0
        max_cl = 0
        blown = False

        for t in range(n_trades):
            if blown:
                # Account blown: no more trades, equity stays flat
                all_equity[sim, t + 1] = equity
                continue

            equity += shuffled[t]
            all_equity[sim, t + 1] = equity

            # Track peak and drawdown (trailing)
            if equity > peak:
                peak = equity
            dd = peak - equity
            if dd > max_dd:
                max_dd = dd

            # Check ruin — circuit breaker: stop trading
            if dd >= loss_limit:
                blown = True

            # Consecutive losses
            if shuffled[t] < 0:
                consec_loss += 1
                if consec_loss > max_cl:
                    max_cl = consec_loss
            else:
                consec_loss = 0

        final_equities[sim] = equity
        max_drawdowns[sim] = max_dd
        max_consec_losses[sim] = max_cl
        if blown:
            ruin_count += 1

    # Compute percentile curves
    percentiles = {5: None, 25: None, 50: None, 75: None, 95: None}
    for pct in percentiles:
        percentiles[pct] = np.percentile(all_equity, pct, axis=0).tolist()

    ruin_prob = ruin_count / n_simulations

    logger.info(
        "Monte Carlo complete: ruin_prob=%.1f%%, median_final=$%.0f, "
        "mean_dd=$%.0f, worst_dd=$%.0f",
        ruin_prob * 100,
        np.median(final_equities),
        np.mean(max_drawdowns),
        np.max(max_drawdowns),
    )

    return MonteCarloResult(
        n_simulations=n_simulations,
        n_trades=n_trades,
        initial_capital=initial_capital,
        loss_limit=loss_limit,
        percentile_curves=percentiles,
        final_equity_stats=_compute_stats(final_equities),
        max_drawdown_stats=_compute_stats(max_drawdowns),
        ruin_probability=ruin_prob,
        max_consec_loss_stats=_compute_stats(max_consec_losses.astype(np.float64)),
        original_pnl=trades_pnl,
    )


def _compute_stats(arr: np.ndarray) -> dict:
    """Compute summary statistics for a 1-D array."""
    if len(arr) == 0:
        return _empty_stats()
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
    }


def _empty_stats() -> dict:
    """Return zeroed stats dict."""
    return {
        "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0,
        "p5": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0,
    }
