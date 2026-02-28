"""
Performance metrics computation for backtesting results.

Computes a comprehensive set of trading statistics from the trade log
and equity curve produced by the backtesting engine.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from data.models import EquityPoint, Trade


def compute_metrics(
    trades: list[Trade],
    equity_curve: list[EquityPoint],
    initial_capital: float,
) -> dict:
    """Compute all performance metrics from completed trades and equity curve.

    Args:
        trades:          List of completed round-trip Trade records.
        equity_curve:    List of EquityPoint snapshots (one per bar).
        initial_capital: Starting account balance.

    Returns:
        Dictionary of metric name -> value.
    """
    result: dict = {}

    # ------------------------------------------------------------------
    # Trade-based metrics
    # ------------------------------------------------------------------

    total_trades = len(trades)
    result["total_trades"] = total_trades

    if total_trades == 0:
        return _empty_metrics(initial_capital)

    pnl_list = [t.pnl for t in trades]
    winning = [t for t in trades if t.pnl > 0]
    losing = [t for t in trades if t.pnl <= 0]

    win_pnls = [t.pnl for t in winning]
    loss_pnls = [t.pnl for t in losing]

    gross_profit = sum(win_pnls) if win_pnls else 0.0
    gross_loss = sum(loss_pnls) if loss_pnls else 0.0
    net_profit = sum(pnl_list)

    result["net_profit"] = net_profit
    result["net_profit_pct"] = (net_profit / initial_capital) * 100.0
    result["gross_profit"] = gross_profit
    result["gross_loss"] = gross_loss
    result["winning_trades"] = len(winning)
    result["losing_trades"] = len(losing)
    result["win_rate"] = (len(winning) / total_trades) * 100.0 if total_trades > 0 else 0.0

    # Profit factor
    if gross_loss != 0.0:
        result["profit_factor"] = gross_profit / abs(gross_loss)
    else:
        result["profit_factor"] = float("inf") if gross_profit > 0 else 0.0

    # Average trade
    result["avg_trade"] = net_profit / total_trades
    result["avg_trade_pct"] = (result["avg_trade"] / initial_capital) * 100.0

    # Largest win / loss
    result["largest_win"] = max(win_pnls) if win_pnls else 0.0
    result["largest_loss"] = min(loss_pnls) if loss_pnls else 0.0

    # Average winning / losing trade
    result["avg_winning_trade"] = (
        (gross_profit / len(winning)) if winning else 0.0
    )
    result["avg_losing_trade"] = (
        (gross_loss / len(losing)) if losing else 0.0
    )

    # Average bars in trade
    bars_held = [t.bars_held for t in trades]
    result["avg_bars_in_trade"] = sum(bars_held) / total_trades if total_trades > 0 else 0.0

    # Consecutive wins / losses
    result["max_consecutive_wins"] = _max_consecutive(trades, winner=True)
    result["max_consecutive_losses"] = _max_consecutive(trades, winner=False)

    # Expectancy: avg_win * win_rate - avg_loss * loss_rate
    win_rate_frac = len(winning) / total_trades if total_trades > 0 else 0.0
    loss_rate_frac = len(losing) / total_trades if total_trades > 0 else 0.0
    avg_win = result["avg_winning_trade"]
    avg_loss = abs(result["avg_losing_trade"])
    result["expectancy"] = (avg_win * win_rate_frac) - (avg_loss * loss_rate_frac)

    # Best Day / Consistency Rule (TopStep)
    daily_pnl = _compute_daily_pnl(trades)
    if daily_pnl:
        result["best_day_pnl"] = max(daily_pnl.values())
        result["worst_day_pnl"] = min(daily_pnl.values())
        if net_profit > 0:
            result["best_day_pct_of_profit"] = (result["best_day_pnl"] / net_profit) * 100.0
        else:
            result["best_day_pct_of_profit"] = 0.0
        result["trading_days"] = len(daily_pnl)
        # Payout readiness (TopStep)
        sorted_days = sorted(daily_pnl.items())
        consec_150 = 0
        max_consec_150 = 0
        for _, day_pnl in sorted_days:
            if day_pnl >= 150:
                consec_150 += 1
                max_consec_150 = max(max_consec_150, consec_150)
            else:
                consec_150 = 0
        result["max_consec_150_days"] = max_consec_150
        result["payout_standard_ready"] = max_consec_150 >= 5
        result["payout_xfa_ready"] = (
            len(daily_pnl) >= 3
            and net_profit > 0
            and result.get("best_day_pct_of_profit", 100) <= 60
        )
    else:
        result["best_day_pnl"] = 0.0
        result["worst_day_pnl"] = 0.0
        result["best_day_pct_of_profit"] = 0.0
        result["trading_days"] = 0
        result["max_consec_150_days"] = 0
        result["payout_standard_ready"] = False
        result["payout_xfa_ready"] = False

    # ------------------------------------------------------------------
    # Equity-curve-based metrics
    # ------------------------------------------------------------------

    if len(equity_curve) > 1:
        equities = np.array([ep.equity for ep in equity_curve], dtype=np.float64)

        # Max drawdown
        max_dd, max_dd_pct = _compute_max_drawdown(equities)
        result["max_drawdown"] = max_dd
        result["max_drawdown_pct"] = max_dd_pct

        # Sharpe ratio (annualized, assume 252 trading days)
        result["sharpe_ratio"] = _compute_sharpe(equities, periods_per_year=252)
    else:
        result["max_drawdown"] = 0.0
        result["max_drawdown_pct"] = 0.0
        result["sharpe_ratio"] = 0.0

    return result


def _compute_daily_pnl(trades: list[Trade]) -> dict[str, float]:
    """Aggregate trade PnL by calendar date.

    Returns dict of date_str -> total_pnl for that day.
    """
    daily: dict[str, float] = {}
    for t in trades:
        day_key = t.exit_time.strftime("%Y-%m-%d") if hasattr(t.exit_time, "strftime") else str(t.exit_time)[:10]
        daily[day_key] = daily.get(day_key, 0.0) + t.pnl
    return daily


def _empty_metrics(initial_capital: float) -> dict:
    """Return a metrics dict with zero values for when there are no trades."""
    return {
        "total_trades": 0,
        "net_profit": 0.0,
        "net_profit_pct": 0.0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "avg_trade": 0.0,
        "avg_trade_pct": 0.0,
        "largest_win": 0.0,
        "largest_loss": 0.0,
        "avg_winning_trade": 0.0,
        "avg_losing_trade": 0.0,
        "avg_bars_in_trade": 0.0,
        "max_consecutive_wins": 0,
        "max_consecutive_losses": 0,
        "expectancy": 0.0,
        "max_drawdown": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe_ratio": 0.0,
        "best_day_pnl": 0.0,
        "worst_day_pnl": 0.0,
        "best_day_pct_of_profit": 0.0,
        "trading_days": 0,
        "max_consec_150_days": 0,
        "payout_standard_ready": False,
        "payout_xfa_ready": False,
    }


def _compute_max_drawdown(equities: np.ndarray) -> tuple[float, float]:
    """Compute the maximum peak-to-trough drawdown.

    Args:
        equities: Array of equity values over time.

    Returns:
        Tuple of (max_drawdown_dollars, max_drawdown_percent).
    """
    peak = np.maximum.accumulate(equities)
    drawdowns = peak - equities
    max_dd = float(np.max(drawdowns))

    # Percentage drawdown relative to the peak at the time
    with np.errstate(divide="ignore", invalid="ignore"):
        dd_pct = np.where(peak > 0, drawdowns / peak * 100.0, 0.0)
    max_dd_pct = float(np.max(dd_pct))

    return max_dd, max_dd_pct


def _compute_sharpe(
    equities: np.ndarray,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> float:
    """Compute the annualized Sharpe ratio from equity series.

    Uses log returns for better statistical properties.

    Args:
        equities:         Array of equity values.
        periods_per_year: Number of bars per year (252 for daily bars).
        risk_free_rate:   Annualized risk-free rate (default 0).

    Returns:
        Annualized Sharpe ratio, or 0.0 if not computable.
    """
    if len(equities) < 2:
        return 0.0

    # Filter out any zero or negative equity values for log returns
    valid = equities[equities > 0]
    if len(valid) < 2:
        return 0.0

    # Compute period returns
    returns = np.diff(valid) / valid[:-1]

    if len(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)

    if std_return == 0.0 or math.isnan(std_return):
        return 0.0

    # Annualize
    risk_free_per_period = risk_free_rate / periods_per_year
    sharpe = (mean_return - risk_free_per_period) / std_return
    annualized_sharpe = sharpe * math.sqrt(periods_per_year)

    return float(annualized_sharpe)


def _max_consecutive(trades: list[Trade], winner: bool) -> int:
    """Count the maximum consecutive winning or losing trades.

    Args:
        trades: List of Trade records in chronological order.
        winner: If True, count consecutive winners; if False, count losers.

    Returns:
        Maximum streak length.
    """
    max_streak = 0
    current_streak = 0

    for trade in trades:
        is_win = trade.pnl > 0
        if is_win == winner:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak
