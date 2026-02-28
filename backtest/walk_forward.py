"""
Walk-Forward Analysis (WFA) for backtesting strategies.

Splits historical data into rolling train/test windows, optimizes parameters
on each train period, then validates on the unseen test period.  The final
result stitches all out-of-sample test segments together for an honest
performance evaluation free of look-ahead bias.

Usage:
    from backtest.walk_forward import run_walk_forward, WalkForwardConfig
    from backtest.engine import BacktestConfig

    base = BacktestConfig(strategy_name="Delta Momentum", instrument="MNQ", ...)
    wfa_cfg = WalkForwardConfig(base_config=base, train_days=180, test_days=30)
    result = run_walk_forward(wfa_cfg)
    print(result.wf_efficiency, result.stitched_metrics)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, BacktestEngine
from backtest.metrics import compute_metrics
from backtest.optimizer import run_optimization
from backtest.topstep_rules import TOPSTEP_ACCOUNTS
from data.models import Trade
from data.store import DataStore

logger = logging.getLogger(__name__)

# Approximate bars-per-trading-day by timeframe
_BARS_PER_DAY: dict[str, float] = {
    "1min": 840,   # ~14 hours futures session
    "5min": 168,
    "15min": 56,
    "1h": 14,
    "daily": 1,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardConfig:
    """Configuration for a walk-forward analysis run."""

    base_config: BacktestConfig

    # Window geometry (calendar days)
    train_days: int = 180
    test_days: int = 30
    step_days: int = 30
    warmup_bars: int = 200

    # Optimization settings per window
    opt_iterations: int = 50
    opt_target_metric: str = "sharpe_ratio"
    opt_min_trades: int = 5
    opt_seed: int | None = 42

    # Overall date range (None = use full data range)
    start_date: str | None = None
    end_date: str | None = None


@dataclass
class WalkForwardWindow:
    """Result from a single train/test window."""

    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str

    # Train (optimization) output
    best_params: dict[str, Any]
    train_metrics: dict
    train_n_valid: int
    train_target_value: float

    # Test (out-of-sample) output
    test_metrics: dict
    test_trades: list[dict]
    test_equity_curve: list[dict]

    # Derived
    efficiency_ratio: float
    opt_elapsed_sec: float = 0.0
    test_elapsed_sec: float = 0.0


@dataclass
class WalkForwardResult:
    """Aggregate results from a complete walk-forward analysis."""

    config: WalkForwardConfig
    windows: list[WalkForwardWindow]

    # Stitched out-of-sample results
    stitched_metrics: dict
    stitched_trades: list[dict]
    stitched_equity_curve: list[dict]

    # Aggregate quality metrics
    wf_efficiency: float
    param_stability: dict[str, dict]
    n_windows: int
    n_profitable_windows: int
    window_consistency: float

    total_elapsed_sec: float = 0.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _generate_windows(
    config: WalkForwardConfig,
    data_start: datetime,
    data_end: datetime,
) -> list[tuple[str, str, str, str]]:
    """Generate (train_start, train_end, test_start, test_end) tuples."""

    overall_start = pd.to_datetime(config.start_date) if config.start_date else data_start
    overall_end = pd.to_datetime(config.end_date) if config.end_date else data_end

    windows: list[tuple[str, str, str, str]] = []
    cursor = overall_start

    while True:
        train_start = cursor
        train_end = cursor + timedelta(days=config.train_days - 1)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=config.test_days - 1)

        if test_end > overall_end:
            break

        windows.append((
            train_start.strftime("%Y-%m-%d"),
            train_end.strftime("%Y-%m-%d"),
            test_start.strftime("%Y-%m-%d"),
            test_end.strftime("%Y-%m-%d"),
        ))

        cursor += timedelta(days=config.step_days)

    return windows


def _compute_warmup_start(
    test_start: str,
    warmup_bars: int,
    timeframe: str,
) -> str:
    """Estimate the date to start loading data so warmup_bars exist before test_start."""

    bars_per_day = _BARS_PER_DAY.get(timeframe, 168)  # default to 5min
    trading_days_needed = math.ceil(warmup_bars / bars_per_day)
    # Calendar days (add buffer for weekends/holidays)
    calendar_days = int(trading_days_needed * 7 / 5) + 5

    ts = pd.to_datetime(test_start)
    warmup_start = ts - timedelta(days=calendar_days)
    return warmup_start.strftime("%Y-%m-%d")


def _stitch_equity_curves(
    windows: list[WalkForwardWindow],
    initial_capital: float,
) -> list[dict]:
    """Chain per-window equity curves into one continuous curve."""

    stitched: list[dict] = []
    running_equity = initial_capital

    for w in windows:
        if not w.test_equity_curve:
            continue

        # Each window's equity starts at initial_capital; offset to continue
        offset = running_equity - initial_capital

        for point in w.test_equity_curve:
            stitched.append({
                "timestamp": point["timestamp"],
                "bar_index": point.get("bar_index", 0),
                "equity": point["equity"] + offset,
                "cash": point.get("cash", 0) + offset,
                "unrealized_pnl": point.get("unrealized_pnl", 0),
                "window_index": w.window_index,
            })

        # Update running equity to end of this window
        last_equity = w.test_equity_curve[-1]["equity"]
        running_equity = last_equity + offset

    return stitched


def _compute_param_stability(windows: list[WalkForwardWindow]) -> dict[str, dict]:
    """Compute per-parameter stability metrics across windows."""

    if not windows:
        return {}

    all_param_names = set()
    for w in windows:
        all_param_names.update(w.best_params.keys())

    stability: dict[str, dict] = {}
    for pname in sorted(all_param_names):
        values = [w.best_params.get(pname) for w in windows if pname in w.best_params]
        numeric_values = [v for v in values if isinstance(v, (int, float))]

        if numeric_values:
            arr = np.array(numeric_values, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr))
            cv = std / abs(mean) if abs(mean) > 1e-9 else 0.0
            stability[pname] = {
                "values": numeric_values,
                "mean": round(mean, 4),
                "std": round(std, 4),
                "cv": round(cv, 4),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
            }
        else:
            stability[pname] = {
                "values": values,
                "mean": 0.0,
                "std": 0.0,
                "cv": 0.0,
                "min": 0.0,
                "max": 0.0,
            }

    return stability


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_walk_forward(
    config: WalkForwardConfig,
    progress_callback: Any | None = None,
) -> WalkForwardResult:
    """Execute a full walk-forward analysis.

    For each window:
      1. Run optimizer on train period -> best_params
      2. Run backtest with best_params on test period (with warmup)
      3. Filter trades to only those within actual test date range

    Then stitch all test results and compute aggregate metrics.

    Args:
        config:            Walk-forward configuration.
        progress_callback: Optional callable(current_window, total_windows, status_str).

    Returns:
        WalkForwardResult with per-window and aggregate results.
    """

    t0 = time.perf_counter()
    base = config.base_config

    # --- 1. Resolve data date range ---
    store = DataStore()
    data_start, data_end, bar_count = store.get_date_range(
        base.instrument, base.timeframe
    )
    logger.info(
        "WFA: data range %s to %s (%d bars), instrument=%s, timeframe=%s",
        data_start, data_end, bar_count, base.instrument, base.timeframe,
    )

    # --- 2. Generate windows ---
    windows_spec = _generate_windows(config, data_start, data_end)

    if len(windows_spec) < 2:
        raise ValueError(
            f"Not enough data for WFA: only {len(windows_spec)} window(s) generated. "
            f"Need at least 2. Try reducing train_days ({config.train_days}) "
            f"or test_days ({config.test_days})."
        )

    n_windows = len(windows_spec)
    logger.info("WFA: %d windows generated (train=%dd, test=%dd, step=%dd)",
                n_windows, config.train_days, config.test_days, config.step_days)

    if config.step_days != config.test_days:
        logger.warning(
            "WFA: step_days (%d) != test_days (%d). Test windows may overlap or have gaps.",
            config.step_days, config.test_days,
        )

    # Minimize metrics need inverted efficiency
    minimize_metrics = {"max_drawdown", "max_drawdown_pct", "largest_loss"}
    is_minimize = config.opt_target_metric in minimize_metrics

    # --- 3. Per-window loop ---
    completed_windows: list[WalkForwardWindow] = []
    account_rules = TOPSTEP_ACCOUNTS.get(base.account_tier)
    initial_capital = float(account_rules.account_size) if account_rules else 50000.0

    for i, (train_start, train_end, test_start, test_end) in enumerate(windows_spec):
        logger.info("WFA window %d/%d: train=%s..%s, test=%s..%s",
                     i + 1, n_windows, train_start, train_end, test_start, test_end)

        if progress_callback:
            progress_callback(i, n_windows, f"Window {i+1}/{n_windows}: 優化中 Optimizing...")

        # --- 3a. Train: Optimize on train period ---
        train_config = BacktestConfig(
            strategy_name=base.strategy_name,
            strategy_params=base.strategy_params,
            instrument=base.instrument,
            timeframe=base.timeframe,
            start_date=train_start,
            end_date=train_end,
            account_tier=base.account_tier,
            quantity=base.quantity,
            slippage_ticks=base.slippage_ticks,
            commission=base.commission,
            circuit_breaker=base.circuit_breaker,
            best_day_limit=base.best_day_limit,
        )

        opt_t0 = time.perf_counter()
        seed = (config.opt_seed + i) if config.opt_seed is not None else None
        try:
            opt_result = run_optimization(
                base_config=train_config,
                n_iterations=config.opt_iterations,
                target_metric=config.opt_target_metric,
                seed=seed,
                max_workers=-1,
            )
        except Exception as e:
            logger.error("WFA window %d: optimization failed: %s", i + 1, e)
            continue

        opt_elapsed = time.perf_counter() - opt_t0

        # Find best result with enough trades
        best_result = None
        for r in opt_result.all_results:
            if r["metrics"].get("total_trades", 0) >= config.opt_min_trades:
                best_result = r
                break

        if best_result is None:
            # Fall back to best overall even if few trades
            best_result = {
                "params": opt_result.best_params,
                "metrics": opt_result.best_metrics,
            }
            logger.warning("WFA window %d: no result with >= %d trades, using best available.",
                           i + 1, config.opt_min_trades)

        best_params = best_result["params"]
        train_metrics = best_result["metrics"]
        train_target = train_metrics.get(config.opt_target_metric, 0) or 0

        train_n_valid = sum(
            1 for r in opt_result.all_results
            if r["metrics"].get("total_trades", 0) >= config.opt_min_trades
        )

        # --- 3b. Test: Run backtest on test period with warmup ---
        if progress_callback:
            progress_callback(i, n_windows, f"Window {i+1}/{n_windows}: 測試中 Testing...")

        warmup_start = _compute_warmup_start(
            test_start, config.warmup_bars, base.timeframe
        )

        test_config = BacktestConfig(
            strategy_name=base.strategy_name,
            strategy_params=best_params,
            instrument=base.instrument,
            timeframe=base.timeframe,
            start_date=warmup_start,
            end_date=test_end,
            account_tier=base.account_tier,
            quantity=base.quantity,
            slippage_ticks=base.slippage_ticks,
            commission=base.commission,
            circuit_breaker=base.circuit_breaker,
            best_day_limit=base.best_day_limit,
        )

        test_t0 = time.perf_counter()
        try:
            engine = BacktestEngine()
            test_result = engine.run(test_config)
        except Exception as e:
            logger.error("WFA window %d: test backtest failed: %s", i + 1, e)
            continue
        test_elapsed = time.perf_counter() - test_t0

        # Filter trades to only those exiting within the actual test window
        test_start_ts = pd.to_datetime(test_start)
        test_trades = [
            t for t in test_result.trades
            if pd.to_datetime(t["exit_time"]) >= test_start_ts
        ]

        # Filter equity curve to test window
        test_eq = [
            p for p in test_result.equity_curve
            if pd.to_datetime(p["timestamp"]) >= test_start_ts
        ]

        # Recompute metrics on filtered trades
        if test_trades:
            trade_objects = []
            for td in test_trades:
                try:
                    trade_objects.append(Trade(
                        entry_time=pd.to_datetime(td["entry_time"]).to_pydatetime(),
                        exit_time=pd.to_datetime(td["exit_time"]).to_pydatetime(),
                        instrument=td["instrument"],
                        direction=1 if td["direction"] == "LONG" else -1,
                        quantity=td["quantity"],
                        entry_price=td["entry_price"],
                        exit_price=td["exit_price"],
                        pnl=td["pnl"],
                        commission=td.get("commission", 0),
                        bars_held=td.get("bars_held", 0),
                    ))
                except Exception:
                    continue

            from data.models import EquityPoint
            eq_objects = []
            for ep in test_eq:
                try:
                    eq_objects.append(EquityPoint(
                        timestamp=pd.to_datetime(ep["timestamp"]).to_pydatetime(),
                        bar_index=ep.get("bar_index", 0),
                        equity=ep["equity"],
                        cash=ep.get("cash", ep["equity"]),
                        unrealized_pnl=ep.get("unrealized_pnl", 0),
                    ))
                except Exception:
                    continue

            test_metrics = compute_metrics(trade_objects, eq_objects, initial_capital)
        else:
            test_metrics = {}

        test_target = test_metrics.get(config.opt_target_metric, 0) or 0

        # Efficiency ratio
        if is_minimize:
            # Lower is better: efficiency = train / test (high if test is also low)
            efficiency = (train_target / test_target) if abs(test_target) > 1e-9 else 0.0
        else:
            # Higher is better: efficiency = test / train
            efficiency = (test_target / train_target) if abs(train_target) > 1e-9 else 0.0

        window = WalkForwardWindow(
            window_index=i,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            best_params=best_params,
            train_metrics=train_metrics,
            train_n_valid=train_n_valid,
            train_target_value=train_target,
            test_metrics=test_metrics,
            test_trades=test_trades,
            test_equity_curve=test_eq,
            efficiency_ratio=round(efficiency, 4),
            opt_elapsed_sec=round(opt_elapsed, 2),
            test_elapsed_sec=round(test_elapsed, 2),
        )
        completed_windows.append(window)
        logger.info(
            "WFA window %d: train_%s=%.4f, test_%s=%.4f, efficiency=%.4f, "
            "test_trades=%d, test_profit=$%.0f",
            i + 1, config.opt_target_metric, train_target,
            config.opt_target_metric, test_target,
            efficiency, len(test_trades),
            test_metrics.get("net_profit", 0),
        )

    if not completed_windows:
        raise ValueError("WFA: No windows completed successfully.")

    # --- 4. Stitch all test results ---
    if progress_callback:
        progress_callback(n_windows, n_windows, "拼接結果 Stitching results...")

    all_test_trades = []
    for w in completed_windows:
        all_test_trades.extend(w.test_trades)
    # Sort by exit time
    all_test_trades.sort(key=lambda t: t.get("exit_time", ""))

    stitched_eq = _stitch_equity_curves(completed_windows, initial_capital)

    # Recompute stitched metrics
    if all_test_trades:
        trade_objects = []
        for td in all_test_trades:
            try:
                trade_objects.append(Trade(
                    entry_time=pd.to_datetime(td["entry_time"]).to_pydatetime(),
                    exit_time=pd.to_datetime(td["exit_time"]).to_pydatetime(),
                    instrument=td["instrument"],
                    direction=1 if td["direction"] == "LONG" else -1,
                    quantity=td["quantity"],
                    entry_price=td["entry_price"],
                    exit_price=td["exit_price"],
                    pnl=td["pnl"],
                    commission=td.get("commission", 0),
                    bars_held=td.get("bars_held", 0),
                ))
            except Exception:
                continue

        from data.models import EquityPoint
        eq_objects = []
        for ep in stitched_eq:
            try:
                eq_objects.append(EquityPoint(
                    timestamp=pd.to_datetime(ep["timestamp"]).to_pydatetime(),
                    bar_index=ep.get("bar_index", 0),
                    equity=ep["equity"],
                    cash=ep.get("cash", ep["equity"]),
                    unrealized_pnl=ep.get("unrealized_pnl", 0),
                ))
            except Exception:
                continue

        stitched_metrics = compute_metrics(trade_objects, eq_objects, initial_capital)
    else:
        stitched_metrics = {}

    # --- 5. Aggregate WFA metrics ---
    train_targets = [w.train_target_value for w in completed_windows]
    test_targets = [w.test_metrics.get(config.opt_target_metric, 0) or 0
                    for w in completed_windows]

    avg_train = np.mean(train_targets) if train_targets else 0
    avg_test = np.mean(test_targets) if test_targets else 0

    if is_minimize:
        wf_eff = (avg_train / avg_test) if abs(avg_test) > 1e-9 else 0.0
    else:
        wf_eff = (avg_test / avg_train) if abs(avg_train) > 1e-9 else 0.0

    n_profitable = sum(
        1 for w in completed_windows
        if (w.test_metrics.get("net_profit", 0) or 0) > 0
    )

    param_stability = _compute_param_stability(completed_windows)

    total_elapsed = time.perf_counter() - t0

    result = WalkForwardResult(
        config=config,
        windows=completed_windows,
        stitched_metrics=stitched_metrics,
        stitched_trades=all_test_trades,
        stitched_equity_curve=stitched_eq,
        wf_efficiency=round(wf_eff, 4),
        param_stability=param_stability,
        n_windows=len(completed_windows),
        n_profitable_windows=n_profitable,
        window_consistency=round(n_profitable / len(completed_windows), 4) if completed_windows else 0.0,
        total_elapsed_sec=round(total_elapsed, 2),
    )

    logger.info(
        "WFA complete: %d windows, efficiency=%.4f, consistency=%.1f%%, "
        "stitched_profit=$%.0f, elapsed=%.1fs",
        result.n_windows, result.wf_efficiency,
        result.window_consistency * 100,
        stitched_metrics.get("net_profit", 0),
        total_elapsed,
    )

    return result
