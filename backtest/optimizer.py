"""
Random parameter optimization for backtesting strategies.

Samples random parameter combinations within defined ranges and runs
full backtests for each, collecting metrics to identify optimal settings.

Supports parallel execution via multiprocessing for faster sweeps.

Usage:
    from backtest.optimizer import run_optimization
    result = run_optimization(base_config, n_iterations=100, target_metric="sharpe_ratio")
    print(result.best_params)
"""

from __future__ import annotations

import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from backtest.engine import BacktestConfig, BacktestEngine, BacktestResult
from strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Results from a parameter optimization run.

    Attributes:
        n_iterations:    Total parameter combinations tested.
        target_metric:   The metric used to rank results.
        all_results:     All results sorted by target metric (best first).
        best_params:     Parameter dict of the best run.
        best_metrics:    Full metrics dict of the best run.
        param_names:     List of parameter names that were optimized.
        param_metric_df: DataFrame with columns [param1, param2, ..., metric].
        elapsed_sec:     Total wall-clock time.
    """

    n_iterations: int
    target_metric: str
    all_results: list[dict]
    best_params: dict
    best_metrics: dict
    param_names: list[str]
    param_metric_df: pd.DataFrame
    elapsed_sec: float = 0.0


def _generate_random_params(
    param_specs: dict[str, dict],
    n_samples: int,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Generate N random parameter combinations within spec ranges.

    Args:
        param_specs: Dict of param_name -> {type, default, min, max, step}.
        n_samples:   Number of random combinations to generate.
        seed:        Random seed for reproducibility.

    Returns:
        List of dicts, each mapping param_name -> sampled_value.
    """
    rng = np.random.default_rng(seed)
    samples = []

    for _ in range(n_samples):
        combo: dict[str, Any] = {}
        for name, spec in param_specs.items():
            ptype = spec["type"]
            pmin = spec["min"]
            pmax = spec["max"]
            step = spec.get("step", 1)

            if pmin is None or pmax is None:
                # No range defined, use default
                combo[name] = spec["default"]
                continue

            if ptype == "int":
                # Sample int in range, aligned to step
                n_steps = max(1, int((pmax - pmin) / step))
                step_idx = rng.integers(0, n_steps + 1)
                value = int(pmin + step_idx * step)
                value = min(value, int(pmax))
                combo[name] = value
            elif ptype == "float":
                # Uniform sample, rounded to step
                raw = rng.uniform(pmin, pmax)
                if step and step > 0:
                    value = round(raw / step) * step
                    value = max(pmin, min(pmax, value))
                else:
                    value = raw
                combo[name] = round(value, 6)
            elif ptype == "bool":
                combo[name] = bool(rng.integers(0, 2))
            else:
                combo[name] = spec["default"]

        samples.append(combo)

    return samples


def _run_single_backtest(config: BacktestConfig, preloaded_df=None) -> dict:
    """Run a single backtest and return params + metrics.

    This function is designed to be called from a process pool.
    """
    try:
        engine = BacktestEngine()
        result = engine.run(config, preloaded_df=preloaded_df)
        return {
            "params": config.strategy_params,
            "metrics": result.metrics,
            "trades": len(result.trades),
            "violations": len(result.violations),
            "error": None,
        }
    except Exception as e:
        return {
            "params": config.strategy_params,
            "metrics": {},
            "trades": 0,
            "violations": 0,
            "error": str(e),
        }


def run_optimization(
    base_config: BacktestConfig,
    n_iterations: int = 100,
    target_metric: str = "sharpe_ratio",
    param_ranges: dict[str, dict] | None = None,
    max_workers: int | None = -1,
    seed: int | None = None,
    progress_callback: Any = None,
    preloaded_df=None,
) -> OptimizationResult:
    """Run random parameter optimization.

    Samples random parameter combinations, runs backtests for each,
    and returns results sorted by the target metric.

    Args:
        base_config:       Base BacktestConfig (strategy_params will be overridden).
        n_iterations:      Number of random parameter combinations to test.
        target_metric:     Metric to optimize (e.g. "sharpe_ratio", "net_profit",
                          "profit_factor", "max_drawdown").
        param_ranges:      Optional override for parameter ranges.
                          Format: {param_name: {type, min, max, step, default}}.
                          If None, reads from strategy's StrategyParam definitions.
        max_workers:       Number of parallel workers. None = sequential.
        seed:              Random seed for reproducibility.
        progress_callback: Optional callable(current, total) for progress updates.

    Returns:
        OptimizationResult with all results sorted by target metric.
    """
    t0 = time.perf_counter()

    # --- 1. Get parameter specs ---
    if param_ranges is None:
        param_specs = _get_strategy_param_specs(base_config.strategy_name)
    else:
        param_specs = param_ranges

    if not param_specs:
        raise ValueError(
            f"No optimizable parameters found for strategy '{base_config.strategy_name}'"
        )

    param_names = list(param_specs.keys())
    logger.info(
        "Optimization: %d iterations, target=%s, params=%s",
        n_iterations, target_metric, param_names,
    )

    # --- 2. Generate random parameter combinations ---
    random_params = _generate_random_params(param_specs, n_iterations, seed=seed)

    # --- 3. Build configs ---
    configs = []
    for params in random_params:
        config = BacktestConfig(
            strategy_name=base_config.strategy_name,
            strategy_params=params,
            instrument=base_config.instrument,
            timeframe=base_config.timeframe,
            start_date=base_config.start_date,
            end_date=base_config.end_date,
            account_tier=base_config.account_tier,
            quantity=base_config.quantity,
            slippage_ticks=base_config.slippage_ticks,
            commission=base_config.commission,
            circuit_breaker=base_config.circuit_breaker,
            best_day_limit=base_config.best_day_limit,
        )
        configs.append(config)

    # --- 4. Run backtests ---
    all_results: list[dict] = []

    # Auto-detect workers: -1 means use all CPUs
    if max_workers == -1:
        max_workers = min(os.cpu_count() or 4, n_iterations, 8)

    if max_workers and max_workers > 1:
        # Parallel execution
        logger.info("Running %d backtests with %d workers...", n_iterations, max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(_run_single_backtest, cfg): i
                for i, cfg in enumerate(configs)
            }
            completed = 0
            for future in as_completed(future_to_idx):
                result = future.result()
                all_results.append(result)
                completed += 1
                if progress_callback:
                    progress_callback(completed, n_iterations)
                if completed % 10 == 0:
                    logger.info("Progress: %d/%d completed", completed, n_iterations)
    else:
        # Sequential execution
        for i, cfg in enumerate(configs):
            result = _run_single_backtest(cfg, preloaded_df=preloaded_df)
            all_results.append(result)
            if progress_callback:
                progress_callback(i + 1, n_iterations)
            if (i + 1) % 10 == 0:
                logger.info("Progress: %d/%d completed", i + 1, n_iterations)

    # --- 5. Sort by target metric ---
    # Handle metrics that should be minimized vs maximized
    minimize_metrics = {"max_drawdown", "max_drawdown_pct", "largest_loss"}
    reverse = target_metric not in minimize_metrics

    # Filter out errored runs and sort
    valid_results = [r for r in all_results if r["error"] is None and r["metrics"]]

    valid_results.sort(
        key=lambda r: r["metrics"].get(target_metric, float("-inf") if reverse else float("inf")),
        reverse=reverse,
    )

    # --- 6. Build DataFrame for analysis ---
    rows = []
    for r in valid_results:
        row = {}
        for pname in param_names:
            row[pname] = r["params"].get(pname, None)
        row[target_metric] = r["metrics"].get(target_metric, None)
        row["net_profit"] = r["metrics"].get("net_profit", None)
        row["win_rate"] = r["metrics"].get("win_rate", None)
        row["total_trades"] = r["metrics"].get("total_trades", None)
        row["max_drawdown"] = r["metrics"].get("max_drawdown", None)
        row["profit_factor"] = r["metrics"].get("profit_factor", None)
        rows.append(row)

    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    elapsed = time.perf_counter() - t0

    best = valid_results[0] if valid_results else {"params": {}, "metrics": {}}
    errored = sum(1 for r in all_results if r["error"] is not None)

    logger.info(
        "Optimization complete: %d valid / %d errored in %.1fs. "
        "Best %s = %s",
        len(valid_results), errored, elapsed,
        target_metric, best["metrics"].get(target_metric, "N/A"),
    )

    return OptimizationResult(
        n_iterations=n_iterations,
        target_metric=target_metric,
        all_results=valid_results,
        best_params=best["params"],
        best_metrics=best["metrics"],
        param_names=param_names,
        param_metric_df=df,
        elapsed_sec=round(elapsed, 2),
    )


def _get_strategy_param_specs(strategy_name: str) -> dict[str, dict]:
    """Extract parameter specs from a strategy's StrategyParam definitions.

    Returns a dict of param_name -> {type, default, min, max, step}.
    Only includes parameters with defined min/max ranges.
    """
    strat_cls = StrategyRegistry.get_strategy(strategy_name)
    specs: dict[str, dict] = {}

    for pname, param in strat_cls.params.items():
        if param.min_val is not None and param.max_val is not None:
            specs[pname] = {
                "type": param.param_type,
                "default": param.default,
                "min": param.min_val,
                "max": param.max_val,
                "step": param.step or 1,
            }

    return specs
