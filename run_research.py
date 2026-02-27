#!/usr/bin/env python3
"""
Research CLI for Monte Carlo simulation and parameter optimization.

Usage
-----
    # Monte Carlo: shuffle trades 1000 times, assess strategy robustness
    python run_research.py monte-carlo \\
        --strategy "KDJ RSI Bot" --instrument MES --account 50K --quantity 3 \\
        --param sl_points=8 --param tp1_ticks=60 \\
        --start 2024-01-01 --end 2024-06-30 \\
        --simulations 1000 --chart mc_results.html

    # Optimize: random search over parameter space
    python run_research.py optimize \\
        --strategy "KDJ RSI Bot" --instrument MES --account 50K --quantity 3 \\
        --start 2024-01-01 --end 2024-06-30 \\
        --iterations 50 --target sharpe_ratio \\
        --chart opt_results.html

    # Full: optimize -> best params backtest -> Monte Carlo
    python run_research.py full \\
        --strategy "KDJ RSI Bot" --instrument MES --account 50K --quantity 3 \\
        --start 2024-01-01 --end 2024-06-30 \\
        --iterations 50 --simulations 1000 \\
        --chart full_report.html
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.engine import BacktestConfig, BacktestEngine
from backtest.monte_carlo import run_monte_carlo
from backtest.optimizer import run_optimization
from backtest.topstep_rules import TOPSTEP_ACCOUNTS
from backtest.visualize import plot_monte_carlo, plot_optimization, plot_full_report


def _build_base_config(args) -> BacktestConfig:
    """Build a BacktestConfig from parsed CLI arguments."""
    strategy_params = {}
    for p in (args.param or []):
        if "=" not in p:
            raise ValueError(f"Parameter must be key=value, got: {p!r}")
        key, val = p.split("=", 1)
        try:
            val = int(val)
        except ValueError:
            try:
                val = float(val)
            except ValueError:
                pass
        strategy_params[key.strip()] = val

    return BacktestConfig(
        strategy_name=args.strategy,
        strategy_params=strategy_params,
        instrument=args.instrument,
        timeframe=args.timeframe,
        start_date=args.start,
        end_date=args.end,
        account_tier=args.account,
        quantity=args.quantity,
        slippage_ticks=args.slippage,
        commission=args.commission,
    )


def _run_backtest_and_extract(config: BacktestConfig):
    """Run a single backtest and return (result, trades_pnl, result_dict)."""
    engine = BacktestEngine()
    result = engine.run(config)

    trades_pnl = [t["pnl"] for t in result.trades]

    result_dict = {
        "config": {
            "strategy_name": config.strategy_name,
            "strategy_params": config.strategy_params,
            "instrument": config.instrument,
            "timeframe": config.timeframe,
            "start_date": config.start_date,
            "end_date": config.end_date,
            "account_tier": config.account_tier,
            "quantity": config.quantity,
        },
        "metrics": result.metrics,
        "trades": result.trades,
        "equity_curve": result.equity_curve,
        "total_bars": result.total_bars,
        "violations": result.violations,
    }

    return result, trades_pnl, result_dict


def _print_metrics_summary(metrics: dict, label: str = "") -> None:
    """Print a compact metrics summary."""
    if label:
        print(f"\n  {label}")
        print("  " + "-" * 50)
    print(f"  Net Profit:    ${metrics.get('net_profit', 0):>10,.2f}")
    print(f"  Win Rate:      {metrics.get('win_rate', 0):>10.1f}%")
    print(f"  Trades:        {metrics.get('total_trades', 0):>10d}")
    print(f"  Profit Factor: {metrics.get('profit_factor', 0):>10.2f}")
    print(f"  Sharpe Ratio:  {metrics.get('sharpe_ratio', 0):>10.3f}")
    print(f"  Max Drawdown:  ${metrics.get('max_drawdown', 0):>10,.2f}")


def cmd_monte_carlo(args) -> None:
    """Run Monte Carlo simulation."""
    config = _build_base_config(args)
    rules = TOPSTEP_ACCOUNTS.get(config.account_tier)
    initial_capital = float(rules.account_size)
    loss_limit = float(rules.max_loss_limit)

    print(f"\n{'='*60}")
    print(f"  MONTE CARLO SIMULATION")
    print(f"{'='*60}")
    print(f"  Strategy:    {config.strategy_name}")
    print(f"  Instrument:  {config.instrument} | Account: {config.account_tier}")
    print(f"  Simulations: {args.simulations:,}")
    print(f"{'='*60}")

    # Step 1: Run base backtest
    print("\n  [1/2] Running base backtest...")
    t0 = time.perf_counter()
    result, trades_pnl, result_dict = _run_backtest_and_extract(config)
    bt_time = time.perf_counter() - t0

    _print_metrics_summary(result.metrics, "Base Backtest Results")
    print(f"  Elapsed: {bt_time:.1f}s")

    if not trades_pnl:
        print("\n  No trades produced. Cannot run Monte Carlo.")
        return

    # Step 2: Monte Carlo
    print(f"\n  [2/2] Running {args.simulations:,} Monte Carlo simulations...")
    t0 = time.perf_counter()
    mc_result = run_monte_carlo(
        trades_pnl=trades_pnl,
        initial_capital=initial_capital,
        loss_limit=loss_limit,
        n_simulations=args.simulations,
        seed=args.seed,
    )
    mc_time = time.perf_counter() - t0

    # Print MC results
    fe = mc_result.final_equity_stats
    dd = mc_result.max_drawdown_stats
    cl = mc_result.max_consec_loss_stats

    print(f"\n  Monte Carlo Results ({args.simulations:,} simulations)")
    print("  " + "-" * 50)
    print(f"  Ruin Probability:    {mc_result.ruin_probability:>10.1%}")
    print(f"  Final Equity (5th):  ${fe['p5']:>10,.0f}")
    print(f"  Final Equity (50th): ${fe['p50']:>10,.0f}")
    print(f"  Final Equity (95th): ${fe['p95']:>10,.0f}")
    print(f"  Max DD Mean:         ${dd['mean']:>10,.0f}")
    print(f"  Max DD (95th pct):   ${dd['p95']:>10,.0f}")
    print(f"  Max Consec Loss Avg: {cl['mean']:>10.1f}")
    print(f"  Max Consec Loss 95%: {cl['p95']:>10.0f}")
    print(f"  Elapsed: {mc_time:.1f}s")

    # Chart
    if args.chart:
        chart_path = None if args.chart == "auto" else args.chart
        plot_monte_carlo(mc_result, output_path=chart_path)

    print()


def cmd_optimize(args) -> None:
    """Run parameter optimization."""
    config = _build_base_config(args)

    print(f"\n{'='*60}")
    print(f"  PARAMETER OPTIMIZATION")
    print(f"{'='*60}")
    print(f"  Strategy:    {config.strategy_name}")
    print(f"  Instrument:  {config.instrument} | Account: {config.account_tier}")
    print(f"  Iterations:  {args.iterations}")
    print(f"  Target:      {args.target}")
    print(f"  Workers:     {args.workers or 'sequential'}")
    print(f"{'='*60}")

    print(f"\n  Running {args.iterations} backtests...")
    opt_result = run_optimization(
        base_config=config,
        n_iterations=args.iterations,
        target_metric=args.target,
        max_workers=args.workers,
        seed=args.seed,
    )

    # Print top results
    print(f"\n  Optimization Complete ({opt_result.elapsed_sec:.1f}s)")
    print("  " + "=" * 55)

    top_n = min(10, len(opt_result.all_results))
    print(f"\n  Top {top_n} Parameter Combinations ({args.target}):")
    print("  " + "-" * 55)

    for i, r in enumerate(opt_result.all_results[:top_n]):
        target_val = r["metrics"].get(args.target, 0)
        net = r["metrics"].get("net_profit", 0)
        wr = r["metrics"].get("win_rate", 0)
        trades_n = r["metrics"].get("total_trades", 0)
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"  #{i+1:2d} | {args.target}={target_val:>8.4f} | "
              f"net=${net:>8,.0f} | wr={wr:>5.1f}% | trades={trades_n:>3d}")
        print(f"       {params_str}")

    print(f"\n  Best Parameters:")
    for k, v in opt_result.best_params.items():
        print(f"    {k} = {v}")

    _print_metrics_summary(opt_result.best_metrics, "Best Run Metrics")

    # Chart
    if args.chart:
        chart_path = None if args.chart == "auto" else args.chart
        plot_optimization(opt_result, output_path=chart_path)

    print()


def cmd_full(args) -> None:
    """Run full research: optimize -> backtest with best -> Monte Carlo."""
    config = _build_base_config(args)
    rules = TOPSTEP_ACCOUNTS.get(config.account_tier)
    initial_capital = float(rules.account_size)
    loss_limit = float(rules.max_loss_limit)

    print(f"\n{'='*60}")
    print(f"  FULL RESEARCH PIPELINE")
    print(f"{'='*60}")
    print(f"  Strategy:    {config.strategy_name}")
    print(f"  Instrument:  {config.instrument} | Account: {config.account_tier}")
    print(f"  Opt Iters:   {args.iterations}")
    print(f"  MC Sims:     {args.simulations}")
    print(f"  Target:      {args.target}")
    print(f"{'='*60}")

    # Step 1: Optimize
    print(f"\n  [1/3] Parameter Optimization ({args.iterations} iterations)...")
    opt_result = run_optimization(
        base_config=config,
        n_iterations=args.iterations,
        target_metric=args.target,
        max_workers=args.workers,
        seed=args.seed,
    )

    print(f"        Best {args.target} = {opt_result.best_metrics.get(args.target, 'N/A')}")
    print(f"        Best params: {opt_result.best_params}")

    # Step 2: Backtest with best params
    print(f"\n  [2/3] Backtest with best parameters...")
    best_config = BacktestConfig(
        strategy_name=config.strategy_name,
        strategy_params=opt_result.best_params,
        instrument=config.instrument,
        timeframe=config.timeframe,
        start_date=config.start_date,
        end_date=config.end_date,
        account_tier=config.account_tier,
        quantity=config.quantity,
        slippage_ticks=config.slippage_ticks,
        commission=config.commission,
    )
    _, trades_pnl, result_dict = _run_backtest_and_extract(best_config)
    _print_metrics_summary(result_dict["metrics"], "Best Params Backtest")

    # Step 3: Monte Carlo
    mc_result = None
    if trades_pnl:
        print(f"\n  [3/3] Monte Carlo ({args.simulations:,} simulations)...")
        mc_result = run_monte_carlo(
            trades_pnl=trades_pnl,
            initial_capital=initial_capital,
            loss_limit=loss_limit,
            n_simulations=args.simulations,
            seed=args.seed,
        )

        fe = mc_result.final_equity_stats
        print(f"        Ruin Probability: {mc_result.ruin_probability:.1%}")
        print(f"        Median Final Equity: ${fe['p50']:,.0f}")
    else:
        print("\n  [3/3] Skipped (no trades from best params)")

    # Charts
    if args.chart:
        chart_path = args.chart if args.chart != "auto" else "research_report"
        plot_full_report(
            backtest_result=result_dict,
            mc_result=mc_result,
            opt_result=opt_result,
            output_path=chart_path,
        )

    # Print top-5 param combos for reference
    print(f"\n  Top 5 Parameter Sets:")
    print("  " + "-" * 55)
    for i, r in enumerate(opt_result.all_results[:5]):
        tv = r["metrics"].get(args.target, 0)
        net = r["metrics"].get("net_profit", 0)
        params_str = ", ".join(f"{k}={v}" for k, v in r["params"].items())
        print(f"  #{i+1} | {args.target}={tv:.4f} | net=${net:,.0f}")
        print(f"      {params_str}")

    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ancserFX Research Tools: Monte Carlo & Parameter Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Research command")

    # --- Shared arguments ---
    def add_common_args(p):
        p.add_argument("--strategy", type=str, required=True)
        p.add_argument("--instrument", type=str, default="MES")
        p.add_argument("--timeframe", type=str, default="5min")
        p.add_argument("--start", type=str, default=None)
        p.add_argument("--end", type=str, default=None)
        p.add_argument("--account", type=str, default="50K", choices=["50K", "100K", "150K"])
        p.add_argument("--quantity", type=int, default=3)
        p.add_argument("--slippage", type=int, default=1)
        p.add_argument("--commission", type=float, default=2.50)
        p.add_argument("--param", action="append", default=[])
        p.add_argument("--seed", type=int, default=None, help="Random seed")
        p.add_argument("--chart", type=str, nargs="?", const="auto", help="Generate chart")
        p.add_argument("--verbose", "-v", action="store_true")

    # --- monte-carlo ---
    mc_parser = subparsers.add_parser("monte-carlo", help="Monte Carlo trade shuffle simulation")
    add_common_args(mc_parser)
    mc_parser.add_argument("--simulations", type=int, default=1000, help="Number of simulations (default: 1000)")

    # --- optimize ---
    opt_parser = subparsers.add_parser("optimize", help="Random parameter optimization")
    add_common_args(opt_parser)
    opt_parser.add_argument("--iterations", type=int, default=100, help="Number of random param combos (default: 100)")
    opt_parser.add_argument("--target", type=str, default="sharpe_ratio",
                           help="Metric to optimize (default: sharpe_ratio)")
    opt_parser.add_argument("--workers", type=int, default=None,
                           help="Parallel workers (default: sequential)")

    # --- full ---
    full_parser = subparsers.add_parser("full", help="Full research: optimize + MC")
    add_common_args(full_parser)
    full_parser.add_argument("--iterations", type=int, default=100)
    full_parser.add_argument("--simulations", type=int, default=1000)
    full_parser.add_argument("--target", type=str, default="sharpe_ratio")
    full_parser.add_argument("--workers", type=int, default=None)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.command == "monte-carlo":
        cmd_monte_carlo(args)
    elif args.command == "optimize":
        cmd_optimize(args)
    elif args.command == "full":
        cmd_full(args)


if __name__ == "__main__":
    main()
