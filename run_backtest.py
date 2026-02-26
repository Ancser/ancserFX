#!/usr/bin/env python3
"""
CLI entry point for running backtests.

Usage
-----
    # Run SMA Crossover on ES 5min bars with default TopStep 50K rules
    python run_backtest.py --strategy "SMA Crossover" --instrument ES --timeframe 5min

    # Run RSI Mean Reversion with custom params
    python run_backtest.py --strategy "RSI Mean Reversion" --instrument NQ --timeframe 1min \
        --param rsi_period=10 --param oversold=25 --param overbought=75

    # Run on a date range with a specific account tier
    python run_backtest.py --strategy "Delta Momentum" --instrument ES --timeframe 5min \
        --start 2024-01-01 --end 2024-06-30 --account 100K

    # List available strategies
    python run_backtest.py --list

    # Show help
    python run_backtest.py --help
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.engine import BacktestConfig, BacktestEngine
from strategies.registry import StrategyRegistry


def _parse_param(param_str: str) -> tuple[str, str]:
    """Parse 'key=value' into (key, value)."""
    if "=" not in param_str:
        raise argparse.ArgumentTypeError(
            f"Parameter must be in key=value format, got: {param_str!r}"
        )
    key, value = param_str.split("=", 1)
    return key.strip(), value.strip()


def _format_metrics(metrics: dict) -> str:
    """Pretty-print metrics as a table."""
    lines = []
    lines.append("")
    lines.append("=" * 50)
    lines.append("  BACKTEST RESULTS")
    lines.append("=" * 50)

    sections = [
        ("TRADE SUMMARY", [
            ("Total Trades", "total_trades", "{:d}"),
            ("Winning Trades", "winning_trades", "{:d}"),
            ("Losing Trades", "losing_trades", "{:d}"),
            ("Win Rate", "win_rate", "{:.1f}%"),
        ]),
        ("PROFIT & LOSS", [
            ("Net Profit", "net_profit", "${:,.2f}"),
            ("Net Profit %", "net_profit_pct", "{:.2f}%"),
            ("Gross Profit", "gross_profit", "${:,.2f}"),
            ("Gross Loss", "gross_loss", "${:,.2f}"),
            ("Profit Factor", "profit_factor", "{:.2f}"),
        ]),
        ("TRADE ANALYSIS", [
            ("Average Trade", "avg_trade", "${:,.2f}"),
            ("Largest Win", "largest_win", "${:,.2f}"),
            ("Largest Loss", "largest_loss", "${:,.2f}"),
            ("Avg Winning Trade", "avg_winning_trade", "${:,.2f}"),
            ("Avg Losing Trade", "avg_losing_trade", "${:,.2f}"),
            ("Avg Bars in Trade", "avg_bars_in_trade", "{:.1f}"),
            ("Expectancy", "expectancy", "${:,.2f}"),
        ]),
        ("RISK METRICS", [
            ("Max Drawdown", "max_drawdown", "${:,.2f}"),
            ("Max Drawdown %", "max_drawdown_pct", "{:.2f}%"),
            ("Sharpe Ratio", "sharpe_ratio", "{:.3f}"),
            ("Max Consec. Wins", "max_consecutive_wins", "{:d}"),
            ("Max Consec. Losses", "max_consecutive_losses", "{:d}"),
        ]),
    ]

    for section_name, fields in sections:
        lines.append(f"\n  {section_name}")
        lines.append("  " + "-" * 46)
        for label, key, fmt in fields:
            value = metrics.get(key, 0)
            formatted = fmt.format(value)
            lines.append(f"  {label:<25} {formatted:>20}")

    lines.append("")
    lines.append("=" * 50)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ancserFX Backtest Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available strategies and exit.",
    )
    parser.add_argument(
        "--strategy", type=str, default=None,
        help="Strategy name (e.g. 'SMA Crossover').",
    )
    parser.add_argument(
        "--instrument", type=str, default="ES",
        help="Instrument symbol (default: ES).",
    )
    parser.add_argument(
        "--timeframe", type=str, default="5min",
        help="Bar timeframe (default: 5min).",
    )
    parser.add_argument(
        "--start", type=str, default=None,
        help="Start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end", type=str, default=None,
        help="End date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--account", type=str, default="50K",
        choices=["50K", "100K", "150K"],
        help="TopStep account tier (default: 50K).",
    )
    parser.add_argument(
        "--quantity", type=int, default=1,
        help="Contracts per trade (default: 1).",
    )
    parser.add_argument(
        "--slippage", type=int, default=1,
        help="Slippage in ticks (default: 1).",
    )
    parser.add_argument(
        "--commission", type=float, default=2.50,
        help="Commission per contract per side (default: 2.50).",
    )
    parser.add_argument(
        "--param", action="append", default=[],
        help="Strategy parameter override in key=value format. Can be repeated.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Path to save full results as JSON.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # List strategies
    if args.list:
        strategies = StrategyRegistry.list_strategies()
        if not strategies:
            print("No strategies registered.")
            return

        print("\nAvailable Strategies:")
        print("-" * 60)
        for s in strategies:
            print(f"  [{s['category']}] {s['name']}")
            if s['description']:
                print(f"         {s['description']}")
            if s['parameters']:
                for pname, pinfo in s['parameters'].items():
                    print(f"           --param {pname}={pinfo['default']}  ({pinfo['type']})")
            print()
        return

    # Validate required args
    if args.strategy is None:
        parser.error("--strategy is required (use --list to see available strategies)")

    # Parse strategy params
    strategy_params: dict = {}
    for p in args.param:
        key, value = _parse_param(p)
        # Try to auto-convert numeric values
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        strategy_params[key] = value

    # Build config
    config = BacktestConfig(
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

    # Run backtest
    print(f"\nRunning: {config.strategy_name} on {config.instrument}/{config.timeframe}")
    print(f"Account: TopStep {config.account_tier} | Qty: {config.quantity}")
    if config.start_date or config.end_date:
        print(f"Period: {config.start_date or 'start'} -> {config.end_date or 'end'}")
    print()

    engine = BacktestEngine()
    result = engine.run(config)

    # Display results
    print(_format_metrics(result.metrics))
    print(f"  Elapsed: {result.elapsed_sec:.1f}s | Bars: {result.total_bars:,}")

    if result.violations:
        print(f"\n  Risk Violations ({len(result.violations)}):")
        for v in result.violations[:10]:
            print(f"    - {v}")
        if len(result.violations) > 10:
            print(f"    ... and {len(result.violations) - 10} more")

    # Save JSON output
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "config": {
                "strategy_name": config.strategy_name,
                "strategy_params": config.strategy_params,
                "instrument": config.instrument,
                "timeframe": config.timeframe,
                "start_date": config.start_date,
                "end_date": config.end_date,
                "account_tier": config.account_tier,
                "quantity": config.quantity,
                "slippage_ticks": config.slippage_ticks,
                "commission": config.commission,
            },
            "metrics": result.metrics,
            "trades": result.trades,
            "equity_curve": result.equity_curve,
            "total_bars": result.total_bars,
            "elapsed_sec": result.elapsed_sec,
            "violations": result.violations,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\n  Results saved to: {output_path}")

    print()


if __name__ == "__main__":
    main()
