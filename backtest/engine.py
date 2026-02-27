"""
Backtest engine -- the main event loop that drives the simulation.

Orchestrates the flow:
    DataLoader -> MarketEvent -> Strategy -> SignalEvent
    -> Portfolio -> OrderEvent -> Broker -> FillEvent -> Portfolio

Also integrates the RiskManager for TopStep rule enforcement and
records the full equity curve and trade log for analysis.
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from backtest.broker import SimulatedBroker
from backtest.events import MarketEvent, SignalDirection
from backtest.metrics import compute_metrics
from backtest.portfolio import Portfolio
from backtest.risk import RiskManager
from backtest.topstep_rules import TOPSTEP_ACCOUNTS, TopStepAccountRules
from data.loader import DataLoader
from data.models import INSTRUMENT_SPECS
from strategies.base import BaseStrategy
from strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for a single backtest run.

    Attributes:
        strategy_name:  Registered strategy name (looked up via StrategyRegistry).
        strategy_params: Override parameters for the strategy.
        instrument:     Instrument symbol (e.g. "ES", "NQ").
        timeframe:      Bar timeframe (e.g. "1min", "5min").
        start_date:     Inclusive start date for the data window.
        end_date:       Inclusive end date for the data window.
        account_tier:   TopStep account tier key (e.g. "50K", "100K", "150K").
        quantity:       Number of contracts per trade.
        slippage_ticks: Ticks of adverse slippage per fill.
        commission:     Commission per contract per side.
    """

    strategy_name: str
    strategy_params: dict[str, Any] = field(default_factory=dict)
    instrument: str = "ES"
    timeframe: str = "5min"
    start_date: str | None = None
    end_date: str | None = None
    account_tier: str = "50K"
    quantity: int = 1
    slippage_ticks: int = 1
    commission: float = 2.50


@dataclass
class BacktestResult:
    """Results from a completed backtest run.

    Attributes:
        config:       The configuration that produced this result.
        metrics:      Dictionary of performance metrics.
        trades:       List of Trade records (as dicts for serialization).
        equity_curve: List of EquityPoint records (as dicts).
        elapsed_sec:  Wall-clock time of the backtest in seconds.
        total_bars:   Number of bars processed.
        violations:   List of risk-rule violation messages encountered.
    """

    config: BacktestConfig
    metrics: dict
    trades: list[dict]
    equity_curve: list[dict]
    elapsed_sec: float
    total_bars: int
    violations: list[str] = field(default_factory=list)


class BacktestEngine:
    """Event-driven backtesting engine.

    Usage::

        engine = BacktestEngine()
        result = engine.run(config)
        print(result.metrics)
    """

    def __init__(self, data_loader: DataLoader | None = None) -> None:
        self._loader = data_loader or DataLoader()

    def run(self, config: BacktestConfig) -> BacktestResult:
        """Execute a full backtest and return the results.

        Steps:
            1. Load bar data via DataLoader.
            2. Instantiate strategy, broker, portfolio, risk manager.
            3. Call strategy.on_init() for pre-computation.
            4. Iterate bars: emit MarketEvent -> strategy -> portfolio -> broker.
            5. After each bar: check risk rules, force-close if violated.
            6. At end: force-close any open position, compute metrics.

        Args:
            config: Backtest configuration.

        Returns:
            BacktestResult with metrics, trades, equity curve, etc.
        """
        t0 = _time.perf_counter()

        # ---- 1. Load data ----
        df = self._loader.get_bars(
            instrument=config.instrument,
            timeframe=config.timeframe,
            start_date=config.start_date,
            end_date=config.end_date,
        )
        if df.empty:
            raise ValueError(
                f"No data found for {config.instrument}/{config.timeframe}"
            )

        total_bars = len(df)
        logger.info(
            "Backtest: %s on %s/%s (%d bars)",
            config.strategy_name,
            config.instrument,
            config.timeframe,
            total_bars,
        )

        # ---- 2. Instantiate components ----
        strategy = self._create_strategy(config)
        inst_specs = INSTRUMENT_SPECS[config.instrument.upper()]

        rules = TOPSTEP_ACCOUNTS.get(config.account_tier)
        if rules is None:
            raise ValueError(
                f"Unknown account tier '{config.account_tier}'. "
                f"Available: {list(TOPSTEP_ACCOUNTS.keys())}"
            )

        initial_capital = float(rules.account_size)

        broker = SimulatedBroker(
            slippage_ticks=config.slippage_ticks,
            tick_size=inst_specs["tick_size"],
            commission=config.commission,
        )
        portfolio = Portfolio(
            initial_capital=initial_capital,
            commission_per_contract=config.commission,
            instrument_specs=inst_specs,
        )
        risk_mgr = RiskManager(rules=rules, initial_capital=initial_capital)

        # ---- 3. Strategy pre-computation ----
        strategy.on_init(df)

        # ---- 4. Main loop ----
        violations: list[str] = []
        account_blown = False

        for i in range(total_bars):
            row = df.iloc[i]
            bar_evt = self._make_bar_event(df, i)

            # -- Process pending orders from previous bar --
            # (always process, even after account blown, to fill force-close orders)
            fills = broker.process_bar(bar_evt)
            for fill in fills:
                portfolio.on_fill(fill, bar_index=i)

                # If this fill closes the position, clear SL/TP and notify strategy
                if portfolio.position == 0:
                    broker.clear_sl_tp()
                    strategy.on_position_closed()

            if account_blown:
                # Record equity for remaining bars (flat)
                portfolio.update_market(bar_evt)
                continue

            # -- Strategy signal --
            history = df.iloc[: i + 1]
            bar_dict = row.to_dict()
            signal = strategy.on_bar(bar_dict, history)

            if signal is not None:
                # -- Time window check: block new entries near must-flat-by --
                if signal.direction.value != 0 and risk_mgr.is_in_no_entry_window(
                    bar_evt.timestamp, buffer_minutes=20
                ):
                    logger.debug(
                        "Entry blocked: too close to must-flat-by (%s)",
                        bar_evt.timestamp.time(),
                    )
                    signal = None

            if signal is not None:
                # Try strategy's custom order builder first (for SL/TP, multi-TP)
                custom_order = strategy.build_order(
                    signal=signal,
                    bar=bar_dict,
                    quantity=config.quantity,
                    tick_size=inst_specs["tick_size"],
                )

                order_to_submit = custom_order
                if order_to_submit is None:
                    # Default: let portfolio convert signal to order
                    order_to_submit = portfolio.on_signal(
                        signal, quantity=config.quantity
                    )

                # -- Circuit breaker: skip if SL loss would blow account --
                if order_to_submit is not None and order_to_submit.stop_loss is not None:
                    equity = portfolio.get_equity()
                    point_value = inst_specs["tick_value"] / inst_specs["tick_size"]
                    entry_price = bar_dict["close"]
                    is_safe, reason = risk_mgr.check_order_safe_with_entry(
                        order=order_to_submit,
                        entry_price=entry_price,
                        equity=equity,
                        point_value=point_value,
                    )
                    if not is_safe:
                        violations.append(reason)
                        order_to_submit = None
                        strategy.on_position_closed()  # reset strategy state

                if order_to_submit is not None:
                    broker.submit_order(order_to_submit)

            # -- Update market (equity snapshot) --
            portfolio.update_market(bar_evt)

            # -- Risk check --
            equity = portfolio.get_equity()
            rule_violations = risk_mgr.check(
                equity=equity,
                position=portfolio.position,
                timestamp=bar_evt.timestamp,
            )

            if rule_violations:
                violations.extend(rule_violations)

                if risk_mgr.should_force_close(rule_violations):
                    close_order = portfolio.force_close_order(bar_evt.timestamp)
                    if close_order is not None:
                        broker.submit_order(close_order)
                        broker.clear_sl_tp()

                    # Check if trailing max loss was breached -> account blown
                    for v in rule_violations:
                        if "TRAILING MAX LOSS" in v:
                            account_blown = True
                            logger.warning("Account blown at bar %d", i)
                            break

        # ---- 5. End-of-backtest: force close open position ----
        if portfolio.position != 0 and not account_blown:
            last_bar = self._make_bar_event(df, total_bars - 1)
            close_order = portfolio.force_close_order(last_bar.timestamp)
            if close_order is not None:
                broker.submit_order(close_order)
                broker.clear_sl_tp()
                # Process one more time to fill the close order
                end_fills = broker.process_bar(last_bar)
                for fill in end_fills:
                    portfolio.on_fill(fill, bar_index=total_bars - 1)

        # ---- 6. Compute metrics ----
        metrics = compute_metrics(
            trades=portfolio.trades,
            equity_curve=portfolio.equity_curve,
            initial_capital=initial_capital,
        )

        elapsed = _time.perf_counter() - t0

        # Serialize trades and equity curve
        trades_dicts = [
            {
                "entry_time": str(t.entry_time),
                "exit_time": str(t.exit_time),
                "instrument": t.instrument,
                "direction": "LONG" if t.direction == 1 else "SHORT",
                "quantity": t.quantity,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "pnl": round(t.pnl, 2),
                "commission": round(t.commission, 2),
                "bars_held": t.bars_held,
            }
            for t in portfolio.trades
        ]

        equity_dicts = [
            {
                "timestamp": str(ep.timestamp),
                "bar_index": ep.bar_index,
                "equity": round(ep.equity, 2),
                "cash": round(ep.cash, 2),
                "unrealized_pnl": round(ep.unrealized_pnl, 2),
            }
            for ep in portfolio.equity_curve
        ]

        result = BacktestResult(
            config=config,
            metrics=metrics,
            trades=trades_dicts,
            equity_curve=equity_dicts,
            elapsed_sec=round(elapsed, 3),
            total_bars=total_bars,
            violations=violations,
        )

        logger.info(
            "Backtest complete: %d trades, net=%.2f, sharpe=%.2f (%.1fs)",
            metrics["total_trades"],
            metrics["net_profit"],
            metrics["sharpe_ratio"],
            elapsed,
        )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _create_strategy(config: BacktestConfig) -> BaseStrategy:
        """Instantiate a strategy by name from the registry."""
        strat_cls = StrategyRegistry.get_strategy(config.strategy_name)
        return strat_cls(**config.strategy_params)

    @staticmethod
    def _make_bar_event(df: pd.DataFrame, index: int) -> MarketEvent:
        """Create a MarketEvent from a DataFrame row."""
        row = df.iloc[index]
        return MarketEvent(
            timestamp=row["timestamp"].to_pydatetime(),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=int(row["volume"]),
            bar_index=index,
        )
