#!/usr/bin/env python3
"""
Streamlit interactive dashboard for ancserFX backtesting.

Usage:
    streamlit run dashboard.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.engine import BacktestConfig, BacktestEngine
from backtest.monte_carlo import run_monte_carlo
from backtest.optimizer import run_optimization
from backtest.topstep_rules import TOPSTEP_ACCOUNTS
from data.models import INSTRUMENT_SPECS
from strategies.registry import StrategyRegistry

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ancserFX Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark theme styling
st.markdown("""
<style>
    .stMetric { background-color: #1e1e1e; padding: 10px; border-radius: 8px; }
    .metric-positive { color: #00c853 !important; }
    .metric-negative { color: #ff1744 !important; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "bt_result" not in st.session_state:
    st.session_state.bt_result = None
if "bt_result_dict" not in st.session_state:
    st.session_state.bt_result_dict = None
if "mc_result" not in st.session_state:
    st.session_state.mc_result = None
if "opt_result" not in st.session_state:
    st.session_state.opt_result = None


# ---------------------------------------------------------------------------
# Sidebar — Strategy & Parameters
# ---------------------------------------------------------------------------
st.sidebar.title("ancserFX")

# Strategy selector
all_strategies = StrategyRegistry.list_strategies()
strat_names = [s["name"] for s in all_strategies]
strat_idx = st.sidebar.selectbox(
    "Strategy",
    range(len(strat_names)),
    format_func=lambda i: strat_names[i],
)
selected_strat = all_strategies[strat_idx]

# Instrument
instruments = list(INSTRUMENT_SPECS.keys())
instrument = st.sidebar.selectbox("Instrument", instruments, index=instruments.index("MES") if "MES" in instruments else 0)

# Account tier
account_tier = st.sidebar.selectbox("Account Tier", ["50K", "100K", "150K"], index=0)
rules = TOPSTEP_ACCOUNTS.get(account_tier)
st.sidebar.caption(
    f"Capital: ${rules.account_size:,.0f} | "
    f"Loss Limit: ${rules.max_loss_limit:,.0f} | "
    f"Max Contracts: {rules.max_contracts}"
)

# Quantity
quantity = st.sidebar.number_input("Quantity (contracts)", min_value=1, max_value=rules.max_contracts, value=min(3, rules.max_contracts))

# Timeframe
timeframe = st.sidebar.selectbox("Timeframe", ["1min", "5min", "15min", "1h", "daily"], index=1)

# Date range
col_d1, col_d2 = st.sidebar.columns(2)
start_date = col_d1.text_input("Start Date", value="2024-01-01", help="YYYY-MM-DD")
end_date = col_d2.text_input("End Date", value="2024-06-30", help="YYYY-MM-DD")

# Slippage & Commission
col_s1, col_s2 = st.sidebar.columns(2)
slippage = col_s1.number_input("Slippage (ticks)", min_value=0, max_value=10, value=1)
commission = col_s2.number_input("Commission", min_value=0.0, max_value=20.0, value=2.50, step=0.50)

# Dynamic parameter sliders
st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Parameters")

strategy_params: dict = {}
if selected_strat.get("parameters"):
    for pname, pinfo in selected_strat["parameters"].items():
        ptype = pinfo.get("type", "int")
        pmin = pinfo.get("min")
        pmax = pinfo.get("max")
        default = pinfo.get("default")
        step = pinfo.get("step", 1)
        desc = pinfo.get("description", "")

        if pmin is not None and pmax is not None:
            if ptype == "int":
                val = st.sidebar.slider(
                    pname,
                    min_value=int(pmin),
                    max_value=int(pmax),
                    value=int(default),
                    step=int(step) if step else 1,
                    help=desc,
                )
                strategy_params[pname] = val
            elif ptype == "float":
                val = st.sidebar.slider(
                    pname,
                    min_value=float(pmin),
                    max_value=float(pmax),
                    value=float(default),
                    step=float(step) if step else 0.1,
                    help=desc,
                )
                strategy_params[pname] = val
            elif ptype == "bool":
                val = st.sidebar.checkbox(pname, value=bool(default), help=desc)
                strategy_params[pname] = val
        else:
            # No range — show as text input
            val = st.sidebar.text_input(pname, value=str(default), help=desc)
            strategy_params[pname] = val


# ---------------------------------------------------------------------------
# Build config helper
# ---------------------------------------------------------------------------
def _build_config(params_override: dict | None = None) -> BacktestConfig:
    params = params_override or strategy_params
    return BacktestConfig(
        strategy_name=selected_strat["name"],
        strategy_params=params,
        instrument=instrument,
        timeframe=timeframe,
        start_date=start_date or None,
        end_date=end_date or None,
        account_tier=account_tier,
        quantity=quantity,
        slippage_ticks=slippage,
        commission=commission,
    )


# ---------------------------------------------------------------------------
# Action buttons
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")

col_btn1, col_btn2 = st.sidebar.columns(2)
run_bt = col_btn1.button("Run Backtest", type="primary", use_container_width=True)
run_mc = col_btn2.button("Monte Carlo", use_container_width=True)

col_btn3, col_btn4 = st.sidebar.columns(2)
run_opt = col_btn3.button("Optimize", use_container_width=True)
run_full = col_btn4.button("Full Pipeline", use_container_width=True)

# MC / Opt settings (expandable)
with st.sidebar.expander("MC / Optimization Settings"):
    mc_simulations = st.number_input("MC Simulations", min_value=100, max_value=10000, value=1000, step=100)
    opt_iterations = st.number_input("Opt Iterations", min_value=10, max_value=500, value=50, step=10)
    opt_target = st.selectbox("Opt Target Metric", ["sharpe_ratio", "net_profit", "profit_factor", "win_rate", "max_drawdown"])
    random_seed = st.number_input("Random Seed", min_value=0, max_value=99999, value=42)


# ---------------------------------------------------------------------------
# Execute actions
# ---------------------------------------------------------------------------
def _run_backtest():
    config = _build_config()
    engine = BacktestEngine()
    with st.spinner(f"Running backtest: {config.strategy_name} on {config.instrument}..."):
        t0 = time.perf_counter()
        result = engine.run(config)
        elapsed = time.perf_counter() - t0

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
        "elapsed_sec": elapsed,
        "violations": result.violations,
    }
    st.session_state.bt_result = result
    st.session_state.bt_result_dict = result_dict
    st.session_state.trades_pnl = trades_pnl
    return result, result_dict, trades_pnl


def _run_monte_carlo(trades_pnl):
    rules_mc = TOPSTEP_ACCOUNTS.get(account_tier)
    with st.spinner(f"Running {mc_simulations:,} Monte Carlo simulations..."):
        mc = run_monte_carlo(
            trades_pnl=trades_pnl,
            initial_capital=float(rules_mc.account_size),
            loss_limit=float(rules_mc.max_loss_limit),
            n_simulations=mc_simulations,
            seed=random_seed,
        )
    st.session_state.mc_result = mc
    return mc


if run_bt:
    _run_backtest()

if run_mc:
    if st.session_state.bt_result is None:
        result, result_dict, trades_pnl = _run_backtest()
    else:
        trades_pnl = st.session_state.get("trades_pnl", [])
    if trades_pnl:
        _run_monte_carlo(trades_pnl)
    else:
        st.warning("No trades produced — cannot run Monte Carlo.")

if run_opt:
    config = _build_config()
    with st.spinner(f"Optimizing {opt_iterations} iterations (target: {opt_target})..."):
        opt = run_optimization(
            base_config=config,
            n_iterations=opt_iterations,
            target_metric=opt_target,
            seed=random_seed,
        )
    st.session_state.opt_result = opt

if run_full:
    # Step 1: Optimize
    config = _build_config()
    progress = st.progress(0, text="Step 1/3: Optimizing parameters...")
    opt = run_optimization(
        base_config=config,
        n_iterations=opt_iterations,
        target_metric=opt_target,
        seed=random_seed,
    )
    st.session_state.opt_result = opt
    progress.progress(33, text="Step 2/3: Backtesting with best params...")

    # Step 2: Backtest with best
    best_config = _build_config(params_override=opt.best_params)
    engine = BacktestEngine()
    result = engine.run(best_config)
    trades_pnl = [t["pnl"] for t in result.trades]
    result_dict = {
        "config": {
            "strategy_name": best_config.strategy_name,
            "strategy_params": best_config.strategy_params,
            "instrument": best_config.instrument,
            "timeframe": best_config.timeframe,
            "start_date": best_config.start_date,
            "end_date": best_config.end_date,
            "account_tier": best_config.account_tier,
            "quantity": best_config.quantity,
        },
        "metrics": result.metrics,
        "trades": result.trades,
        "equity_curve": result.equity_curve,
        "total_bars": result.total_bars,
        "violations": result.violations,
    }
    st.session_state.bt_result = result
    st.session_state.bt_result_dict = result_dict
    st.session_state.trades_pnl = trades_pnl
    progress.progress(66, text="Step 3/3: Monte Carlo simulation...")

    # Step 3: MC
    if trades_pnl:
        _run_monte_carlo(trades_pnl)
    progress.progress(100, text="Complete!")


# ---------------------------------------------------------------------------
# Main display area
# ---------------------------------------------------------------------------
st.title("ancserFX Dashboard")

# --- Tab layout ---
tabs = st.tabs(["Backtest", "Monte Carlo", "Optimization", "Trade Log"])

# ============================================================
# TAB 1: BACKTEST
# ============================================================
with tabs[0]:
    rd = st.session_state.bt_result_dict
    if rd is None:
        st.info("Click **Run Backtest** to see results.")
    else:
        m = rd["metrics"]

        # Metrics row
        cols = st.columns(6)
        cols[0].metric("Net Profit", f"${m.get('net_profit', 0):,.2f}",
                       delta=f"{m.get('net_profit_pct', 0):.1f}%")
        cols[1].metric("Win Rate", f"{m.get('win_rate', 0):.1f}%")
        cols[2].metric("Trades", f"{m.get('total_trades', 0)}")
        cols[3].metric("Profit Factor", f"{m.get('profit_factor', 0):.2f}")
        cols[4].metric("Sharpe Ratio", f"{m.get('sharpe_ratio', 0):.3f}")
        cols[5].metric("Max Drawdown", f"${m.get('max_drawdown', 0):,.2f}")

        # Second metrics row
        cols2 = st.columns(6)
        cols2[0].metric("Largest Win", f"${m.get('largest_win', 0):,.2f}")
        cols2[1].metric("Largest Loss", f"${m.get('largest_loss', 0):,.2f}")
        cols2[2].metric("Avg Trade", f"${m.get('avg_trade', 0):,.2f}")
        cols2[3].metric("Expectancy", f"${m.get('expectancy', 0):,.2f}")
        cols2[4].metric("Max Consec Wins", f"{m.get('max_consecutive_wins', 0)}")
        cols2[5].metric("Max Consec Losses", f"{m.get('max_consecutive_losses', 0)}")

        # Equity curve chart
        eq = rd.get("equity_curve", [])
        if eq:
            eq_df = pd.DataFrame(eq)
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.7, 0.3],
                vertical_spacing=0.05,
                subplot_titles=["Equity Curve", "Drawdown"],
            )

            fig.add_trace(
                go.Scatter(
                    x=eq_df.get("timestamp", eq_df.index),
                    y=eq_df["equity"],
                    mode="lines",
                    name="Equity",
                    line=dict(color="#2196F3", width=1.5),
                ),
                row=1, col=1,
            )

            # Compute drawdown
            equity_arr = eq_df["equity"].values
            peak = np.maximum.accumulate(equity_arr)
            dd = peak - equity_arr

            fig.add_trace(
                go.Scatter(
                    x=eq_df.get("timestamp", eq_df.index),
                    y=-dd,
                    mode="lines",
                    fill="tozeroy",
                    name="Drawdown",
                    line=dict(color="#ff1744", width=1),
                    fillcolor="rgba(255,23,68,0.3)",
                ),
                row=2, col=1,
            )

            fig.update_layout(
                template="plotly_dark",
                height=500,
                showlegend=False,
                margin=dict(l=50, r=20, t=40, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Trade P&L bar chart
        trades = rd.get("trades", [])
        if trades:
            pnl_vals = [t["pnl"] for t in trades]
            colors = ["#00c853" if p >= 0 else "#ff1744" for p in pnl_vals]
            fig_pnl = go.Figure(
                go.Bar(
                    x=list(range(1, len(pnl_vals) + 1)),
                    y=pnl_vals,
                    marker_color=colors,
                    name="Trade P&L",
                )
            )
            fig_pnl.update_layout(
                template="plotly_dark",
                height=250,
                title="Trade P&L",
                xaxis_title="Trade #",
                yaxis_title="P&L ($)",
                margin=dict(l=50, r=20, t=40, b=30),
            )
            st.plotly_chart(fig_pnl, use_container_width=True)

        # Violations
        if rd.get("violations"):
            with st.expander(f"Risk Violations ({len(rd['violations'])})"):
                for v in rd["violations"][:50]:
                    st.text(v)

        # Config info
        cfg = rd["config"]
        st.caption(
            f"Strategy: {cfg['strategy_name']} | Instrument: {cfg['instrument']} | "
            f"Timeframe: {cfg.get('timeframe', '5min')} | "
            f"Bars: {rd.get('total_bars', 'N/A'):,} | "
            f"Elapsed: {rd.get('elapsed_sec', 0):.1f}s"
        )


# ============================================================
# TAB 2: MONTE CARLO
# ============================================================
with tabs[1]:
    mc = st.session_state.mc_result
    if mc is None:
        st.info("Click **Monte Carlo** to run simulation.")
    else:
        fe = mc.final_equity_stats
        dd = mc.max_drawdown_stats
        cl = mc.max_consec_loss_stats

        # Key metrics
        cols = st.columns(4)
        ruin_color = "normal" if mc.ruin_probability < 0.05 else "inverse"
        cols[0].metric("Ruin Probability", f"{mc.ruin_probability:.1%}", delta_color=ruin_color)
        cols[1].metric("Median Final Equity", f"${fe['p50']:,.0f}")
        cols[2].metric("95th Max Drawdown", f"${dd['p95']:,.0f}")
        cols[3].metric("95th Consec Losses", f"{cl['p95']:.0f}")

        # More detail
        cols2 = st.columns(4)
        cols2[0].metric("5th Final Equity", f"${fe['p5']:,.0f}")
        cols2[1].metric("95th Final Equity", f"${fe['p95']:,.0f}")
        cols2[2].metric("Mean Max DD", f"${dd['mean']:,.0f}")
        cols2[3].metric("Avg Consec Losses", f"{cl['mean']:.1f}")

        # Fan chart
        pc = mc.percentile_curves
        n_points = len(pc[50]) if 50 in pc and pc[50] else 0
        if n_points > 0:
            x_axis = list(range(n_points))

            fig_fan = go.Figure()

            # 5%-95% band
            fig_fan.add_trace(go.Scatter(
                x=x_axis, y=pc[95],
                mode="lines", line=dict(width=0),
                showlegend=False,
            ))
            fig_fan.add_trace(go.Scatter(
                x=x_axis, y=pc[5],
                mode="lines", line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(33,150,243,0.15)",
                name="5th-95th",
            ))

            # 25%-75% band
            fig_fan.add_trace(go.Scatter(
                x=x_axis, y=pc[75],
                mode="lines", line=dict(width=0),
                showlegend=False,
            ))
            fig_fan.add_trace(go.Scatter(
                x=x_axis, y=pc[25],
                mode="lines", line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(33,150,243,0.3)",
                name="25th-75th",
            ))

            # Median
            fig_fan.add_trace(go.Scatter(
                x=x_axis, y=pc[50],
                mode="lines",
                line=dict(color="#2196F3", width=2),
                name="Median (50th)",
            ))

            # Ruin line
            fig_fan.add_hline(
                y=mc.initial_capital - mc.loss_limit,
                line_dash="dash",
                line_color="#ff1744",
                annotation_text=f"Ruin (${mc.initial_capital - mc.loss_limit:,.0f})",
            )

            fig_fan.update_layout(
                template="plotly_dark",
                title=f"Monte Carlo Fan Chart ({mc.n_simulations:,} simulations)",
                xaxis_title="Trade #",
                yaxis_title="Equity ($)",
                height=450,
                margin=dict(l=60, r=20, t=50, b=30),
            )
            st.plotly_chart(fig_fan, use_container_width=True)

        # Distributions side by side
        col_dist1, col_dist2 = st.columns(2)

        with col_dist1:
            # Final equity distribution (approximate from stats)
            fig_fe = go.Figure()
            fe_vals = [fe["p5"], fe["p25"], fe["p50"], fe["p75"], fe["p95"]]
            fe_labels = ["5th", "25th", "50th", "75th", "95th"]
            fig_fe.add_trace(go.Bar(
                x=fe_labels, y=fe_vals,
                marker_color=["#ff1744", "#ff9800", "#2196F3", "#4caf50", "#00c853"],
            ))
            fig_fe.update_layout(
                template="plotly_dark",
                title="Final Equity Distribution",
                yaxis_title="Equity ($)",
                height=300,
                margin=dict(l=60, r=20, t=40, b=30),
            )
            st.plotly_chart(fig_fe, use_container_width=True)

        with col_dist2:
            # Max drawdown distribution
            fig_dd = go.Figure()
            dd_vals = [dd["p5"], dd["p25"], dd["p50"], dd["p75"], dd["p95"]]
            dd_labels = ["5th", "25th", "50th", "75th", "95th"]
            fig_dd.add_trace(go.Bar(
                x=dd_labels, y=dd_vals,
                marker_color=["#00c853", "#4caf50", "#ff9800", "#ff5722", "#ff1744"],
            ))
            fig_dd.add_hline(
                y=mc.loss_limit,
                line_dash="dash",
                line_color="#ff1744",
                annotation_text=f"Loss Limit (${mc.loss_limit:,.0f})",
            )
            fig_dd.update_layout(
                template="plotly_dark",
                title="Max Drawdown Distribution",
                yaxis_title="Drawdown ($)",
                height=300,
                margin=dict(l=60, r=20, t=40, b=30),
            )
            st.plotly_chart(fig_dd, use_container_width=True)

        st.caption(
            f"Simulations: {mc.n_simulations:,} | "
            f"Trades: {mc.n_trades} | "
            f"Capital: ${mc.initial_capital:,.0f} | "
            f"Loss Limit: ${mc.loss_limit:,.0f}"
        )


# ============================================================
# TAB 3: OPTIMIZATION
# ============================================================
with tabs[2]:
    opt = st.session_state.opt_result
    if opt is None:
        st.info("Click **Optimize** to run parameter search.")
    else:
        # Best params
        st.subheader("Best Parameters Found")
        cols_best = st.columns(min(len(opt.best_params), 6) or 1)
        for i, (k, v) in enumerate(opt.best_params.items()):
            col_idx = i % len(cols_best)
            cols_best[col_idx].metric(k, f"{v}")

        # Best metrics
        bm = opt.best_metrics
        cols_bm = st.columns(6)
        cols_bm[0].metric("Net Profit", f"${bm.get('net_profit', 0):,.2f}")
        cols_bm[1].metric("Win Rate", f"{bm.get('win_rate', 0):.1f}%")
        cols_bm[2].metric("Profit Factor", f"{bm.get('profit_factor', 0):.2f}")
        cols_bm[3].metric("Sharpe Ratio", f"{bm.get('sharpe_ratio', 0):.3f}")
        cols_bm[4].metric("Max Drawdown", f"${bm.get('max_drawdown', 0):,.2f}")
        cols_bm[5].metric("Total Trades", f"{bm.get('total_trades', 0)}")

        st.markdown("---")

        # Top-10 table
        st.subheader(f"Top 10 Results (by {opt.target_metric})")
        top_rows = []
        for i, r in enumerate(opt.all_results[:10]):
            row = {"Rank": i + 1}
            row.update(r["params"])
            row[opt.target_metric] = r["metrics"].get(opt.target_metric, 0)
            row["net_profit"] = r["metrics"].get("net_profit", 0)
            row["win_rate"] = r["metrics"].get("win_rate", 0)
            row["profit_factor"] = r["metrics"].get("profit_factor", 0)
            row["trades"] = r["metrics"].get("total_trades", 0)
            top_rows.append(row)
        if top_rows:
            st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)

        # Parameter sensitivity scatter plots
        if not opt.param_metric_df.empty and opt.param_names:
            st.subheader("Parameter Sensitivity")
            n_params = len(opt.param_names)
            n_cols = min(n_params, 3)
            scatter_cols = st.columns(n_cols)
            for i, pname in enumerate(opt.param_names):
                col_idx = i % n_cols
                with scatter_cols[col_idx]:
                    df = opt.param_metric_df
                    if pname in df.columns and opt.target_metric in df.columns:
                        fig_sc = go.Figure()
                        fig_sc.add_trace(go.Scatter(
                            x=df[pname],
                            y=df[opt.target_metric],
                            mode="markers",
                            marker=dict(
                                size=6,
                                color=df.get("net_profit", df[opt.target_metric]),
                                colorscale="RdYlGn",
                                showscale=False,
                                opacity=0.7,
                            ),
                        ))
                        fig_sc.update_layout(
                            template="plotly_dark",
                            title=pname,
                            xaxis_title=pname,
                            yaxis_title=opt.target_metric,
                            height=280,
                            margin=dict(l=50, r=10, t=35, b=30),
                        )
                        st.plotly_chart(fig_sc, use_container_width=True)

        st.caption(
            f"Iterations: {opt.n_iterations} | "
            f"Target: {opt.target_metric} | "
            f"Elapsed: {opt.elapsed_sec:.1f}s"
        )


# ============================================================
# TAB 4: TRADE LOG
# ============================================================
with tabs[3]:
    rd = st.session_state.bt_result_dict
    if rd is None:
        st.info("Run a backtest first to see the trade log.")
    else:
        trades = rd.get("trades", [])
        if not trades:
            st.warning("No trades in this backtest run.")
        else:
            trade_df = pd.DataFrame(trades)
            # Format columns
            display_cols = ["entry_time", "exit_time", "direction", "quantity",
                           "entry_price", "exit_price", "pnl", "commission", "bars_held"]
            available_cols = [c for c in display_cols if c in trade_df.columns]
            st.dataframe(
                trade_df[available_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "pnl": st.column_config.NumberColumn("P&L", format="$%.2f"),
                    "commission": st.column_config.NumberColumn("Commission", format="$%.2f"),
                    "entry_price": st.column_config.NumberColumn("Entry", format="%.2f"),
                    "exit_price": st.column_config.NumberColumn("Exit", format="%.2f"),
                },
            )

            # Summary
            total_pnl = sum(t["pnl"] for t in trades)
            wins = sum(1 for t in trades if t["pnl"] > 0)
            st.caption(
                f"Total: {len(trades)} trades | "
                f"Wins: {wins} | Losses: {len(trades) - wins} | "
                f"Net P&L: ${total_pnl:,.2f}"
            )
