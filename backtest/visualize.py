"""
Backtest visualization module.

Generates interactive HTML charts (Plotly) or static PNG charts (matplotlib)
from BacktestResult data.

Usage from CLI:
    python run_backtest.py --strategy "KDJ RSI Bot" --instrument MES ... --output results.json
    python -m backtest.visualize results.json

Usage from Python:
    from backtest.visualize import plot_backtest
    plot_backtest(result)            # opens in browser (Plotly)
    plot_backtest(result, "png")     # saves equity.png + trades.png
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Plotly-based interactive charts (preferred)
# ---------------------------------------------------------------------------

def plot_backtest_plotly(
    result: dict[str, Any],
    output_path: str | None = None,
) -> None:
    """Generate an interactive HTML dashboard with Plotly.

    Contains:
        1. Equity curve with drawdown shading
        2. Trade markers (green = win, red = loss)
        3. P&L distribution histogram

    Args:
        result:      BacktestResult as a dict (from JSON export).
        output_path: Path for HTML output. If None, opens in browser.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
        print("Falling back to matplotlib...")
        plot_backtest_matplotlib(result, output_path)
        return

    equity_curve = result.get("equity_curve", [])
    trades = result.get("trades", [])
    metrics = result.get("metrics", {})
    config = result.get("config", {})

    strategy_name = config.get("strategy_name", "Unknown")
    instrument = config.get("instrument", "?")
    account = config.get("account_tier", "?")

    # -- Parse equity curve --
    eq_times = [ep["timestamp"] for ep in equity_curve]
    eq_values = [ep["equity"] for ep in equity_curve]

    # Compute drawdown series
    peak = 0.0
    drawdowns = []
    for v in eq_values:
        if v > peak:
            peak = v
        drawdowns.append(v - peak)  # negative

    # -- Build figure --
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(
            "Equity Curve",
            "Drawdown",
            "Trade P&L",
        ),
        row_heights=[0.5, 0.25, 0.25],
    )

    # 1. Equity curve
    fig.add_trace(
        go.Scatter(
            x=eq_times, y=eq_values,
            mode="lines",
            name="Equity",
            line=dict(color="#2196F3", width=1.5),
        ),
        row=1, col=1,
    )

    # Mark trades on equity curve
    win_x, win_y = [], []
    loss_x, loss_y = [], []
    for t in trades:
        exit_time = t["exit_time"]
        pnl = t["pnl"]
        # Find nearest equity point
        idx = _find_nearest_index(eq_times, exit_time)
        if idx is not None:
            if pnl >= 0:
                win_x.append(eq_times[idx])
                win_y.append(eq_values[idx])
            else:
                loss_x.append(eq_times[idx])
                loss_y.append(eq_values[idx])

    if win_x:
        fig.add_trace(
            go.Scatter(
                x=win_x, y=win_y,
                mode="markers",
                name="Winning Trade",
                marker=dict(color="#4CAF50", size=6, symbol="triangle-up"),
            ),
            row=1, col=1,
        )
    if loss_x:
        fig.add_trace(
            go.Scatter(
                x=loss_x, y=loss_y,
                mode="markers",
                name="Losing Trade",
                marker=dict(color="#F44336", size=6, symbol="triangle-down"),
            ),
            row=1, col=1,
        )

    # 2. Drawdown
    fig.add_trace(
        go.Scatter(
            x=eq_times, y=drawdowns,
            fill="tozeroy",
            mode="lines",
            name="Drawdown",
            line=dict(color="#F44336", width=1),
            fillcolor="rgba(244,67,54,0.2)",
        ),
        row=2, col=1,
    )

    # 3. Trade P&L bars
    trade_exits = [t["exit_time"] for t in trades]
    trade_pnls = [t["pnl"] for t in trades]
    bar_colors = ["#4CAF50" if p >= 0 else "#F44336" for p in trade_pnls]

    fig.add_trace(
        go.Bar(
            x=trade_exits,
            y=trade_pnls,
            name="Trade P&L",
            marker_color=bar_colors,
        ),
        row=3, col=1,
    )

    # -- Layout --
    title_text = (
        f"{strategy_name} | {instrument} | {account} Account | "
        f"Net: ${metrics.get('net_profit', 0):,.2f} | "
        f"Trades: {metrics.get('total_trades', 0)} | "
        f"Win: {metrics.get('win_rate', 0):.1f}% | "
        f"Sharpe: {metrics.get('sharpe_ratio', 0):.3f}"
    )

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=14)),
        height=900,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )

    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown ($)", row=2, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=3, col=1)

    if output_path:
        out = Path(output_path)
        if out.suffix != ".html":
            out = out.with_suffix(".html")
        fig.write_html(str(out), auto_open=False)
        print(f"Chart saved to: {out}")
    else:
        fig.show()


# ---------------------------------------------------------------------------
# Matplotlib-based static charts (fallback)
# ---------------------------------------------------------------------------

def plot_backtest_matplotlib(
    result: dict[str, Any],
    output_path: str | None = None,
) -> None:
    """Generate static PNG charts with matplotlib.

    Args:
        result:      BacktestResult as a dict.
        output_path: Base path for PNG output. Generates {path}_equity.png
                     and {path}_trades.png. If None, displays interactively.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    equity_curve = result.get("equity_curve", [])
    trades = result.get("trades", [])
    metrics = result.get("metrics", {})
    config = result.get("config", {})

    strategy_name = config.get("strategy_name", "Unknown")
    instrument = config.get("instrument", "?")

    # Parse timestamps
    eq_times = [_parse_ts(ep["timestamp"]) for ep in equity_curve]
    eq_values = [ep["equity"] for ep in equity_curve]

    # Drawdown
    peak = 0.0
    drawdowns = []
    for v in eq_values:
        if v > peak:
            peak = v
        drawdowns.append(v - peak)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    fig.suptitle(
        f"{strategy_name} | {instrument} | "
        f"Net: ${metrics.get('net_profit', 0):,.2f} | "
        f"Trades: {metrics.get('total_trades', 0)}",
        fontsize=13, fontweight="bold",
    )

    # 1. Equity
    ax1 = axes[0]
    ax1.plot(eq_times, eq_values, color="#2196F3", linewidth=1.2, label="Equity")
    ax1.set_ylabel("Equity ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown
    ax2 = axes[1]
    ax2.fill_between(eq_times, drawdowns, color="#F44336", alpha=0.3)
    ax2.plot(eq_times, drawdowns, color="#F44336", linewidth=0.8)
    ax2.set_ylabel("Drawdown ($)")
    ax2.grid(True, alpha=0.3)

    # 3. Trade P&L
    ax3 = axes[2]
    trade_exits = [_parse_ts(t["exit_time"]) for t in trades]
    trade_pnls = [t["pnl"] for t in trades]
    colors = ["#4CAF50" if p >= 0 else "#F44336" for p in trade_pnls]
    ax3.bar(trade_exits, trade_pnls, color=colors, width=0.5)
    ax3.set_ylabel("P&L ($)")
    ax3.axhline(y=0, color="gray", linewidth=0.5)
    ax3.grid(True, alpha=0.3)

    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if output_path:
        out = Path(output_path).with_suffix(".png")
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"Chart saved to: {out}")
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def plot_backtest(
    result: dict[str, Any],
    mode: str = "plotly",
    output_path: str | None = None,
) -> None:
    """Plot backtest results.

    Args:
        result:      BacktestResult as a dict.
        mode:        "plotly" (interactive HTML) or "matplotlib" (static PNG).
        output_path: Output file path, or None to display.
    """
    if mode == "plotly":
        plot_backtest_plotly(result, output_path)
    elif mode in ("matplotlib", "mpl", "png"):
        plot_backtest_matplotlib(result, output_path)
    else:
        raise ValueError(f"Unknown plot mode: {mode}. Use 'plotly' or 'matplotlib'.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts_str: str) -> datetime:
    """Parse a timestamp string into a datetime object."""
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(ts_str, fmt)
        except ValueError:
            continue
    return datetime.fromisoformat(ts_str)


def _find_nearest_index(timestamps: list[str], target: str) -> int | None:
    """Find index of the nearest timestamp (string comparison)."""
    if not timestamps:
        return None
    # Simple linear scan (equity curves are ordered)
    for i, ts in enumerate(timestamps):
        if ts >= target:
            return i
    return len(timestamps) - 1


# ---------------------------------------------------------------------------
# Monte Carlo visualization
# ---------------------------------------------------------------------------

def plot_monte_carlo(
    mc_result,
    output_path: str | None = None,
) -> None:
    """Generate interactive Monte Carlo simulation chart.

    Three panels:
        1. Fan chart: percentile equity curves (5th-95th)
        2. Final equity distribution histogram
        3. Max drawdown distribution histogram

    Args:
        mc_result:   MonteCarloResult dataclass instance.
        output_path: Path for HTML output. If None, opens in browser.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
        return

    pct_curves = mc_result.percentile_curves
    trade_indices = list(range(mc_result.n_trades + 1))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Equity Curve Confidence Band (Monte Carlo)",
            "Final Equity Distribution",
            "Max Drawdown Distribution",
            "Max Consecutive Losses Distribution",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )

    # -- Panel 1: Fan chart --
    # 5th-95th band
    fig.add_trace(
        go.Scatter(
            x=trade_indices, y=pct_curves[95],
            mode="lines", line=dict(width=0),
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=trade_indices, y=pct_curves[5],
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(33,150,243,0.15)",
            name="5th-95th Percentile",
        ),
        row=1, col=1,
    )
    # 25th-75th band
    fig.add_trace(
        go.Scatter(
            x=trade_indices, y=pct_curves[75],
            mode="lines", line=dict(width=0),
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=trade_indices, y=pct_curves[25],
            mode="lines", line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(33,150,243,0.3)",
            name="25th-75th Percentile",
        ),
        row=1, col=1,
    )
    # 50th median line
    fig.add_trace(
        go.Scatter(
            x=trade_indices, y=pct_curves[50],
            mode="lines",
            line=dict(color="#2196F3", width=2),
            name="Median (50th)",
        ),
        row=1, col=1,
    )
    # Ruin line
    ruin_line = mc_result.initial_capital - mc_result.loss_limit
    fig.add_hline(
        y=ruin_line, line_dash="dash", line_color="red",
        annotation_text=f"Ruin ${ruin_line:,.0f}",
        row=1, col=1,
    )

    # -- Panel 2: Final equity histogram --
    fe = mc_result.final_equity_stats
    # Generate samples from stats for histogram
    n_bins = 50
    fig.add_trace(
        go.Histogram(
            x=_reconstruct_samples(fe, mc_result.n_simulations),
            nbinsx=n_bins,
            marker_color="rgba(33,150,243,0.7)",
            name="Final Equity",
        ),
        row=1, col=2,
    )
    fig.add_vline(
        x=ruin_line, line_dash="dash", line_color="red",
        annotation_text=f"Ruin: {mc_result.ruin_probability:.1%}",
        row=1, col=2,
    )

    # -- Panel 3: Max drawdown histogram --
    dd = mc_result.max_drawdown_stats
    fig.add_trace(
        go.Histogram(
            x=_reconstruct_samples(dd, mc_result.n_simulations),
            nbinsx=n_bins,
            marker_color="rgba(244,67,54,0.7)",
            name="Max Drawdown",
        ),
        row=2, col=1,
    )
    fig.add_vline(
        x=mc_result.loss_limit, line_dash="dash", line_color="red",
        annotation_text=f"Loss Limit ${mc_result.loss_limit:,.0f}",
        row=2, col=1,
    )

    # -- Panel 4: Max consecutive losses histogram --
    cl = mc_result.max_consec_loss_stats
    fig.add_trace(
        go.Histogram(
            x=_reconstruct_samples(cl, mc_result.n_simulations),
            nbinsx=min(n_bins, int(cl["max"]) + 1) if cl["max"] > 0 else 10,
            marker_color="rgba(255,152,0,0.7)",
            name="Max Consec. Losses",
        ),
        row=2, col=2,
    )

    # Layout
    title = (
        f"Monte Carlo Simulation | {mc_result.n_simulations:,} runs x "
        f"{mc_result.n_trades} trades | "
        f"Ruin Prob: {mc_result.ruin_probability:.1%} | "
        f"Median Final: ${fe['p50']:,.0f}"
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=800,
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(title_text="Trade #", row=1, col=1)
    fig.update_yaxes(title_text="Equity ($)", row=1, col=1)
    fig.update_xaxes(title_text="Final Equity ($)", row=1, col=2)
    fig.update_xaxes(title_text="Max Drawdown ($)", row=2, col=1)
    fig.update_xaxes(title_text="Max Consecutive Losses", row=2, col=2)

    if output_path:
        out = Path(output_path)
        if out.suffix != ".html":
            out = out.with_suffix(".html")
        fig.write_html(str(out), auto_open=False)
        print(f"Monte Carlo chart saved to: {out}")
    else:
        fig.show()


def _reconstruct_samples(stats: dict, n: int) -> list[float]:
    """Approximate a sample distribution from summary statistics using
    percentile-based interpolation. Good enough for histogram visualization."""
    import numpy as np
    # Use percentile points to create a rough distribution
    percentiles = [stats["p5"], stats["p25"], stats["p50"], stats["p75"], stats["p95"]]
    # Linear interpolation between percentile points
    rng = np.random.default_rng(42)
    samples = []
    weights = [0.05, 0.20, 0.25, 0.25, 0.20, 0.05]
    bounds = [stats["min"]] + percentiles + [stats["max"]]
    for i in range(len(bounds) - 1):
        count = max(1, int(n * weights[i]))
        segment = rng.uniform(bounds[i], bounds[i + 1], size=count)
        samples.extend(segment.tolist())
    return samples[:n]


# ---------------------------------------------------------------------------
# Optimization visualization
# ---------------------------------------------------------------------------

def plot_optimization(
    opt_result,
    output_path: str | None = None,
) -> None:
    """Generate interactive optimization results chart.

    Shows parameter sensitivity and best results.

    Args:
        opt_result:  OptimizationResult dataclass instance.
        output_path: Path for HTML output. If None, opens in browser.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Plotly not installed.")
        return

    df = opt_result.param_metric_df
    if df.empty:
        print("No optimization results to plot.")
        return

    target = opt_result.target_metric
    params = opt_result.param_names

    # Determine subplot layout: parameter scatter plots + top results table
    n_params = len(params)
    n_cols = min(3, n_params)
    n_rows = max(1, -(-n_params // n_cols))  # ceiling division
    n_rows += 1  # extra row for top results

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"{p} vs {target}" for p in params] + [""] * (n_rows * n_cols - n_params),
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    # Color scale based on target metric
    metric_vals = df[target].dropna()
    if len(metric_vals) == 0:
        return

    color_min = metric_vals.min()
    color_max = metric_vals.max()

    # Scatter plots: each parameter vs target metric
    for idx, param in enumerate(params):
        row = idx // n_cols + 1
        col = idx % n_cols + 1

        fig.add_trace(
            go.Scatter(
                x=df[param],
                y=df[target],
                mode="markers",
                marker=dict(
                    size=5,
                    color=df[target],
                    colorscale="RdYlGn",
                    cmin=color_min,
                    cmax=color_max,
                    opacity=0.6,
                ),
                name=param,
                showlegend=False,
                text=[f"{target}={v:.4f}" for v in df[target]],
            ),
            row=row, col=col,
        )
        fig.update_xaxes(title_text=param, row=row, col=col)
        fig.update_yaxes(title_text=target, row=row, col=col)

    # Top-10 results as annotation
    top10 = opt_result.all_results[:10]
    table_text = f"<b>Top 10 Results ({target})</b><br>"
    table_text += "-" * 60 + "<br>"
    for i, r in enumerate(top10):
        metric_val = r["metrics"].get(target, 0)
        net = r["metrics"].get("net_profit", 0)
        wr = r["metrics"].get("win_rate", 0)
        params_str = ", ".join(f"{k}={v}" for k, v in list(r["params"].items())[:4])
        table_text += f"#{i+1}: {target}={metric_val:.4f} | net=${net:,.0f} | wr={wr:.1f}% | {params_str}<br>"

    # Layout
    best_val = opt_result.best_metrics.get(target, "N/A")
    title = (
        f"Parameter Optimization | {opt_result.n_iterations} iterations | "
        f"Best {target} = {best_val} | "
        f"Elapsed: {opt_result.elapsed_sec:.1f}s"
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        height=300 * n_rows,
        template="plotly_dark",
        showlegend=False,
        annotations=[
            dict(
                text=table_text,
                xref="paper", yref="paper",
                x=0.0, y=-0.05,
                showarrow=False,
                font=dict(family="monospace", size=10),
                align="left",
            )
        ],
    )

    if output_path:
        out = Path(output_path)
        if out.suffix != ".html":
            out = out.with_suffix(".html")
        fig.write_html(str(out), auto_open=False)
        print(f"Optimization chart saved to: {out}")
    else:
        fig.show()


# ---------------------------------------------------------------------------
# Full research report
# ---------------------------------------------------------------------------

def plot_full_report(
    backtest_result: dict[str, Any],
    mc_result=None,
    opt_result=None,
    output_path: str | None = None,
) -> None:
    """Generate a combined HTML report with all analysis.

    Saves individual charts and combines references.
    """
    out_base = Path(output_path) if output_path else Path("research_report")
    out_dir = out_base.parent
    stem = out_base.stem

    # Save individual charts
    if backtest_result:
        bt_path = out_dir / f"{stem}_backtest.html"
        plot_backtest_plotly(backtest_result, str(bt_path))

    if mc_result:
        mc_path = out_dir / f"{stem}_montecarlo.html"
        plot_monte_carlo(mc_result, str(mc_path))

    if opt_result:
        opt_path = out_dir / f"{stem}_optimization.html"
        plot_optimization(opt_result, str(opt_path))

    print(f"\nFull report generated:")
    if backtest_result:
        print(f"  Backtest:     {bt_path}")
    if mc_result:
        print(f"  Monte Carlo:  {mc_path}")
    if opt_result:
        print(f"  Optimization: {opt_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load a JSON result file and plot it."""
    if len(sys.argv) < 2:
        print("Usage: python -m backtest.visualize <results.json> [--mode plotly|matplotlib] [--output chart.html]")
        sys.exit(1)

    json_path = sys.argv[1]
    mode = "plotly"
    output = None

    # Parse optional args
    args = sys.argv[2:]
    i = 0
    while i < len(args):
        if args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
        elif args[i] == "--output" and i + 1 < len(args):
            output = args[i + 1]
            i += 2
        else:
            i += 1

    with open(json_path, "r") as f:
        result = json.load(f)

    plot_backtest(result, mode=mode, output_path=output)


if __name__ == "__main__":
    main()
