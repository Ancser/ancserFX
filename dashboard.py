#!/usr/bin/env python3
"""
Streamlit interactive dashboard for ancserFX backtesting.

Usage:
    streamlit run dashboard.py
"""

from __future__ import annotations

import json
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
from data.loader import DataLoader
from data.models import INSTRUMENT_SPECS
from strategies.registry import StrategyRegistry


def _get_available_timeframes(instrument: str) -> list[str]:
    """Scan parquet directory to find which timeframes have data."""
    data_dir = Path(__file__).resolve().parent / "data" / "parquet" / instrument.lower()
    all_tfs = ["1min", "5min", "15min", "1h", "daily"]
    available = []
    for tf in all_tfs:
        pf = data_dir / tf / "data.parquet"
        if pf.exists():
            available.append(tf)
    return available if available else ["5min"]  # fallback

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
PRESETS_DIR = Path(__file__).resolve().parent / "presets"
PRESETS_DIR.mkdir(exist_ok=True)

# Parameter grouping for KDJ RSI Bot (Chinese labels)
PARAM_GROUPS = {
    "KDJ RSI Bot": {
        "KDJ 隨機指標": ["kdj_period", "kdj_signal"],
        "RSI 相對強弱": ["rsi_period", "rsi_threshold", "d_cross_level"],
        "止損 Stop Loss": ["sl_points"],
        "止盈 Take Profit": ["tp1_ticks", "tp1_pct", "tp2_ticks", "tp2_pct", "tp3_ticks"],
    },
    "SMA Crossover": {
        "均線 Moving Avg": ["fast_period", "slow_period"],
        "波動率 ATR": ["atr_period", "atr_sl_mult", "atr_tp_mult"],
    },
    "RSI Mean Reversion": {
        "RSI 參數": ["rsi_period", "oversold", "overbought", "exit_level"],
        "趨勢過濾 Trend": ["ema_filter_period"],
    },
    "Delta Momentum": {
        "Delta 動量": ["delta_ema_period", "imbalance_lookback", "imbalance_threshold"],
        "退出/趨勢 Exit": ["exit_imbalance", "trend_ema_period"],
    },
}

# Chinese labels for parameters
PARAM_LABELS = {
    "kdj_period":       "KDJ週期 kdj_period",
    "kdj_signal":       "KDJ信號 kdj_signal",
    "rsi_period":       "RSI週期 rsi_period",
    "rsi_threshold":    "RSI閾值 rsi_threshold",
    "d_cross_level":    "D線交叉 d_cross_level",
    "sl_points":        "止損點數 sl_points",
    "tp1_ticks":        "止盈1距離 tp1_ticks",
    "tp1_pct":          "止盈1比例% tp1_pct",
    "tp2_ticks":        "止盈2距離 tp2_ticks",
    "tp2_pct":          "止盈2比例% tp2_pct",
    "tp3_ticks":        "止盈3距離 tp3_ticks",
    "fast_period":      "快線週期 fast_period",
    "slow_period":      "慢線週期 slow_period",
    "atr_period":       "ATR週期 atr_period",
    "atr_sl_mult":      "ATR止損倍數 atr_sl_mult",
    "atr_tp_mult":      "ATR止盈倍數 atr_tp_mult",
    "oversold":         "超賣線 oversold",
    "overbought":       "超買線 overbought",
    "exit_level":       "退出線 exit_level",
    "ema_filter_period":"EMA過濾週期 ema_filter",
    "delta_ema_period": "Delta EMA delta_ema",
    "imbalance_lookback":"失衡回看 imb_lookback",
    "imbalance_threshold":"失衡閾值 imb_threshold",
    "exit_imbalance":   "退出失衡 exit_imbalance",
    "trend_ema_period": "趨勢EMA trend_ema",
}

# TopStep Best Day limits (consistency rule)
BEST_DAY_LIMITS = {
    "50K": 1_500.0,
    "100K": 3_000.0,
    "150K": 4_500.0,
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ancserFX Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric { background-color: #1e1e1e; padding: 10px; border-radius: 8px; }
    div[data-testid="stMetricValue"] { font-size: 1.3rem; }
    .param-card { border: 1px solid #333; border-radius: 8px; padding: 8px; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
for key in ["bt_result", "bt_result_dict", "mc_result", "opt_result", "trades_pnl", "wfa_result"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Stores the best params from optimizer so sidebar sliders auto-update
if "active_params" not in st.session_state:
    st.session_state.active_params = None


# ---------------------------------------------------------------------------
# Data availability calendar (GitHub-style heatmap)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def _build_data_calendar() -> pd.DataFrame:
    """Scan all parquet files and build a month-level availability matrix.

    Returns a DataFrame: rows = 'INSTRUMENT/timeframe', columns = 'YYYY-MM',
    values = bar count (0 if no data).
    """
    from data.store import DataStore
    store = DataStore()
    records: list[dict] = []
    for inst in store.list_instruments():
        for tf in store.list_timeframes(inst):
            try:
                df = store.load_bars(inst, tf)
                if df.empty:
                    continue
                monthly = df.groupby(df["timestamp"].dt.to_period("M")).size()
                for period, count in monthly.items():
                    records.append({
                        "instrument": inst,
                        "timeframe": tf,
                        "label": f"{inst}/{tf}",
                        "month": str(period),
                        "bars": int(count),
                    })
            except Exception:
                continue
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _render_data_calendar(cal_df: pd.DataFrame, selected_tf: str | None = None):
    """Render a compact data availability heatmap in the sidebar."""
    if cal_df.empty:
        st.caption("📅 無數據")
        return

    # Filter by selected timeframe if given
    if selected_tf:
        filtered = cal_df[cal_df["timeframe"] == selected_tf]
    else:
        filtered = cal_df

    if filtered.empty:
        st.caption(f"📅 {selected_tf} 無可用數據")
        return

    # Pivot: rows = instrument, columns = month
    pivot = filtered.groupby(["instrument", "month"])["bars"].sum().reset_index()
    pivot_table = pivot.pivot(index="instrument", columns="month", values="bars").fillna(0)
    pivot_table = pivot_table.reindex(sorted(pivot_table.columns), axis=1)

    # Build Plotly heatmap
    instruments = list(pivot_table.index)
    months = list(pivot_table.columns)
    z = pivot_table.values

    # Custom colorscale: 0 = dark gray, >0 = shades of green
    colorscale = [[0, "#2a2a2a"], [0.001, "#2a2a2a"], [0.002, "#1b5e20"], [0.3, "#388e3c"], [0.6, "#4caf50"], [1.0, "#81c784"]]

    # Short month labels (show year only at Jan or first)
    short_labels = []
    for m in months:
        parts = m.split("-")
        if len(parts) == 2:
            if parts[1] == "01" or m == months[0]:
                short_labels.append(f"{parts[0]}\n{parts[1]}")
            else:
                short_labels.append(parts[1])
        else:
            short_labels.append(m)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=short_labels,
        y=instruments,
        colorscale=colorscale,
        showscale=False,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:,} bars<extra></extra>",
        xgap=1, ygap=1,
    ))
    fig.update_layout(
        template="plotly_dark",
        height=max(60, 30 * len(instruments) + 40),
        margin=dict(l=50, r=5, t=5, b=30),
        xaxis=dict(tickfont=dict(size=8), dtick=1),
        yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)


def _get_data_date_range(cal_df: pd.DataFrame, instrument: str, timeframe: str) -> tuple[str, str]:
    """Return (first_month, last_month) for a specific instrument+timeframe."""
    sub = cal_df[(cal_df["instrument"] == instrument) & (cal_df["timeframe"] == timeframe)]
    if sub.empty:
        return "", ""
    months = sorted(sub["month"].unique())
    # Convert period to start/end dates
    first = months[0] + "-01"
    last_period = pd.Period(months[-1], freq="M")
    last = str(last_period.end_time.date())
    return first, last


# ---------------------------------------------------------------------------
# Preset helpers
# ---------------------------------------------------------------------------
def _get_presets_file(strategy_name: str) -> Path:
    safe_name = strategy_name.replace(" ", "_").lower()
    return PRESETS_DIR / f"{safe_name}_presets.json"


def _load_presets(strategy_name: str) -> list[dict]:
    f = _get_presets_file(strategy_name)
    if f.exists():
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save_presets(strategy_name: str, presets: list[dict]):
    f = _get_presets_file(strategy_name)
    f.write_text(json.dumps(presets, indent=2, ensure_ascii=False), encoding="utf-8")


def _save_optimizer_presets(strategy_name: str, opt_result, min_trades: int = 5):
    """Auto-save top 10 results with enough trades as presets."""
    filtered = [
        r for r in opt_result.all_results
        if r["metrics"].get("total_trades", 0) >= min_trades
    ]
    presets = []
    for i, r in enumerate(filtered[:10]):
        presets.append({
            "name": f"Opt#{i+1} ({opt_result.target_metric}={r['metrics'].get(opt_result.target_metric, 0):.4f}, {r['metrics'].get('total_trades', 0)} trades)",
            "params": r["params"],
            "metrics_summary": {
                "net_profit": r["metrics"].get("net_profit", 0),
                "win_rate": r["metrics"].get("win_rate", 0),
                "profit_factor": r["metrics"].get("profit_factor", 0),
                "sharpe_ratio": r["metrics"].get("sharpe_ratio", 0),
                "max_drawdown": r["metrics"].get("max_drawdown", 0),
                "total_trades": r["metrics"].get("total_trades", 0),
            },
            "source": "optimizer",
        })
    # Merge with existing manual presets
    existing = [p for p in _load_presets(strategy_name) if p.get("source") != "optimizer"]
    _save_presets(strategy_name, existing + presets)
    return len(presets)


# ---------------------------------------------------------------------------
# Sidebar — Strategy & Global settings
# ---------------------------------------------------------------------------
st.sidebar.title("ancserFX")

# Strategy selector
all_strategies = StrategyRegistry.list_strategies()
strat_names = [s["name"] for s in all_strategies]
strat_idx = st.sidebar.selectbox(
    "策略 Strategy",
    range(len(strat_names)),
    format_func=lambda i: strat_names[i],
)
selected_strat = all_strategies[strat_idx]
strat_name = selected_strat["name"]

# Preset loader
presets = _load_presets(strat_name)
preset_names = ["-- 默認 Default --"] + [p["name"] for p in presets]
preset_idx = st.sidebar.selectbox("預設 Preset", range(len(preset_names)), format_func=lambda i: preset_names[i])
preset_params = presets[preset_idx - 1]["params"] if preset_idx > 0 else None

# Selecting a preset clears the optimizer active_params
if preset_idx > 0 and st.session_state.get("active_params"):
    st.session_state.active_params = None

# Instrument
instruments = list(INSTRUMENT_SPECS.keys())
instrument = st.sidebar.selectbox(
    "合約 Instrument",
    instruments,
    index=instruments.index("NQ") if "NQ" in instruments else 0,
)
inst_spec = INSTRUMENT_SPECS[instrument]
st.sidebar.caption(f"Tick: {inst_spec['tick_size']} = ${inst_spec['tick_value']} | 1點 = ${inst_spec['tick_value']/inst_spec['tick_size']:.2f}")

# Account tier
account_tier = st.sidebar.selectbox("賬戶 Account", ["50K", "100K", "150K"], index=2)
rules = TOPSTEP_ACCOUNTS.get(account_tier)
best_day_limit = BEST_DAY_LIMITS.get(account_tier, 1500)
st.sidebar.caption(
    f"💰 ${rules.account_size:,.0f} | "
    f"🔴 回撤限制 ${rules.max_loss_limit:,.0f} | "
    f"📊 最大合約 {rules.max_contracts} | "
    f"⚡ Best Day < ${best_day_limit:,.0f}"
)

# Quantity
quantity = st.sidebar.number_input("數量 Quantity", min_value=1, max_value=rules.max_contracts, value=1)

# Timeframe (only show available data)
available_tfs = _get_available_timeframes(instrument)
tf_default_idx = available_tfs.index("5min") if "5min" in available_tfs else 0
timeframe = st.sidebar.selectbox("週期 Timeframe", available_tfs, index=tf_default_idx)

# Data availability calendar
_cal_df = _build_data_calendar()
with st.sidebar.expander("📅 數據日曆 Data Calendar", expanded=False):
    _render_data_calendar(_cal_df, selected_tf=timeframe)
    _auto_start, _auto_end = _get_data_date_range(_cal_df, instrument, timeframe)
    if _auto_start:
        st.caption(f"可用範圍: {_auto_start} ~ {_auto_end}")

# Date range (auto-fill from calendar if available)
col_d1, col_d2 = st.sidebar.columns(2)
_default_start = _auto_start if _auto_start else "2024-01-01"
_default_end = _auto_end if _auto_end else "2024-06-30"
start_date = col_d1.text_input("開始 Start", value="2024-01-01")
end_date = col_d2.text_input("結束 End", value="2024-06-30")

# Slippage & Commission
col_s1, col_s2 = st.sidebar.columns(2)
slippage = col_s1.number_input("滑點 Slippage", min_value=0, max_value=10, value=1)
commission = col_s2.number_input("手續費 Commission", min_value=0.0, max_value=20.0, value=2.50, step=0.50)

# Risk controls
col_r1, col_r2 = st.sidebar.columns(2)
circuit_breaker = col_r1.checkbox("🔌 斷路器 CB", value=True, help="開啟=SL虧損>=剩餘預算時拒單; 關閉=允許冒險交易")
use_best_day = col_r2.checkbox("📅 Best Day", value=True, help="單日盈利超限自動暫停當天交易")
best_day_val = float(best_day_limit) if use_best_day else 0.0

# ---------------------------------------------------------------------------
# Sidebar — Strategy Parameters (grouped with Chinese labels)
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("策略參數 Strategy Params")

# Show badge + clear button when optimizer params are active
if st.session_state.active_params:
    _ap_col1, _ap_col2 = st.sidebar.columns([3, 1])
    _ap_col1.success("🏆 優化最佳參數已加載")
    if _ap_col2.button("✖", help="清除優化參數，恢復默認 Clear optimizer params"):
        st.session_state.active_params = None
        st.rerun()

strategy_params: dict = {}
params_dict = selected_strat.get("parameters", {})
groups = PARAM_GROUPS.get(strat_name, {})

# Gather all params that belong to a group
grouped_params = set()
for group_params in groups.values():
    grouped_params.update(group_params)


def _render_slider(pname, pinfo, container):
    """Render a single parameter slider and return the value."""
    ptype = pinfo.get("type", "int")
    pmin = pinfo.get("min")
    pmax = pinfo.get("max")
    default = pinfo.get("default")
    step = pinfo.get("step", 1)
    desc = pinfo.get("description", "")
    label = PARAM_LABELS.get(pname, pname)

    # Priority: active_params (from optimizer) > preset > default
    ap = st.session_state.get("active_params")
    if ap and pname in ap:
        default = ap[pname]
    elif preset_params and pname in preset_params:
        default = preset_params[pname]

    if pmin is not None and pmax is not None:
        if ptype == "int":
            return container.slider(
                label, min_value=int(pmin), max_value=int(pmax),
                value=int(default), step=int(step) if step else 1, help=desc,
            )
        elif ptype == "float":
            return container.slider(
                label, min_value=float(pmin), max_value=float(pmax),
                value=float(default), step=float(step) if step else 0.1, help=desc,
            )
        elif ptype == "bool":
            return container.checkbox(label, value=bool(default), help=desc)
    return container.text_input(label, value=str(default), help=desc)


# Render grouped parameters in expandable cards
for group_label, group_param_names in groups.items():
    with st.sidebar.expander(f"📦 {group_label}", expanded=True):
        for pname in group_param_names:
            if pname in params_dict:
                strategy_params[pname] = _render_slider(pname, params_dict[pname], st)

# Render any ungrouped parameters
ungrouped = [p for p in params_dict if p not in grouped_params]
if ungrouped:
    with st.sidebar.expander("其他 Other", expanded=True):
        for pname in ungrouped:
            strategy_params[pname] = _render_slider(pname, params_dict[pname], st)

# Manual save preset button
st.sidebar.markdown("---")
col_save1, col_save2 = st.sidebar.columns([3, 1])
preset_save_name = col_save1.text_input("保存名稱 Preset Name", value="", placeholder="My Preset")
if col_save2.button("💾", help="保存當前參數為預設"):
    if preset_save_name.strip():
        existing = _load_presets(strat_name)
        existing.append({
            "name": preset_save_name.strip(),
            "params": dict(strategy_params),
            "source": "manual",
        })
        _save_presets(strat_name, existing)
        st.sidebar.success(f"已保存: {preset_save_name}")
        st.rerun()


# ---------------------------------------------------------------------------
# Build config helper
# ---------------------------------------------------------------------------
def _build_config(params_override: dict | None = None) -> BacktestConfig:
    params = params_override or strategy_params
    return BacktestConfig(
        strategy_name=strat_name,
        strategy_params=params,
        instrument=instrument,
        timeframe=timeframe,
        start_date=start_date or None,
        end_date=end_date or None,
        account_tier=account_tier,
        quantity=quantity,
        slippage_ticks=slippage,
        commission=commission,
        circuit_breaker=circuit_breaker,
        best_day_limit=best_day_val,
    )


# ---------------------------------------------------------------------------
# Action buttons
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")

col_btn1, col_btn2 = st.sidebar.columns(2)
run_bt = col_btn1.button("▶ 回測 Backtest", type="primary", use_container_width=True)
run_mc = col_btn2.button("🎲 蒙特卡洛 MC", use_container_width=True)

col_btn3, col_btn4 = st.sidebar.columns(2)
run_opt = col_btn3.button("🔍 優化 Optimize", use_container_width=True)
run_full = col_btn4.button("🚀 全流程 Full", use_container_width=True)

col_btn5, col_btn6 = st.sidebar.columns(2)
run_wfa = col_btn5.button("🔄 前推 WFA", use_container_width=True)
# Placeholder for future button
col_btn6.empty()

# MC / Opt settings
with st.sidebar.expander("⚙️ 高級設置 MC/Opt Settings"):
    mc_simulations = st.number_input("MC模擬次數 Simulations", min_value=100, max_value=10000, value=1000, step=100)
    opt_iterations = st.number_input("優化迭代 Opt Iterations", min_value=10, max_value=500, value=50, step=10)
    opt_target = st.selectbox("優化目標 Target", ["sharpe_ratio", "net_profit", "profit_factor", "win_rate", "max_drawdown"])
    opt_min_trades = st.number_input("最少交易數 Min Trades", min_value=1, max_value=100, value=5, step=1, help="過濾交易次數過少的結果")
    random_seed = st.number_input("隨機種子 Seed", min_value=0, max_value=99999, value=42)

# WFA settings
with st.sidebar.expander("🔄 前推設置 WFA Settings"):
    wfa_train_days = st.number_input("訓練天數 Train Days", min_value=30, max_value=730, value=180, step=30,
                                      help="每個窗口用於優化的天數")
    wfa_test_days = st.number_input("測試天數 Test Days", min_value=7, max_value=120, value=30, step=7,
                                     help="每個窗口用於樣本外測試的天數")
    wfa_step_days = st.number_input("步進天數 Step Days", min_value=7, max_value=120, value=30, step=7,
                                     help="窗口滾動步長 (=Test Days時無重疊)")
    wfa_warmup = st.number_input("暖機K線 Warmup Bars", min_value=0, max_value=500, value=200, step=50,
                                  help="測試期前預載的K線數 (用於指標初始化)")
    wfa_opt_iters = st.number_input("WFA優化次數 WFA Opt Iters", min_value=10, max_value=200, value=30, step=10,
                                     help="每個窗口內的優化迭代次數")


# ---------------------------------------------------------------------------
# Execute actions
# ---------------------------------------------------------------------------
def _run_backtest():
    config = _build_config()
    engine = BacktestEngine()
    with st.spinner(f"回測中: {config.strategy_name} on {config.instrument}..."):
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
    with st.spinner(f"蒙特卡洛模擬 {mc_simulations:,} 次..."):
        mc = run_monte_carlo(
            trades_pnl=trades_pnl,
            initial_capital=float(rules_mc.account_size),
            loss_limit=float(rules_mc.max_loss_limit),
            n_simulations=mc_simulations,
            seed=random_seed,
            profit_target=float(rules_mc.profit_target),
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
        st.warning("沒有交易記錄，無法運行蒙特卡洛。")

if run_opt:
    config = _build_config()
    with st.spinner(f"優化中 {opt_iterations} 次 (目標: {opt_target})..."):
        opt = run_optimization(
            base_config=config,
            n_iterations=opt_iterations,
            target_metric=opt_target,
            seed=random_seed,
        )
    st.session_state.opt_result = opt
    n_saved = _save_optimizer_presets(strat_name, opt, min_trades=opt_min_trades)
    # Auto-load best params into sidebar
    _best = next((r for r in opt.all_results if r["metrics"].get("total_trades", 0) >= opt_min_trades), None)
    st.session_state.active_params = _best["params"] if _best else opt.best_params
    st.toast(f"✅ 優化完成！已保存 {n_saved} 個預設，參數已加載到側欄")
    st.rerun()

if run_full:
    config = _build_config()
    progress = st.progress(0, text="步驟 1/3: 優化參數...")
    opt = run_optimization(
        base_config=config,
        n_iterations=opt_iterations,
        target_metric=opt_target,
        seed=random_seed,
    )
    st.session_state.opt_result = opt
    n_saved = _save_optimizer_presets(strat_name, opt, min_trades=opt_min_trades)
    progress.progress(33, text="步驟 2/3: 用最優參數回測...")

    # Find best with enough trades
    best_with_trades = None
    for r in opt.all_results:
        if r["metrics"].get("total_trades", 0) >= opt_min_trades:
            best_with_trades = r
            break
    best_params = best_with_trades["params"] if best_with_trades else opt.best_params

    best_config = _build_config(params_override=best_params)
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
    progress.progress(66, text="步驟 3/3: 蒙特卡洛模擬...")

    if trades_pnl:
        _run_monte_carlo(trades_pnl)
    progress.progress(100, text="✅ 全部完成！")

    # Auto-load best params into sidebar sliders
    st.session_state.active_params = best_params
    st.rerun()

if run_wfa:
    from backtest.walk_forward import run_walk_forward, WalkForwardConfig
    wfa_config = WalkForwardConfig(
        base_config=_build_config(),
        train_days=wfa_train_days,
        test_days=wfa_test_days,
        step_days=wfa_step_days,
        warmup_bars=wfa_warmup,
        opt_iterations=wfa_opt_iters,
        opt_target_metric=opt_target,
        opt_min_trades=opt_min_trades,
        opt_seed=random_seed,
        start_date=start_date or None,
        end_date=end_date or None,
    )
    _wfa_progress_bar = st.progress(0, text="WFA: 準備中...")
    _wfa_status = st.empty()

    def _wfa_progress(current, total, status):
        pct = min(int((current / max(total, 1)) * 100), 99)
        _wfa_progress_bar.progress(pct, text=f"WFA: {status}")

    try:
        wfa_result = run_walk_forward(wfa_config, progress_callback=_wfa_progress)
        st.session_state.wfa_result = wfa_result
        _wfa_progress_bar.progress(100, text="✅ WFA 完成！")
        st.toast(f"✅ WFA完成！{wfa_result.n_windows} 個窗口, 效率={wfa_result.wf_efficiency:.2%}")
    except Exception as e:
        _wfa_progress_bar.empty()
        st.error(f"WFA 失敗: {e}")


# ---------------------------------------------------------------------------
# Main display area
# ---------------------------------------------------------------------------
st.title("ancserFX Dashboard")

tabs = st.tabs(["📈 回測 Backtest", "🎲 蒙特卡洛 MC", "🔍 優化 Optimize", "🔄 前推 WFA", "📋 交易記錄 Trades", "🕯 K線圖 Chart"])

# ============================================================
# TAB 1: BACKTEST
# ============================================================
with tabs[0]:
    rd = st.session_state.bt_result_dict
    if rd is None:
        st.info("點擊左側 **▶ 回測 Backtest** 開始。")
    else:
        m = rd["metrics"]

        # Row 1: Core metrics
        cols = st.columns(6)
        cols[0].metric("淨利潤 Net Profit", f"${m.get('net_profit', 0):,.2f}",
                       delta=f"{m.get('net_profit_pct', 0):.1f}%")
        cols[1].metric("勝率 Win Rate", f"{m.get('win_rate', 0):.1f}%")
        cols[2].metric("交易數 Trades", f"{m.get('total_trades', 0)}")
        cols[3].metric("盈虧比 Profit Factor", f"{m.get('profit_factor', 0):.2f}")
        cols[4].metric("夏普 Sharpe", f"{m.get('sharpe_ratio', 0):.3f}")
        cols[5].metric("最大回撤 Max DD", f"${m.get('max_drawdown', 0):,.2f}")

        # Row 2: Detail metrics
        cols2 = st.columns(6)
        cols2[0].metric("最大單贏 Largest Win", f"${m.get('largest_win', 0):,.2f}")
        cols2[1].metric("最大單虧 Largest Loss", f"${m.get('largest_loss', 0):,.2f}")
        cols2[2].metric("平均交易 Avg Trade", f"${m.get('avg_trade', 0):,.2f}")
        cols2[3].metric("期望值 Expectancy", f"${m.get('expectancy', 0):,.2f}")
        cols2[4].metric("連勝 Consec Wins", f"{m.get('max_consecutive_wins', 0)}")
        cols2[5].metric("連虧 Consec Losses", f"{m.get('max_consecutive_losses', 0)}")

        # Row 3: Pass & Best Day / Consistency
        passed = m.get("passed", False)
        days_to_pass = m.get("days_to_pass")
        profit_target = TOPSTEP_ACCOUNTS.get(rd["config"].get("account_tier", "50K")).profit_target

        best_day = m.get("best_day_pnl", 0)
        best_day_pct = m.get("best_day_pct_of_profit", 0)
        consistency_ok = best_day <= best_day_limit if m.get("net_profit", 0) > 0 else True

        cols3 = st.columns(6)
        cols3[0].metric("🎯 通過目標 Target", f"${profit_target:,.0f}",
                       delta=f"{'✅ 已通過' if passed else '❌ 未達標'}")
        cols3[1].metric("⏱ 通過天數 Days", f"{days_to_pass or '—'}",
                       delta=f"{str(m.get('pass_timestamp') or '')[:10] if passed else ''}")
        cols3[2].metric("最佳單日 Best Day", f"${best_day:,.2f}",
                       delta=f"{'✅' if consistency_ok else '❌'} 限制${best_day_limit:,.0f}")
        cols3[3].metric("最差單日 Worst Day", f"${m.get('worst_day_pnl', 0):,.2f}")
        cols3[4].metric("Best Day占比", f"{best_day_pct:.1f}%",
                       delta="需<50%" if best_day_pct > 50 else "✅ OK")
        cols3[5].metric("交易天數 Trading Days", f"{m.get('trading_days', 0)}")

        # Row 4: Circuit Breaker & Best Day enforcement
        cb_blocks = m.get("circuit_breaker_blocks", 0)
        bd_pauses = m.get("best_day_pauses", [])
        if cb_blocks > 0 or len(bd_pauses) > 0:
            cols4 = st.columns(3)
            cols4[0].metric("🔌 斷路器攔截 CB Blocks", f"{cb_blocks}",
                           delta="⚠ 預算不足" if cb_blocks > 0 else "")
            cols4[1].metric("⏸ Best Day暫停", f"{len(bd_pauses)} 天",
                           delta=", ".join(bd_pauses[:3]) if bd_pauses else "")
            cols4[2].metric("📊 有效交易日", f"{m.get('trading_days', 0) - len(bd_pauses)}")

        # Row 5: Payout Readiness
        consec_150 = m.get("max_consec_150_days", 0)
        std_ready = m.get("payout_standard_ready", False)
        xfa_ready = m.get("payout_xfa_ready", False)
        cols5 = st.columns(4)
        cols5[0].metric("💵 連續$150+天 Consec Days", f"{consec_150}",
                       delta=f"{'✅ ≥5' if std_ready else f'需{5 - consec_150}天'}")
        cols5[1].metric("📋 標準付款 Standard", "✅ 就緒" if std_ready else "❌ 未達標",
                       delta="5天連續$150+" if std_ready else "")
        cols5[2].metric("⚡ XFA快速付款", "✅ 就緒" if xfa_ready else "❌ 未達標",
                       delta="3天+40%一致性" if xfa_ready else "")
        cols5[3].metric("📊 Best Day佔比", f"{best_day_pct:.1f}%",
                       delta="需≤50% (XFA≤60%)" if best_day_pct > 50 else "✅ OK")

        # Equity curve
        eq = rd.get("equity_curve", [])
        if eq:
            eq_df = pd.DataFrame(eq)
            # Convert string timestamps to datetime for plotly compatibility
            if "timestamp" in eq_df.columns:
                eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])
            x_axis = eq_df.get("timestamp", eq_df.index)

            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.7, 0.3], vertical_spacing=0.05,
                subplot_titles=["權益曲線 Equity Curve", "回撤 Drawdown"],
            )
            fig.add_trace(go.Scatter(
                x=x_axis, y=eq_df["equity"],
                mode="lines", name="Equity", line=dict(color="#2196F3", width=1.5),
            ), row=1, col=1)

            equity_arr = eq_df["equity"].values
            peak = np.maximum.accumulate(equity_arr)
            dd = peak - equity_arr
            fig.add_trace(go.Scatter(
                x=x_axis, y=-dd,
                mode="lines", fill="tozeroy", name="Drawdown",
                line=dict(color="#ff1744", width=1), fillcolor="rgba(255,23,68,0.3)",
            ), row=2, col=1)

            # Add loss limit line on drawdown
            fig.add_hline(y=-rules.max_loss_limit, line_dash="dash", line_color="#ff9800",
                         annotation_text=f"回撤限制 -${rules.max_loss_limit:,.0f}", row=2, col=1)

            # Add profit target line on equity
            acct_rules = TOPSTEP_ACCOUNTS.get(rd["config"].get("account_tier", "50K"))
            if acct_rules and acct_rules.profit_target > 0:
                target_equity = acct_rules.account_size + acct_rules.profit_target
                fig.add_hline(y=target_equity, line_dash="dash", line_color="#00c853",
                             annotation_text=f"🎯 通過目標 ${target_equity:,.0f}", row=1, col=1)

            # Mark pass point with a star
            pass_idx = m.get("pass_bar_index")
            if pass_idx is not None and pass_idx < len(eq_df):
                pass_row = eq_df.iloc[pass_idx]
                fig.add_trace(go.Scatter(
                    x=[pass_row["timestamp"] if "timestamp" in eq_df.columns else pass_idx],
                    y=[pass_row["equity"]],
                    mode="markers+text",
                    marker=dict(symbol="star", size=16, color="#FFD700"),
                    text=["✅ PASSED"],
                    textposition="top center",
                    textfont=dict(color="#FFD700", size=12),
                    showlegend=False,
                ), row=1, col=1)

            # 1-month evaluation deadline vertical line
            # (use add_shape + add_annotation to avoid plotly sum(Timestamp) bug)
            if "timestamp" in eq_df.columns and len(eq_df) > 0:
                first_ts = eq_df["timestamp"].iloc[0]
                one_month_str = (first_ts + pd.Timedelta(days=30)).isoformat()
                fig.add_shape(
                    type="line", x0=one_month_str, x1=one_month_str,
                    y0=0, y1=1, yref="y domain",
                    line=dict(dash="dot", color="#9C27B0", width=1),
                    row=1, col=1,
                )
                fig.add_annotation(
                    x=one_month_str, y=1, yref="y domain",
                    text="📅 30天評估期", showarrow=False, yanchor="bottom",
                    font=dict(color="#9C27B0", size=10), row=1, col=1,
                )

            # Mark Best Day pauses
            bd_pauses = m.get("best_day_pauses", [])
            for bd_date in bd_pauses:
                bd_str = pd.Timestamp(bd_date).isoformat()
                fig.add_shape(
                    type="line", x0=bd_str, x1=bd_str,
                    y0=0, y1=1, yref="y domain",
                    line=dict(dash="dot", color="#FF9800", width=1),
                    row=1, col=1,
                )
                fig.add_annotation(
                    x=bd_str, y=0, yref="y domain",
                    text="⏸ Best Day", showarrow=False, yanchor="top",
                    font=dict(color="#FF9800", size=10), row=1, col=1,
                )

            # Show CB blocks count as annotation
            cb_blocks = m.get("circuit_breaker_blocks", 0)
            if cb_blocks > 0:
                fig.add_annotation(
                    text=f"🔌 斷路器攔截 {cb_blocks} 次",
                    xref="paper", yref="paper", x=0.01, y=0.98,
                    showarrow=False, font=dict(color="#ff9800", size=11),
                    bgcolor="rgba(0,0,0,0.6)",
                )

            fig.update_layout(template="plotly_dark", height=500, showlegend=False,
                            margin=dict(l=50, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

        # Trade P&L
        trades = rd.get("trades", [])
        if trades:
            pnl_vals = [t["pnl"] for t in trades]
            colors = ["#00c853" if p >= 0 else "#ff1744" for p in pnl_vals]
            fig_pnl = go.Figure(go.Bar(
                x=list(range(1, len(pnl_vals) + 1)), y=pnl_vals,
                marker_color=colors, name="Trade P&L",
            ))
            fig_pnl.update_layout(template="plotly_dark", height=250,
                                title="單筆盈虧 Trade P&L", xaxis_title="Trade #",
                                yaxis_title="P&L ($)", margin=dict(l=50, r=20, t=40, b=30))
            st.plotly_chart(fig_pnl, use_container_width=True)

        if rd.get("violations"):
            with st.expander(f"⚠️ 風控違規 Risk Violations ({len(rd['violations'])})"):
                for v in rd["violations"][:50]:
                    st.text(v)

        cfg = rd["config"]
        st.caption(
            f"策略: {cfg['strategy_name']} | 合約: {cfg['instrument']} | "
            f"週期: {cfg.get('timeframe', '5min')} | "
            f"K線: {rd.get('total_bars', 'N/A'):,} | "
            f"耗時: {rd.get('elapsed_sec', 0):.1f}s"
        )


# ============================================================
# TAB 2: MONTE CARLO
# ============================================================
with tabs[1]:
    mc = st.session_state.mc_result
    if mc is None:
        st.info("點擊左側 **🎲 蒙特卡洛 MC** 運行模擬。")
    else:
        fe = mc.final_equity_stats
        dd = mc.max_drawdown_stats
        cl = mc.max_consec_loss_stats

        cols = st.columns(6)
        ruin_color = "normal" if mc.ruin_probability < 0.05 else "inverse"
        cols[0].metric("💀 爆倉概率 Ruin", f"{mc.ruin_probability:.1%}", delta_color=ruin_color)
        cols[1].metric("中位終值 Median", f"${fe['p50']:,.0f}")
        cols[2].metric("95%最大回撤 DD", f"${dd['p95']:,.0f}")
        cols[3].metric("95%連虧 ConsecL", f"{cl['p95']:.0f}")
        pass_color = "normal" if mc.pass_probability > 0.5 else "inverse"
        cols[4].metric("✅ 通過概率 Pass", f"{mc.pass_probability:.1%}", delta_color=pass_color)
        cols[5].metric("📅 30天通過 30d", f"{mc.pass_30d_probability:.1%}",
                       delta_color="normal" if mc.pass_30d_probability > 0.3 else "inverse")

        cols2 = st.columns(4)
        cols2[0].metric("5%終值 5th Equity", f"${fe['p5']:,.0f}")
        cols2[1].metric("95%終值 95th Equity", f"${fe['p95']:,.0f}")
        cols2[2].metric("平均最大回撤 Avg DD", f"${dd['mean']:,.0f}")
        cols2[3].metric("平均連虧 Avg Consec", f"{cl['mean']:.1f}")

        # Fan chart
        pc = mc.percentile_curves
        n_points = len(pc[50]) if 50 in pc and pc[50] else 0
        if n_points > 0:
            x_axis = list(range(n_points))
            fig_fan = go.Figure()
            fig_fan.add_trace(go.Scatter(x=x_axis, y=pc[95], mode="lines", line=dict(width=0), showlegend=False))
            fig_fan.add_trace(go.Scatter(x=x_axis, y=pc[5], mode="lines", line=dict(width=0),
                                        fill="tonexty", fillcolor="rgba(33,150,243,0.15)", name="5th-95th"))
            fig_fan.add_trace(go.Scatter(x=x_axis, y=pc[75], mode="lines", line=dict(width=0), showlegend=False))
            fig_fan.add_trace(go.Scatter(x=x_axis, y=pc[25], mode="lines", line=dict(width=0),
                                        fill="tonexty", fillcolor="rgba(33,150,243,0.3)", name="25th-75th"))
            fig_fan.add_trace(go.Scatter(x=x_axis, y=pc[50], mode="lines",
                                        line=dict(color="#2196F3", width=2), name="中位數 Median"))
            fig_fan.add_hline(y=mc.initial_capital - mc.loss_limit, line_dash="dash",
                            line_color="#ff1744", annotation_text=f"爆倉線 (${mc.initial_capital - mc.loss_limit:,.0f})")
            fig_fan.update_layout(template="plotly_dark",
                                title=f"蒙特卡洛扇形圖 ({mc.n_simulations:,} 次模擬)",
                                xaxis_title="交易編號 Trade #", yaxis_title="權益 Equity ($)",
                                height=450, margin=dict(l=60, r=20, t=50, b=30))
            st.plotly_chart(fig_fan, use_container_width=True)

        col_dist1, col_dist2 = st.columns(2)
        with col_dist1:
            fig_fe = go.Figure()
            fig_fe.add_trace(go.Bar(x=["5th", "25th", "50th", "75th", "95th"],
                                   y=[fe["p5"], fe["p25"], fe["p50"], fe["p75"], fe["p95"]],
                                   marker_color=["#ff1744", "#ff9800", "#2196F3", "#4caf50", "#00c853"]))
            fig_fe.update_layout(template="plotly_dark", title="終值分佈 Final Equity",
                               yaxis_title="Equity ($)", height=300, margin=dict(l=60, r=20, t=40, b=30))
            st.plotly_chart(fig_fe, use_container_width=True)

        with col_dist2:
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Bar(x=["5th", "25th", "50th", "75th", "95th"],
                                   y=[dd["p5"], dd["p25"], dd["p50"], dd["p75"], dd["p95"]],
                                   marker_color=["#00c853", "#4caf50", "#ff9800", "#ff5722", "#ff1744"]))
            fig_dd.add_hline(y=mc.loss_limit, line_dash="dash", line_color="#ff1744",
                           annotation_text=f"回撤限額 ${mc.loss_limit:,.0f}")
            fig_dd.update_layout(template="plotly_dark", title="最大回撤分佈 Max DD",
                               yaxis_title="Drawdown ($)", height=300, margin=dict(l=60, r=20, t=40, b=30))
            st.plotly_chart(fig_dd, use_container_width=True)

        st.caption(f"模擬: {mc.n_simulations:,} | 交易: {mc.n_trades} | "
                  f"資金: ${mc.initial_capital:,.0f} | 回撤限額: ${mc.loss_limit:,.0f}")


# ============================================================
# TAB 3: OPTIMIZATION
# ============================================================
with tabs[2]:
    opt = st.session_state.opt_result
    if opt is None:
        st.info("點擊左側 **🔍 優化 Optimize** 搜索最佳參數。")
    else:
        # Filter by min trades
        filtered_results = [
            r for r in opt.all_results
            if r["metrics"].get("total_trades", 0) >= opt_min_trades
        ]

        if not filtered_results:
            st.warning(f"沒有結果滿足最少 {opt_min_trades} 筆交易的條件。")
        else:
            best = filtered_results[0]

            st.subheader("🏆 最優參數 Best Parameters")
            st.caption(f"(已過濾: 僅顯示 ≥{opt_min_trades} 筆交易的結果，共 {len(filtered_results)}/{len(opt.all_results)} 組)")

            cols_best = st.columns(min(len(best["params"]), 6) or 1)
            for i, (k, v) in enumerate(best["params"].items()):
                label = PARAM_LABELS.get(k, k)
                cols_best[i % len(cols_best)].metric(label, f"{v}")

            bm = best["metrics"]
            cols_bm = st.columns(6)
            cols_bm[0].metric("淨利潤", f"${bm.get('net_profit', 0):,.2f}")
            cols_bm[1].metric("勝率", f"{bm.get('win_rate', 0):.1f}%")
            cols_bm[2].metric("盈虧比", f"{bm.get('profit_factor', 0):.2f}")
            cols_bm[3].metric("夏普", f"{bm.get('sharpe_ratio', 0):.3f}")
            cols_bm[4].metric("最大回撤", f"${bm.get('max_drawdown', 0):,.2f}")
            cols_bm[5].metric("交易數", f"{bm.get('total_trades', 0)}")

            st.markdown("---")

            # Top-10 table (filtered)
            st.subheader(f"📊 Top 10 (按 {opt.target_metric}, ≥{opt_min_trades} 筆交易)")
            top_rows = []
            for i, r in enumerate(filtered_results[:10]):
                row = {"Rank": i + 1}
                row.update(r["params"])
                row[opt.target_metric] = r["metrics"].get(opt.target_metric, 0)
                row["net_profit"] = r["metrics"].get("net_profit", 0)
                row["win_rate"] = r["metrics"].get("win_rate", 0)
                row["profit_factor"] = r["metrics"].get("profit_factor", 0)
                row["max_drawdown"] = r["metrics"].get("max_drawdown", 0)
                row["trades"] = r["metrics"].get("total_trades", 0)
                top_rows.append(row)
            if top_rows:
                st.dataframe(pd.DataFrame(top_rows), use_container_width=True, hide_index=True)

            # Parameter sensitivity
            if not opt.param_metric_df.empty and opt.param_names:
                st.subheader("📈 參數敏感度 Parameter Sensitivity")
                n_params = len(opt.param_names)
                n_cols = min(n_params, 3)
                scatter_cols = st.columns(n_cols)
                for i, pname in enumerate(opt.param_names):
                    with scatter_cols[i % n_cols]:
                        df = opt.param_metric_df
                        if pname in df.columns and opt.target_metric in df.columns:
                            # Filter by min trades for scatter too
                            df_filt = df[df["total_trades"] >= opt_min_trades] if "total_trades" in df.columns else df
                            if not df_filt.empty:
                                fig_sc = go.Figure()
                                fig_sc.add_trace(go.Scatter(
                                    x=df_filt[pname], y=df_filt[opt.target_metric],
                                    mode="markers", marker=dict(
                                        size=6, color=df_filt.get("net_profit", df_filt[opt.target_metric]),
                                        colorscale="RdYlGn", showscale=False, opacity=0.7,
                                    ),
                                ))
                                label = PARAM_LABELS.get(pname, pname)
                                fig_sc.update_layout(template="plotly_dark", title=label,
                                                   xaxis_title=pname, yaxis_title=opt.target_metric,
                                                   height=280, margin=dict(l=50, r=10, t=35, b=30))
                                st.plotly_chart(fig_sc, use_container_width=True)

        st.caption(f"迭代: {opt.n_iterations} | 目標: {opt.target_metric} | 耗時: {opt.elapsed_sec:.1f}s")


# ============================================================
# TAB 4: WALK-FORWARD ANALYSIS
# ============================================================
with tabs[3]:
    wfa = st.session_state.wfa_result
    if wfa is None:
        st.info("點擊左側 **🔄 前推 WFA** 運行前推分析。")
    else:
        sm = wfa.stitched_metrics

        # Row 1: Headline metrics
        wfa_cols = st.columns(6)
        eff_color = "normal" if wfa.wf_efficiency >= 0.5 else "inverse"
        wfa_cols[0].metric("📊 WF效率 Efficiency", f"{wfa.wf_efficiency:.2%}",
                           delta="≥50%佳" if wfa.wf_efficiency >= 0.5 else "⚠ <50%",
                           delta_color=eff_color)
        wfa_cols[1].metric("💰 拼接淨利 Stitched P&L", f"${sm.get('net_profit', 0):,.2f}")
        wfa_cols[2].metric("夏普 Sharpe", f"{sm.get('sharpe_ratio', 0):.3f}")
        wfa_cols[3].metric("窗口一致 Consistency", f"{wfa.window_consistency:.0%}",
                           delta=f"{wfa.n_profitable_windows}/{wfa.n_windows} 盈利")
        wfa_cols[4].metric("窗口數 Windows", f"{wfa.n_windows}")
        wfa_cols[5].metric("交易數 Trades", f"{sm.get('total_trades', 0)}")

        wfa_cols2 = st.columns(6)
        wfa_cols2[0].metric("勝率 Win Rate", f"{sm.get('win_rate', 0):.1f}%")
        wfa_cols2[1].metric("盈虧比 PF", f"{sm.get('profit_factor', 0):.2f}")
        wfa_cols2[2].metric("最大回撤 Max DD", f"${sm.get('max_drawdown', 0):,.2f}")
        wfa_cols2[3].metric("平均交易 Avg Trade", f"${sm.get('avg_trade', 0):,.2f}")
        wfa_cols2[4].metric("期望值 Expectancy", f"${sm.get('expectancy', 0):,.2f}")
        wfa_cols2[5].metric("⏱ 耗時 Elapsed", f"{wfa.total_elapsed_sec:.0f}s")

        # Stitched equity curve with window boundaries
        if wfa.stitched_equity_curve:
            eq_df = pd.DataFrame(wfa.stitched_equity_curve)
            eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])

            fig_wfa_eq = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.7, 0.3], vertical_spacing=0.05,
                subplot_titles=["拼接權益曲線 Stitched Equity (OOS)", "回撤 Drawdown"],
            )
            fig_wfa_eq.add_trace(go.Scatter(
                x=eq_df["timestamp"], y=eq_df["equity"],
                mode="lines", name="Equity", line=dict(color="#2196F3", width=1.5),
            ), row=1, col=1)

            equity_arr = eq_df["equity"].values
            peak_arr = np.maximum.accumulate(equity_arr)
            dd_arr = peak_arr - equity_arr
            fig_wfa_eq.add_trace(go.Scatter(
                x=eq_df["timestamp"], y=-dd_arr,
                mode="lines", fill="tozeroy", name="Drawdown",
                line=dict(color="#ff1744", width=1), fillcolor="rgba(255,23,68,0.3)",
            ), row=2, col=1)

            # Window boundary markers
            _wfa_colors = ["rgba(255,255,255,0.05)", "rgba(100,100,255,0.05)"]
            for w in wfa.windows:
                ts_start = pd.Timestamp(w.test_start)
                ts_end = pd.Timestamp(w.test_end)
                fig_wfa_eq.add_vrect(
                    x0=ts_start, x1=ts_end,
                    fillcolor=_wfa_colors[w.window_index % 2],
                    line_width=0, row=1, col=1,
                )
                fig_wfa_eq.add_annotation(
                    x=ts_start + (ts_end - ts_start) / 2, y=1, yref="y domain",
                    text=f"W{w.window_index + 1}", showarrow=False,
                    font=dict(size=9, color="#888"), yanchor="bottom",
                    row=1, col=1,
                )

            # Loss limit line
            acct_wfa = TOPSTEP_ACCOUNTS.get(wfa.config.base_config.account_tier)
            if acct_wfa:
                fig_wfa_eq.add_hline(y=-acct_wfa.max_loss_limit, line_dash="dash",
                                     line_color="#ff9800",
                                     annotation_text=f"回撤限制 -${acct_wfa.max_loss_limit:,.0f}",
                                     row=2, col=1)

            fig_wfa_eq.update_layout(template="plotly_dark", height=500, showlegend=False,
                                     margin=dict(l=50, r=20, t=40, b=20))
            st.plotly_chart(fig_wfa_eq, use_container_width=True)

        # Per-window summary table
        st.subheader("📋 窗口明細 Window Details")
        wfa_rows = []
        for w in wfa.windows:
            test_pnl = w.test_metrics.get("net_profit", 0)
            wfa_rows.append({
                "Window": f"W{w.window_index + 1}",
                "Train": f"{w.train_start} ~ {w.train_end}",
                "Test": f"{w.test_start} ~ {w.test_end}",
                "Train Target": round(w.train_target_value, 4),
                "Test Target": round(w.test_metrics.get(wfa.config.opt_target_metric, 0) or 0, 4),
                "Efficiency": f"{w.efficiency_ratio:.2%}",
                "Test P&L": f"${test_pnl:,.0f}",
                "Win Rate": f"{w.test_metrics.get('win_rate', 0):.0f}%",
                "Trades": w.test_metrics.get("total_trades", 0),
            })
        if wfa_rows:
            st.dataframe(pd.DataFrame(wfa_rows), use_container_width=True, hide_index=True)

        # Train vs Test bar chart
        st.subheader("📊 訓練 vs 測試 Train vs Test")
        _wfa_labels = [f"W{w.window_index + 1}" for w in wfa.windows]
        _wfa_train_vals = [w.train_target_value for w in wfa.windows]
        _wfa_test_vals = [w.test_metrics.get(wfa.config.opt_target_metric, 0) or 0 for w in wfa.windows]

        fig_tvt = go.Figure()
        fig_tvt.add_trace(go.Bar(x=_wfa_labels, y=_wfa_train_vals, name="Train",
                                  marker_color="#2196F3", opacity=0.8))
        fig_tvt.add_trace(go.Bar(x=_wfa_labels, y=_wfa_test_vals, name="Test (OOS)",
                                  marker_color="#FF9800", opacity=0.8))
        fig_tvt.update_layout(template="plotly_dark", barmode="group",
                              title=f"Train vs Test: {wfa.config.opt_target_metric}",
                              height=300, margin=dict(l=50, r=20, t=40, b=30))
        st.plotly_chart(fig_tvt, use_container_width=True)

        # Parameter stability
        if wfa.param_stability:
            st.subheader("🔧 參數穩定性 Parameter Stability")
            n_p = len(wfa.param_stability)
            ps_cols = st.columns(min(n_p, 3))
            for idx, (pname, pstat) in enumerate(wfa.param_stability.items()):
                with ps_cols[idx % min(n_p, 3)]:
                    vals = pstat.get("values", [])
                    if vals and isinstance(vals[0], (int, float)):
                        x_labels = [f"W{j+1}" for j in range(len(vals))]
                        fig_ps = go.Figure()
                        fig_ps.add_trace(go.Scatter(
                            x=x_labels, y=vals, mode="lines+markers",
                            line=dict(color="#4caf50", width=2),
                            marker=dict(size=6),
                        ))
                        fig_ps.add_hline(y=pstat["mean"], line_dash="dash",
                                         line_color="#888",
                                         annotation_text=f"μ={pstat['mean']:.2f}")
                        label = PARAM_LABELS.get(pname, pname)
                        fig_ps.update_layout(
                            template="plotly_dark",
                            title=f"{label}<br><sub>CV={pstat['cv']:.3f}</sub>",
                            height=250,
                            margin=dict(l=50, r=10, t=50, b=30),
                        )
                        st.plotly_chart(fig_ps, use_container_width=True)

        st.caption(
            f"訓練: {wfa.config.train_days}天 | 測試: {wfa.config.test_days}天 | "
            f"步進: {wfa.config.step_days}天 | 優化: {wfa.config.opt_iterations}次/窗 | "
            f"總耗時: {wfa.total_elapsed_sec:.0f}s"
        )


# ============================================================
# TAB 5: TRADE LOG
# ============================================================
with tabs[4]:
    rd = st.session_state.bt_result_dict
    if rd is None:
        st.info("先運行回測查看交易記錄。")
    else:
        trades = rd.get("trades", [])
        if not trades:
            st.warning("本次回測沒有交易。")
        else:
            trade_df = pd.DataFrame(trades)
            display_cols = ["entry_time", "exit_time", "direction", "quantity",
                           "entry_price", "exit_price", "pnl", "commission", "bars_held"]
            available_cols = [c for c in display_cols if c in trade_df.columns]
            st.dataframe(
                trade_df[available_cols], use_container_width=True, hide_index=True,
                column_config={
                    "pnl": st.column_config.NumberColumn("盈虧 P&L", format="$%.2f"),
                    "commission": st.column_config.NumberColumn("手續費", format="$%.2f"),
                    "entry_price": st.column_config.NumberColumn("入場價 Entry", format="%.2f"),
                    "exit_price": st.column_config.NumberColumn("出場價 Exit", format="%.2f"),
                },
            )
            total_pnl = sum(t["pnl"] for t in trades)
            wins = sum(1 for t in trades if t["pnl"] > 0)
            st.caption(f"總計: {len(trades)} 筆 | 贏: {wins} | 虧: {len(trades) - wins} | 淨盈虧: ${total_pnl:,.2f}")


# ============================================================
# TAB 6: CANDLESTICK CHART (K-Line with trade entry/exit)
# ============================================================
with tabs[5]:
    rd = st.session_state.bt_result_dict
    if rd is None:
        st.info("先運行回測以查看K線圖。")
    else:
        cfg = rd["config"]
        trades = rd.get("trades", [])

        # Load OHLCV data for the same instrument/timeframe/dates
        @st.cache_data(show_spinner=False)
        def _load_chart_bars(inst: str, tf: str, start: str, end: str):
            loader = DataLoader()
            return loader.get_bars(inst, tf, start_date=start, end_date=end)

        try:
            bar_df = _load_chart_bars(
                cfg["instrument"], cfg.get("timeframe", "5min"),
                cfg.get("start_date", ""), cfg.get("end_date", ""),
            )
        except Exception as e:
            st.error(f"無法載入K線數據: {e}")
            bar_df = pd.DataFrame()

        if bar_df.empty:
            st.warning("無K線數據可顯示。")
        else:
            bar_df["timestamp"] = pd.to_datetime(bar_df["timestamp"])

            # Trade navigation
            if trades:
                trade_labels = [
                    f"#{i+1} {t['direction']} {t['entry_time'][:16]} → {t['exit_time'][:16]} "
                    f"({'✅' if t['pnl'] > 0 else '❌'} ${t['pnl']:,.2f})"
                    for i, t in enumerate(trades)
                ]
                col_nav1, col_nav2 = st.columns([3, 1])
                with col_nav1:
                    selected_trade_idx = st.selectbox(
                        "🔍 跳轉交易 Jump to Trade", range(len(trade_labels)),
                        format_func=lambda i: trade_labels[i],
                    )
                with col_nav2:
                    chart_padding = st.number_input("前後K線 Padding Bars", min_value=5, max_value=200, value=30, step=5)
            else:
                selected_trade_idx = None
                chart_padding = 50

            # Determine visible range
            if selected_trade_idx is not None and trades:
                sel_trade = trades[selected_trade_idx]
                entry_ts = pd.Timestamp(sel_trade["entry_time"])
                exit_ts = pd.Timestamp(sel_trade["exit_time"])
                # Find bar indices
                entry_idx = bar_df["timestamp"].searchsorted(entry_ts)
                exit_idx = bar_df["timestamp"].searchsorted(exit_ts)
                view_start = max(0, entry_idx - chart_padding)
                view_end = min(len(bar_df), exit_idx + chart_padding)
            else:
                view_start = 0
                view_end = min(len(bar_df), 500)

            view_df = bar_df.iloc[view_start:view_end].copy()

            if view_df.empty:
                st.warning("選定範圍內無K線數據。")
            else:
                # Build candlestick chart with volume
                fig = make_subplots(
                    rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.8, 0.2], vertical_spacing=0.02,
                    subplot_titles=["", ""],
                )

                # Candlestick
                fig.add_trace(go.Candlestick(
                    x=view_df["timestamp"],
                    open=view_df["open"],
                    high=view_df["high"],
                    low=view_df["low"],
                    close=view_df["close"],
                    increasing=dict(line=dict(color="#26A69A"), fillcolor="#26A69A"),
                    decreasing=dict(line=dict(color="#EF5350"), fillcolor="#EF5350"),
                    name="OHLC",
                    showlegend=False,
                ), row=1, col=1)

                # Volume
                vol_colors = [
                    "#26A69A" if c >= o else "#EF5350"
                    for o, c in zip(view_df["open"], view_df["close"])
                ]
                fig.add_trace(go.Bar(
                    x=view_df["timestamp"],
                    y=view_df["volume"],
                    marker_color=vol_colors,
                    opacity=0.5,
                    name="Volume",
                    showlegend=False,
                ), row=2, col=1)

                # Overlay trades on the chart
                view_min_ts = view_df["timestamp"].iloc[0]
                view_max_ts = view_df["timestamp"].iloc[-1]

                for i, t in enumerate(trades):
                    entry_ts = pd.Timestamp(t["entry_time"])
                    exit_ts = pd.Timestamp(t["exit_time"])

                    # Only show trades that overlap with visible range
                    if exit_ts < view_min_ts or entry_ts > view_max_ts:
                        continue

                    is_win = t["pnl"] > 0
                    is_long = t["direction"] == "LONG"
                    trade_color = "#00C853" if is_win else "#FF1744"

                    # Entry marker
                    entry_symbol = "triangle-up" if is_long else "triangle-down"
                    fig.add_trace(go.Scatter(
                        x=[entry_ts], y=[t["entry_price"]],
                        mode="markers",
                        marker=dict(
                            symbol=entry_symbol, size=12,
                            color=trade_color, line=dict(color="white", width=1),
                        ),
                        name=f"#{i+1} Entry",
                        hovertemplate=(
                            f"<b>#{i+1} {'LONG' if is_long else 'SHORT'} Entry</b><br>"
                            f"價格: {t['entry_price']:.2f}<br>"
                            f"數量: {t['quantity']}<br>"
                            f"時間: {str(entry_ts)[:19]}<extra></extra>"
                        ),
                        showlegend=False,
                    ), row=1, col=1)

                    # Exit marker
                    fig.add_trace(go.Scatter(
                        x=[exit_ts], y=[t["exit_price"]],
                        mode="markers",
                        marker=dict(
                            symbol="x", size=10,
                            color=trade_color, line=dict(color="white", width=1),
                        ),
                        name=f"#{i+1} Exit",
                        hovertemplate=(
                            f"<b>#{i+1} Exit</b><br>"
                            f"價格: {t['exit_price']:.2f}<br>"
                            f"盈虧: ${t['pnl']:,.2f}<br>"
                            f"持倉: {t['bars_held']} bars<extra></extra>"
                        ),
                        showlegend=False,
                    ), row=1, col=1)

                    # Connection line (entry → exit)
                    fig.add_trace(go.Scatter(
                        x=[entry_ts, exit_ts],
                        y=[t["entry_price"], t["exit_price"]],
                        mode="lines",
                        line=dict(color=trade_color, width=1.5, dash="dot"),
                        showlegend=False,
                        hoverinfo="skip",
                    ), row=1, col=1)

                    # Stop loss line (if entry has SL info — approximate from trade data)
                    # Highlight selected trade
                    if selected_trade_idx is not None and i == selected_trade_idx:
                        # Add shaded background for selected trade
                        fig.add_vrect(
                            x0=entry_ts, x1=exit_ts,
                            fillcolor="rgba(255,215,0,0.1)", line_width=0,
                            row=1, col=1,
                        )
                        # Add P&L annotation
                        mid_ts = entry_ts + (exit_ts - entry_ts) / 2
                        mid_price = (t["entry_price"] + t["exit_price"]) / 2
                        fig.add_annotation(
                            x=mid_ts, y=mid_price,
                            text=f"{'✅' if is_win else '❌'} ${t['pnl']:,.2f}",
                            showarrow=True, arrowhead=2,
                            font=dict(color=trade_color, size=12),
                            bgcolor="rgba(0,0,0,0.7)",
                            bordercolor=trade_color,
                            row=1, col=1,
                        )

                # Layout
                fig.update_layout(
                    template="plotly_dark",
                    height=700,
                    margin=dict(l=60, r=20, t=40, b=30),
                    xaxis_rangeslider_visible=False,
                    xaxis2_rangeslider_visible=False,
                    title=f"🕯 {cfg['instrument']} {cfg.get('timeframe', '5min')} K線圖",
                    yaxis_title="價格 Price",
                    yaxis2_title="成交量 Vol",
                    hovermode="x unified",
                    dragmode="zoom",
                )

                # Remove gaps for non-trading hours
                fig.update_xaxes(
                    rangebreaks=[
                        dict(bounds=["sat", "mon"]),  # Skip weekends
                    ]
                )

                st.plotly_chart(fig, use_container_width=True)

                # Trade summary for selected trade
                if selected_trade_idx is not None and trades:
                    sel = trades[selected_trade_idx]
                    cols_detail = st.columns(8)
                    cols_detail[0].metric("方向", sel["direction"])
                    cols_detail[1].metric("入場", f"{sel['entry_price']:.2f}")
                    cols_detail[2].metric("出場", f"{sel['exit_price']:.2f}")
                    cols_detail[3].metric("盈虧", f"${sel['pnl']:,.2f}")
                    cols_detail[4].metric("數量", f"{sel['quantity']}")
                    cols_detail[5].metric("持倉", f"{sel['bars_held']} bars")
                    cols_detail[6].metric("手續費", f"${sel.get('commission', 0):,.2f}")
                    cols_detail[7].metric("入場時間", str(sel["entry_time"])[:16])
