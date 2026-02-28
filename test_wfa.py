"""Quick WFA smoke test - small windows, few iterations."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from backtest.engine import BacktestConfig
from backtest.walk_forward import run_walk_forward, WalkForwardConfig

base = BacktestConfig(
    strategy_name="Delta Momentum",
    strategy_params={},  # Will be optimized
    instrument="MNQ",
    timeframe="5min",
    account_tier="150K",
    quantity=3,
    slippage_ticks=1,
    commission=2.50,
)

wfa_cfg = WalkForwardConfig(
    base_config=base,
    train_days=60,
    test_days=30,
    step_days=30,
    warmup_bars=100,
    opt_iterations=10,  # Very few for speed
    opt_target_metric="sharpe_ratio",
    opt_min_trades=3,
    opt_seed=42,
    start_date="2023-01-01",
    end_date="2024-06-30",
)

def progress(current, total, status):
    print(f"  [{current}/{total}] {status}")

print("Starting WFA smoke test...")
result = run_walk_forward(wfa_cfg, progress_callback=progress)

print(f"\n{'='*70}")
print(f"WFA RESULT SUMMARY")
print(f"{'='*70}")
print(f"Windows:        {result.n_windows}")
print(f"Profitable:     {result.n_profitable_windows}/{result.n_windows}")
print(f"Consistency:    {result.window_consistency:.0%}")
print(f"WF Efficiency:  {result.wf_efficiency:.4f}")
print(f"Stitched Profit: ${result.stitched_metrics.get('net_profit', 0):,.2f}")
print(f"Stitched Sharpe: {result.stitched_metrics.get('sharpe_ratio', 0):.4f}")
print(f"Total Trades:   {result.stitched_metrics.get('total_trades', 0)}")
print(f"Elapsed:        {result.total_elapsed_sec:.1f}s")

print(f"\nPer-window:")
for w in result.windows:
    tp = w.test_metrics.get('net_profit', 0)
    tt = w.test_metrics.get('total_trades', 0)
    print(f"  W{w.window_index+1}: train_sharpe={w.train_target_value:.4f}, "
          f"test_P&L=${tp:,.0f}, trades={tt}, eff={w.efficiency_ratio:.2%}")

print(f"\nParam stability:")
for pname, pstat in result.param_stability.items():
    print(f"  {pname}: mean={pstat['mean']:.2f}, std={pstat['std']:.2f}, CV={pstat['cv']:.3f}")
