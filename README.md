# ancserFX

TopStep 期貨量化回測系統 — Quantitative futures backtesting system for TopStep funded account evaluation.

Supports ES / NQ / MES / MNQ futures with strategy backtesting, parameter optimization, Monte Carlo simulation, and Walk-Forward Analysis (WFA).

---

## Requirements

- Python >= 3.11
- Windows / macOS / Linux

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/ancser/ancserFX.git
cd ancserFX

# 2. Create virtual environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install dashboard dependencies (if not already included)
pip install streamlit plotly kagglehub
```

## Data Setup

### Directory Structure

```
data/
  raw/         ← downloaded CSV files (from Kaggle or other sources)
  parquet/     ← converted bar data (auto-managed by DataStore)
```

### Download Data from Kaggle

Requires Kaggle API credentials. Place `kaggle.json` in `~/.kaggle/` or set `KAGGLE_USERNAME` / `KAGGLE_KEY` in `.env`.

```bash
# List available datasets
python -m scripts.download_kaggle --list

# Download ES OHLCV data
python -m scripts.download_kaggle --dataset es

# Download NQ CME data
python -m scripts.download_kaggle --dataset nq-cme

# Download all futures data
python -m scripts.download_kaggle --dataset all-futures

# Download LOB (Limit Order Book) research data
python -m scripts.download_kaggle --dataset lob
```

### Convert CSV to Parquet

```bash
# Auto-detect CSVs in data/raw/es/ and convert
python -m scripts.convert_to_parquet --instrument es

# Specify custom paths and timeframe
python -m scripts.convert_to_parquet --instrument nq --timeframe 1min
```

---

## Usage

### Dashboard (Recommended)

```bash
streamlit run dashboard.py
```

Dashboard features:

| Tab | Description |
|-----|-------------|
| **Backtest** | Run single-period backtest with equity curve, metrics, drawdown chart |
| **Monte Carlo** | Shuffle trades 1000x, measure ruin probability and pass probability |
| **Optimize** | Random parameter search with parallel execution and heatmaps |
| **WFA** | Walk-Forward Analysis with rolling train/test windows |
| **Trades** | Full trade log with entry/exit details |
| **K-line Chart** | Candlestick chart with trade markers |

Sidebar controls:
- Instrument (ES / NQ / MES / MNQ), Timeframe, Date range
- Strategy selection and parameter tuning
- TopStep account tier (50K / 100K / 150K)
- Quantity, slippage, commission settings
- **Data Calendar** — GitHub-style heatmap showing data availability by instrument and month
- WFA settings (train days, test days, step days, warmup bars)

### TopStep Account Rules

| Account | Balance | Max Drawdown | Max Contracts | Profit Target |
|---------|---------|-------------|---------------|---------------|
| 50K     | $50,000 | $2,000      | 5             | $3,000        |
| 100K    | $100,000| $3,000      | 10            | $6,000        |
| 150K    | $150,000| $4,500      | 15            | $9,000        |

---

## Strategies

### Built-in Strategies

| Strategy | Type | Description |
|----------|------|-------------|
| `delta_momentum` | Orderflow | Cumulative delta EMA + volume imbalance to detect institutional flow |
| `sma_crossover` | Basic | Simple / Exponential moving average crossover |
| `rsi_mean_reversion` | Basic | RSI overbought/oversold mean reversion |
| `kdj_rsi_bot` | Basic | KDJ + RSI combination signal |

Strategies are auto-registered via `strategies/registry.py`. Add new strategies under `strategies/basic/` or `strategies/orderflow/`.

### Presets

Pre-optimized parameter sets are stored in `presets/`:
```
presets/
  delta_momentum_presets.json
  rsi_mean_reversion_presets.json
  kdj_rsi_bot_presets.json
```

---

## Walk-Forward Analysis (WFA)

WFA prevents overfitting by using rolling train/test windows:

```
Window 1:  [===== Train =====][== Test ==]
Window 2:       [===== Train =====][== Test ==]
Window 3:            [===== Train =====][== Test ==]
...
```

Each window:
1. **Train** — Run parameter optimization on training period
2. **Test** — Apply best parameters to unseen test period
3. **Stitch** — Chain all test-period equity curves for honest evaluation

Key metrics:
- **WF Efficiency** = test_metric / train_metric (closer to 1.0 = less overfit)
- **Parameter Stability** = coefficient of variation per parameter across windows
- **Window Consistency** = fraction of windows with positive test profit

### Run WFA from Dashboard

1. Open sidebar → WFA settings expander
2. Set train/test/step days (e.g., 60/14/14)
3. Click "前推 WFA" button
4. View results in the WFA tab

---

## Project Structure

```
ancserFX/
├── dashboard.py              # Streamlit interactive dashboard
├── requirements.txt
├── pyproject.toml
│
├── backtest/
│   ├── engine.py             # Core backtest engine
│   ├── optimizer.py          # Random parameter optimization
│   ├── walk_forward.py       # Walk-Forward Analysis
│   ├── monte_carlo.py        # Monte Carlo simulation
│   ├── metrics.py            # Performance metrics calculation
│   ├── topstep_rules.py      # TopStep account rules (50K/100K/150K)
│   ├── risk.py               # Risk management
│   ├── broker.py             # Order execution simulation
│   ├── events.py             # Event system
│   ├── portfolio.py          # Portfolio tracking
│   └── visualize.py          # Chart generation
│
├── strategies/
│   ├── base.py               # Base strategy class
│   ├── registry.py           # Strategy auto-registration
│   ├── indicators.py         # Technical indicators (EMA, delta, imbalance)
│   ├── basic/                # Basic strategies (SMA, RSI, KDJ)
│   └── orderflow/            # Order flow strategies (Delta Momentum)
│
├── data/
│   ├── store.py              # Parquet data store
│   ├── loader.py             # Data loading facade
│   └── models.py             # Instrument specs, timeframes, data models
│
├── scripts/
│   ├── download_kaggle.py    # Kaggle dataset downloader
│   └── convert_to_parquet.py # CSV → Parquet converter
│
├── presets/                   # Pre-optimized strategy parameters
├── api/                       # FastAPI endpoints (WIP)
└── execution/                 # Live execution module (WIP)
```

---

## Supported Instruments

| Symbol | Name | Tick Size | Tick Value |
|--------|------|-----------|------------|
| ES     | E-mini S&P 500 | 0.25 | $12.50 |
| NQ     | E-mini Nasdaq 100 | 0.25 | $5.00 |
| MES    | Micro E-mini S&P | 0.25 | $1.25 |
| MNQ    | Micro E-mini Nasdaq | 0.25 | $0.50 |

## Supported Timeframes

`tick` · `1min` · `5min` · `15min` · `1h` · `daily`

---

## License

MIT
