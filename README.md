# ancserFX

TopStep 期貨量化回測系統 — Quantitative futures backtesting system for TopStep funded account evaluation.

Supports ES / NQ / MES / MNQ futures with strategy backtesting, parameter optimization, Monte Carlo simulation, and Walk-Forward Analysis (WFA).

---

## Requirements

- Python >= 3.11

---

## Installation

### Windows

```bat
git clone https://github.com/ancser/ancserFX.git
cd ancserFX
pip install -r requirements.txt
```

啟動 Dashboard：

```bat
start_dashboard.bat
```

或手動：

```bat
streamlit run dashboard.py
```

### macOS / Linux

```bash
git clone https://github.com/ancser/ancserFX.git
cd ancserFX
pip install -r requirements.txt
```

啟動 Dashboard：

```bash
streamlit run dashboard.py
```

### Environment Config

複製環境變量範例並填入你的 API keys：

```bash
cp .env.example .env
```

`.env` 内容：

```
PROJECTX_USERNAME=your_username
PROJECTX_API_KEY=your_api_key
KAGGLE_USERNAME=your_kaggle_username
KAGGLE_KEY=your_kaggle_api_key
```

> Kaggle key 只有下載數據時才需要。TopStepX key 只有實盤連接時才需要。
> 回測和 Dashboard 不需要任何 API key。

---

## Data Setup

### Directory Structure

```
data/
  raw/         ← downloaded CSV files (from Kaggle or other sources)
  parquet/     ← converted bar data (auto-managed by DataStore)
```

### Download Data from Kaggle

```bash
# List available datasets
python -m scripts.download_kaggle --list

# Download ES / NQ data
python -m scripts.download_kaggle --dataset es
python -m scripts.download_kaggle --dataset nq-cme

# Download all futures data
python -m scripts.download_kaggle --dataset all-futures
```

### Convert CSV to Parquet

```bash
python -m scripts.convert_to_parquet --instrument es
python -m scripts.convert_to_parquet --instrument nq --timeframe 1min
```

---

## Usage

### Dashboard (Recommended)

```bash
streamlit run dashboard.py
```

| Tab | Description |
|-----|-------------|
| **Backtest** | 單次回測：權益曲線、績效指標、回撤圖 |
| **Monte Carlo** | 打亂交易順序 1000 次，測量爆倉概率和通關概率 |
| **Optimize** | 隨機參數搜索，找最優參數組合 |
| **WFA** | 前推分析：滾動訓練/測試窗口，防止過擬合 |
| **Trades** | 完整交易記錄 |
| **K-line Chart** | K 線圖 + 交易標記 |

Sidebar controls:
- 合約 (ES / NQ / MES / MNQ)、時間周期、日期範圍
- 策略選擇和參數調整
- TopStep 帳戶等級 (50K / 100K / 150K)
- 數量、滑點、手續費
- **Data Calendar** — 綠色日曆顯示各合約的數據覆蓋範圍
- WFA 設置 (訓練天數、測試天數、步進天數)

### CLI Backtest

```bash
# Single backtest
python run_backtest.py --strategy "Delta Momentum" --instrument ES --timeframe 5min

# With custom params
python run_backtest.py --strategy "KDJ RSI Bot" --instrument MES --account 50K \
    --param sl_points=15 --param tp1_ticks=200
```

### CLI Research (Optimize + Monte Carlo)

```bash
# Optimize only
python run_research.py optimize \
    --strategy "KDJ RSI Bot" --instrument MES --account 50K \
    --iterations 100 --target sharpe_ratio

# Monte Carlo only
python run_research.py monte-carlo \
    --strategy "KDJ RSI Bot" --instrument MES --account 50K \
    --simulations 1000

# Full pipeline: optimize → backtest best → Monte Carlo
python run_research.py full \
    --strategy "KDJ RSI Bot" --instrument MES --account 50K \
    --iterations 100 --simulations 1000
```

---

## TopStep Account Rules

| Account | Balance | Max Trailing DD | Max Contracts | Profit Target |
|---------|---------|----------------|---------------|---------------|
| 50K     | $50,000 | $2,000         | 5             | $3,000        |
| 100K    | $100,000| $3,000         | 10            | $6,000        |
| 150K    | $150,000| $4,500         | 15            | $9,000        |

---

## Strategies

| Strategy | Type | Description |
|----------|------|-------------|
| `Delta Momentum` | Orderflow | 累積 Delta EMA + Volume Imbalance 偵測機構資金流 |
| `SMA Crossover` | Basic | 均線交叉 |
| `RSI Mean Reversion` | Basic | RSI 超買超賣均值回歸 |
| `KDJ RSI Bot` | Basic | KDJ + RSI 組合信號 |

Strategies are auto-registered via `strategies/registry.py`.

### Presets

`presets/` 存放示例參數。運行 Optimizer 後 Dashboard 會自動保存最優結果到此目錄。

---

## Walk-Forward Analysis (WFA)

防止過擬合的滾動前推驗證：

```
Window 1:  [===== Train =====][== Test ==]
Window 2:       [===== Train =====][== Test ==]
Window 3:            [===== Train =====][== Test ==]
```

1. **Train** — 在訓練期優化參數
2. **Test** — 用最優參數在未見過的測試期驗證
3. **Stitch** — 串聯所有測試期權益曲線

Key metrics:
- **WF Efficiency** = test / train metric (越接近 1.0 = 越不過擬合)
- **Parameter Stability** = 各窗口參數的變異係數
- **Window Consistency** = 測試期盈利的窗口占比

---

## Project Structure

```
ancserFX/
├── dashboard.py              # Streamlit Dashboard
├── run_backtest.py           # CLI 回測入口
├── run_research.py           # CLI 優化 + Monte Carlo
├── start_dashboard.bat       # Windows 一鍵啟動
├── requirements.txt
├── .env.example              # 環境變量範例
│
├── backtest/
│   ├── engine.py             # 回測引擎
│   ├── optimizer.py          # 隨機參數優化
│   ├── walk_forward.py       # 前推分析 WFA
│   ├── monte_carlo.py        # Monte Carlo 模擬
│   ├── metrics.py            # 績效指標計算
│   ├── topstep_rules.py      # TopStep 帳戶規則
│   ├── risk.py               # 風險管理
│   ├── broker.py             # 訂單執行模擬
│   ├── events.py             # 事件系統
│   ├── portfolio.py          # 組合追蹤
│   └── visualize.py          # 圖表生成
│
├── strategies/
│   ├── base.py               # 策略基類
│   ├── registry.py           # 策略自動註冊
│   ├── indicators.py         # 技術指標 (EMA, Delta, Imbalance)
│   ├── basic/                # 基礎策略 (SMA, RSI, KDJ)
│   └── orderflow/            # Order Flow 策略
│
├── data/
│   ├── store.py              # Parquet 數據存儲
│   ├── loader.py             # 數據加載
│   └── models.py             # 合約規格、時間周期
│
├── scripts/
│   ├── download_kaggle.py    # Kaggle 數據下載
│   └── convert_to_parquet.py # CSV → Parquet 轉換
│
└── presets/                   # 策略參數預設
```

---

## Supported Instruments

| Symbol | Name | Tick Size | Tick Value |
|--------|------|-----------|------------|
| ES     | E-mini S&P 500 | 0.25 | $12.50 |
| NQ     | E-mini Nasdaq 100 | 0.25 | $5.00 |
| MES    | Micro E-mini S&P | 0.25 | $1.25 |
| MNQ    | Micro E-mini Nasdaq | 0.25 | $0.50 |

**Timeframes:** `tick` · `1min` · `5min` · `15min` · `1h` · `daily`

---

## License

MIT
