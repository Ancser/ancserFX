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
download_data.bat
start_dashboard.bat
```

### macOS / Linux

```bash
git clone https://github.com/ancser/ancserFX.git
cd ancserFX
pip install -r requirements.txt
bash download_data.sh
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

Repo 內含 `test-data.zip` (ES / NQ / MES / MNQ parquet)，首次使用解壓即可，不需要任何 API key。

**Windows:**

```bat
download_data.bat
```

**macOS / Linux:**

```bash
bash download_data.sh
```

### Data Coverage

解壓後所有數據集的覆蓋範圍和類型：

| 合約 | 週期 | 類型 | 範圍 | Bars | 備註 |
|------|------|------|------|------|------|
| ES | 5min | OHLCV | 2019-08 ~ 2024-08 | 353,206 | 主要數據集 |
| ES | daily | OHLCV | 1999-09 ~ 2022-12 | 5,928 | 16% rows O=H=L=C，質量一般 |
| ES | tick | OHLCV | 2020-09 ~ 2021-12 | 317 | 實為 daily，標記錯誤，不可用 |
| MES | 5min | OHLCV | 2019-08 ~ 2024-08 | 353,206 | 與 ES 相同數據 (Kaggle 無獨立 MES) |
| NQ | 5min | OHLCV | 2019-08 ~ 2024-08 | 329,458 | 主要數據集 |
| MNQ | 5min | OHLCV | 2019-08 ~ 2024-08 | 329,458 | 與 NQ 相同數據 (Kaggle 無獨立 MNQ) |

> 目前無 Orderflow (delta / bid-ask volume) 數據。
> Delta Momentum 策略使用 OHLCV 近似法計算 delta，不需要真正的訂單流數據。

### 選擇合約/週期/策略時的實際行為

| 選擇 | 結果 | 說明 |
|------|------|------|
| ES + 5min | 正常 | 353K bars, 2019-2024 |
| ES + daily | 可用但質量一般 | 1999-2022，部分 bar 數據異常 |
| ES + tick | 不可用 | 假數據，僅 317 個 daily bar 被錯標為 tick |
| ES + 1min / 15min / 1h | 無數據 | 只有 5min / daily / tick 可用 |
| MES + 5min | 正常 | 價格與 ES 相同，PnL 按 MES tick value ($1.25) 計算 |
| NQ + 5min | 正常 | 329K bars, 2019-2024 |
| MNQ + 5min | 正常 | 價格與 NQ 相同，PnL 按 MNQ tick value ($0.50) 計算 |
| Delta Momentum + OHLCV | 正常 | 用 (close-open)/volume 近似 delta，不需要訂單流數據 |
| 任何策略 + 不存在的週期 | 報錯 | Dashboard 已限制只顯示有數據的週期 |

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
├── download_data.bat         # Windows 解壓測試數據
├── download_data.sh          # macOS/Linux 解壓測試數據
├── test-data.zip             # 預轉換 parquet 測試數據
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
