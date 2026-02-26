"""
Pure-Python / NumPy / pandas indicator library.

All indicators are implemented from scratch so there is no dependency on
TA-Lib or pandas-ta (avoids C-extension build issues on Windows).
Every function accepts and returns pandas Series for seamless integration
with the DataFrame-based backtest engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Trend / moving averages
# ---------------------------------------------------------------------------


def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average (span-based, no look-ahead bias)."""
    return series.ewm(span=period, adjust=False).mean()


# ---------------------------------------------------------------------------
# Oscillators
# ---------------------------------------------------------------------------


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder smoothing via EWM alpha=1/period)."""
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    result = 100.0 - (100.0 / (1.0 + rs))
    return result


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Moving Average Convergence / Divergence.

    Returns:
        (macd_line, signal_line, histogram)
    """
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    ilong: int = 9,
    isig: int = 3,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """KDJ oscillator (TradingView-compatible bcwsma smoothing).

    bcwsma(source, length, weight) = (weight * source + (length - weight) * prev) / length
    For KDJ, weight is always 1.

    Returns:
        (K, D, J) where J = 3*K - 2*D.
    """
    lowest_low = low.rolling(window=ilong, min_periods=ilong).min()
    highest_high = high.rolling(window=ilong, min_periods=ilong).max()

    denom = highest_high - lowest_low
    denom = denom.replace(0.0, np.nan)
    rsv = 100.0 * (close - lowest_low) / denom

    k_values = np.full(len(rsv), np.nan)
    d_values = np.full(len(rsv), np.nan)

    # TradingView initializes K and D at 50
    k_prev = 50.0
    d_prev = 50.0

    for i in range(len(rsv)):
        if np.isnan(rsv.iloc[i]):
            continue
        k_val = (1.0 * rsv.iloc[i] + (isig - 1) * k_prev) / isig
        d_val = (1.0 * k_val + (isig - 1) * d_prev) / isig

        k_values[i] = k_val
        d_values[i] = d_val
        k_prev = k_val
        d_prev = d_val

    k_series = pd.Series(k_values, index=close.index)
    d_series = pd.Series(d_values, index=close.index)
    j_series = 3.0 * k_series - 2.0 * d_series

    return k_series, d_series, j_series


# ---------------------------------------------------------------------------
# Volatility
# ---------------------------------------------------------------------------


def bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands.

    Returns:
        (upper_band, middle_band, lower_band)
    """
    middle = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    return upper, middle, lower


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average True Range (simple rolling mean of TR)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


# ---------------------------------------------------------------------------
# Volume-based
# ---------------------------------------------------------------------------


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
) -> pd.Series:
    """Volume-Weighted Average Price (session cumulative)."""
    typical_price = (high + low + close) / 3.0
    cumulative_tp_vol = (typical_price * volume).cumsum()
    cumulative_vol = volume.cumsum()
    # Avoid division by zero in early bars with no volume
    return cumulative_tp_vol / cumulative_vol.replace(0.0, np.nan)


# ---------------------------------------------------------------------------
# Directional / trend strength
# ---------------------------------------------------------------------------


def adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Average Directional Index (ADX).

    Uses the classic Wilder DM approach.  Because we smooth with EMA
    rather than Wilder's own smoothing, values will be very close but
    not identical to platform implementations.
    """
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.copy()
    minus_dm = down_move.copy()

    # +DM valid only when up_move > down_move and up_move > 0
    plus_dm[(up_move <= down_move) | (up_move <= 0)] = 0.0
    # -DM valid only when down_move > up_move and down_move > 0
    minus_dm[(down_move <= up_move) | (down_move <= 0)] = 0.0

    atr_val = atr(high, low, close, period)
    # Guard against division by zero
    safe_atr = atr_val.replace(0.0, np.nan)

    plus_di = 100.0 * ema(plus_dm, period) / safe_atr
    minus_di = 100.0 * ema(minus_dm, period) / safe_atr

    di_sum = plus_di + minus_di
    safe_sum = di_sum.replace(0.0, np.nan)
    dx = 100.0 * (plus_di - minus_di).abs() / safe_sum
    return ema(dx, period)


# ---------------------------------------------------------------------------
# Order-flow proxies (OHLCV-based approximations)
# ---------------------------------------------------------------------------


def cumulative_delta(
    volume: pd.Series,
    close: pd.Series,
    open_price: pd.Series,
) -> pd.Series:
    """Approximate cumulative delta from OHLCV data.

    Uses the (close - open) / |close - open| heuristic to estimate the
    fraction of volume attributable to buyers vs. sellers.  When
    close == open the bar is treated as neutral (delta = 0).
    """
    diff = close - open_price
    abs_diff = diff.abs()
    # Where the bar has zero body, assign neutral ratio (0.5)
    ratio = np.where(abs_diff > 0.0, diff / abs_diff, 0.0)
    # Normalize from [-1, 1] to [0, 1]: 1 = all buying, 0 = all selling
    buy_ratio = (ratio + 1.0) / 2.0
    buy_vol = volume * buy_ratio
    sell_vol = volume * (1.0 - buy_ratio)
    delta = buy_vol - sell_vol
    return pd.Series(delta, index=volume.index).cumsum()


def volume_imbalance(
    volume: pd.Series,
    close: pd.Series,
    open_price: pd.Series,
    lookback: int = 10,
) -> pd.Series:
    """Rolling buy/sell volume imbalance ratio.

    Returns a value in [-1, 1] where +1 means all buying and -1 means all
    selling over the lookback window.
    """
    diff = close - open_price
    abs_diff = diff.abs()
    ratio = np.where(abs_diff > 0.0, diff / abs_diff, 0.0)
    buy_ratio = (ratio + 1.0) / 2.0
    buy_vol = pd.Series(volume.values * buy_ratio, index=volume.index)
    sell_vol = pd.Series(volume.values * (1.0 - buy_ratio), index=volume.index)

    rolling_buy = buy_vol.rolling(window=lookback, min_periods=lookback).sum()
    rolling_sell = sell_vol.rolling(window=lookback, min_periods=lookback).sum()
    total = rolling_buy + rolling_sell
    safe_total = total.replace(0.0, np.nan)
    return (rolling_buy - rolling_sell) / safe_total
