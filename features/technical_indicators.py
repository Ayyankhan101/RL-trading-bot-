"""
Technical Indicators for Bitcoin Trading

Every indicator here is strictly causal: the value at row ``i`` is a function of
rows ``0..i`` only. ``tests/test_indicators.py`` asserts this by truncating the
input and checking that earlier values are unchanged.
"""

import pandas as pd
import numpy as np
from typing import Dict, List

# Indicator columns produced by add_technical_indicators() that do not depend on
# volume. Kept as a module constant so the environment can build a stable,
# explicit feature list instead of guessing at column names.
PRICE_INDICATOR_COLUMNS: List[str] = [
    'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
    'EMA_12', 'EMA_26',
    'RSI',
    'MACD', 'MACD_signal', 'MACD_histogram',
    'BB_upper', 'BB_middle', 'BB_lower', 'BB_width',
    'STOCH_k', 'STOCH_d',
    'ATR',
    'Price_momentum_5', 'Price_momentum_10',
    'Price_momentum_96', 'Price_momentum_384',
    'Volatility',
]

VOLUME_INDICATOR_COLUMNS: List[str] = ['Volume_SMA', 'Volume_ratio']


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Element-wise division that yields NaN instead of inf where the denominator is 0."""
    return numerator / denominator.replace(0, np.nan)


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to an OHLC(V) dataframe.

    The ``volume`` column is optional: the bundled dataset (data/BTC.csv) has no
    volume, so volume-derived indicators are only added when the column exists.

    Args:
        df: DataFrame with at least open/high/low/close columns, indexed
            chronologically (oldest first).

    Returns:
        DataFrame with indicator columns appended.
    """
    required = {'high', 'low', 'close'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"add_technical_indicators requires columns {sorted(missing)}")

    data = df.copy()

    # Simple Moving Averages
    data['SMA_10'] = data['close'].rolling(window=10).mean()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['SMA_50'] = data['close'].rolling(window=50).mean()
    # A long trend anchor. At 4h this is ~33 days; at 15m it is ~2 days, which
    # is the shortest horizon at which a 15m series shows any trend at all.
    data['SMA_200'] = data['close'].rolling(window=200).mean()

    # Exponential Moving Averages (recursive form, warm-up suppressed)
    data['EMA_12'] = calculate_ema(data['close'], 12)
    data['EMA_26'] = calculate_ema(data['close'], 26)

    # RSI (Wilder)
    data['RSI'] = calculate_rsi(data['close'])

    # MACD
    macd_data = calculate_macd(data['close'])
    data['MACD'] = macd_data['MACD']
    data['MACD_signal'] = macd_data['Signal']
    data['MACD_histogram'] = macd_data['Histogram']

    # Bollinger Bands
    bb_data = calculate_bollinger_bands(data['close'])
    data['BB_upper'] = bb_data['Upper']
    data['BB_middle'] = bb_data['Middle']
    data['BB_lower'] = bb_data['Lower']
    data['BB_width'] = bb_data['Width']

    # Stochastic Oscillator
    stoch_data = calculate_stochastic(data['high'], data['low'], data['close'])
    data['STOCH_k'] = stoch_data['%K']
    data['STOCH_d'] = stoch_data['%D']

    # Average True Range (Wilder)
    data['ATR'] = calculate_atr(data['high'], data['low'], data['close'])

    # Volume indicators - only when the dataset actually carries volume
    if 'volume' in data.columns:
        data['Volume_SMA'] = data['volume'].rolling(window=20).mean()
        data['Volume_ratio'] = _safe_divide(data['volume'], data['Volume_SMA'])

    # Price momentum. The 5/10-bar pair spans barely two hours on 15m bars, so
    # the longer pair supplies the day- and week-scale context the agent would
    # otherwise be blind to.
    data['Price_momentum_5'] = data['close'].pct_change(5)
    data['Price_momentum_10'] = data['close'].pct_change(10)
    data['Price_momentum_96'] = data['close'].pct_change(96)
    data['Price_momentum_384'] = data['close'].pct_change(384)

    # Volatility: rolling std of returns, not of raw price. A price-level std is
    # unusable as a model feature because it scales with the price itself.
    data['Volatility'] = data['close'].pct_change().rolling(window=20).std(ddof=0)

    return data


def indicator_columns(df: pd.DataFrame) -> List[str]:
    """Indicator columns present in ``df``, in a deterministic order."""
    candidates = PRICE_INDICATOR_COLUMNS + VOLUME_INDICATOR_COLUMNS
    return [col for col in candidates if col in df.columns]


def calculate_ema(prices: pd.Series, span: int) -> pd.Series:
    """Standard recursive EMA (adjust=False), with the warm-up period masked out."""
    return prices.ewm(span=span, adjust=False, min_periods=span).mean()


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing is an EMA with alpha = 1/window.
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    rs = _safe_divide(avg_gain, avg_loss)
    rsi = 100 - (100 / (1 + rs))

    # All-gain windows have zero average loss: RSI is 100 by definition there.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    # Flat windows (no gain, no loss) are conventionally neutral.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)
    return rsi


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26,
                   signal: int = 9) -> Dict[str, pd.Series]:
    """MACD line, signal line and histogram."""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False, min_periods=signal).mean()
    histogram = macd - signal_line

    return {'MACD': macd, 'Signal': signal_line, 'Histogram': histogram}


def calculate_bollinger_bands(prices: pd.Series, window: int = 20,
                              num_std: float = 2) -> Dict[str, pd.Series]:
    """Bollinger Bands using the population standard deviation (ddof=0)."""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std(ddof=0)

    upper = sma + (std * num_std)
    lower = sma - (std * num_std)

    return {'Upper': upper, 'Middle': sma, 'Lower': lower, 'Width': upper - lower}


def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                         k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
    """Stochastic Oscillator %K and %D."""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()

    k_percent = 100 * _safe_divide(close - lowest_low, highest_high - lowest_low)
    # A perfectly flat window has no range; treat it as mid-band.
    k_percent = k_percent.mask(highest_high == lowest_low, 50.0)
    d_percent = k_percent.rolling(window=d_window).mean()

    return {'%K': k_percent, '%D': d_percent}


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series,
                  window: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing."""
    prev_close = close.shift()
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
