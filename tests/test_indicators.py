"""Indicator correctness and, above all, causality."""

import numpy as np
import pandas as pd
import pytest

from features.technical_indicators import (
    add_technical_indicators,
    calculate_atr,
    calculate_bollinger_bands,
    calculate_ema,
    calculate_macd,
    calculate_rsi,
    calculate_stochastic,
)


def test_indicators_are_causal(synthetic_ohlc):
    """
    Truncating the input must not change any earlier indicator value.

    This is the property that separates a backtest from a fantasy: if an
    indicator at bar t moves when bars after t are removed, the strategy was
    reading the future.
    """
    full = add_technical_indicators(synthetic_ohlc)
    cutoff = 250
    truncated = add_technical_indicators(synthetic_ohlc.iloc[:cutoff])

    indicator_cols = [c for c in truncated.columns if c not in synthetic_ohlc.columns]
    assert indicator_cols

    for col in indicator_cols:
        pd.testing.assert_series_equal(
            full[col].iloc[:cutoff], truncated[col],
            check_names=False, rtol=1e-12, atol=1e-12,
        )


def test_sma_matches_manual_mean(synthetic_ohlc):
    result = add_technical_indicators(synthetic_ohlc)
    expected = synthetic_ohlc['close'].iloc[40:50].mean()
    assert result['SMA_10'].iloc[49] == pytest.approx(expected)


def test_ema_recursive_form():
    prices = pd.Series([10.0] * 5 + [20.0] * 5)
    ema = calculate_ema(prices, span=3)

    assert np.isnan(ema.iloc[1])          # warm-up suppressed
    assert ema.iloc[4] == pytest.approx(10.0)

    alpha = 2 / (3 + 1)
    expected = 10.0 + alpha * (20.0 - 10.0)
    assert ema.iloc[5] == pytest.approx(expected)


def test_rsi_bounds_and_extremes():
    rising = pd.Series(np.arange(1, 60, dtype=float))
    rsi = calculate_rsi(rising)
    assert rsi.dropna().iloc[-1] == pytest.approx(100.0)

    falling = pd.Series(np.arange(60, 1, -1, dtype=float))
    assert calculate_rsi(falling).dropna().iloc[-1] == pytest.approx(0.0)

    flat = pd.Series([100.0] * 40)
    assert calculate_rsi(flat).dropna().iloc[-1] == pytest.approx(50.0)


def test_rsi_within_range(synthetic_ohlc):
    rsi = calculate_rsi(synthetic_ohlc['close']).dropna()
    assert rsi.between(0, 100).all()


def test_macd_histogram_identity(synthetic_ohlc):
    macd = calculate_macd(synthetic_ohlc['close'])
    diff = (macd['MACD'] - macd['Signal']).dropna()
    pd.testing.assert_series_equal(macd['Histogram'].dropna(), diff, check_names=False)


def test_bollinger_bands_are_symmetric(synthetic_ohlc):
    bands = calculate_bollinger_bands(synthetic_ohlc['close'], window=20)
    upper_gap = (bands['Upper'] - bands['Middle']).dropna()
    lower_gap = (bands['Middle'] - bands['Lower']).dropna()
    pd.testing.assert_series_equal(upper_gap, lower_gap, check_names=False)


def test_bollinger_uses_population_std():
    prices = pd.Series(np.arange(1, 41, dtype=float))
    bands = calculate_bollinger_bands(prices, window=20, num_std=2)
    expected_std = prices.iloc[20:40].std(ddof=0)
    width = bands['Upper'].iloc[39] - bands['Middle'].iloc[39]
    assert width == pytest.approx(2 * expected_std)


def test_stochastic_endpoints():
    high = pd.Series(np.arange(10, 40, dtype=float))
    low = high - 5
    close = high.copy()                       # closing at the window high
    stoch = calculate_stochastic(high, low, close, k_window=14)
    assert stoch['%K'].dropna().iloc[-1] == pytest.approx(100.0)


def test_stochastic_flat_window_is_neutral():
    flat = pd.Series([50.0] * 30)
    stoch = calculate_stochastic(flat, flat, flat, k_window=14)
    assert stoch['%K'].dropna().iloc[-1] == pytest.approx(50.0)


def test_atr_true_range_includes_gaps():
    high = pd.Series([10.0, 25.0])
    low = pd.Series([9.0, 24.0])
    close = pd.Series([9.5, 24.5])
    atr = calculate_atr(high, low, close, window=1)
    # Bar 2 gapped up: true range is 25 - 9.5, not 25 - 24.
    assert atr.iloc[1] == pytest.approx(15.5)


def test_volume_columns_optional(synthetic_ohlc):
    """The bundled dataset has no volume; this must not raise."""
    assert 'volume' not in synthetic_ohlc.columns
    result = add_technical_indicators(synthetic_ohlc)
    assert 'Volume_ratio' not in result.columns

    with_volume = synthetic_ohlc.assign(volume=1000.0)
    assert 'Volume_ratio' in add_technical_indicators(with_volume).columns


def test_missing_ohlc_raises(synthetic_ohlc):
    with pytest.raises(KeyError):
        add_technical_indicators(synthetic_ohlc.drop(columns=['high']))
