"""Loading guarantees: ordering, index type, causal normalization."""

import numpy as np
import pandas as pd
import pytest

from utils.data_utils import (
    ensure_fear_greed,
    infer_periods_per_year,
    load_data,
    normalize_causal,
    split_data,
    validate_data,
)


def test_load_data_sorts_oldest_first(tmp_path, synthetic_ohlc):
    """
    data/BTC.csv ships newest-first. Loading it as-is would walk the environment
    backwards through time, so the loader must reorder unconditionally.
    """
    path = tmp_path / 'reversed.csv'
    synthetic_ohlc.iloc[::-1].to_csv(path)

    loaded = load_data(str(path))

    assert isinstance(loaded.index, pd.DatetimeIndex)
    assert loaded.index.is_monotonic_increasing
    assert loaded.index[0] == synthetic_ohlc.index[0]


def test_load_data_drops_duplicate_timestamps(tmp_path, synthetic_ohlc):
    doubled = pd.concat([synthetic_ohlc, synthetic_ohlc.iloc[:5]])
    path = tmp_path / 'dupes.csv'
    doubled.to_csv(path)

    assert len(load_data(str(path))) == len(synthetic_ohlc)


def test_load_data_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_data('does/not/exist.csv')


def test_load_data_missing_ohlc_raises(tmp_path, synthetic_ohlc):
    path = tmp_path / 'partial.csv'
    synthetic_ohlc.drop(columns=['high']).to_csv(path)
    with pytest.raises(KeyError):
        load_data(str(path))


def test_infer_periods_per_year_for_4h_bars():
    index = pd.date_range('2021-01-01', periods=500, freq='4h')
    assert infer_periods_per_year(index) == pytest.approx(2190.0)


def test_infer_periods_per_year_for_daily_bars():
    index = pd.date_range('2021-01-01', periods=500, freq='1D')
    assert infer_periods_per_year(index) == pytest.approx(365.0)


def test_split_data_purges_the_boundary():
    frame = pd.DataFrame({'x': range(100)})
    train, test = split_data(frame, train_fraction=0.8, purge=10)

    assert len(train) == 80
    assert test.index[0] == 90          # 10 bars dropped at the seam


def test_normalize_causal_ignores_the_future():
    """Whole-series z-scoring leaks the future into every historical row."""
    series = pd.Series(np.arange(300, dtype=float))
    full = normalize_causal(series, window=50)
    truncated = normalize_causal(series.iloc[:200], window=50)

    pd.testing.assert_series_equal(full.iloc[:200], truncated, rtol=1e-12)


def test_ensure_fear_greed_fills_gaps(synthetic_ohlc):
    data = synthetic_ohlc.copy()
    data.loc[data.index[5:8], 'Fear & Greed Index'] = np.nan

    filled = ensure_fear_greed(data)
    assert filled['Fear & Greed Index'].notna().all()
    assert filled['Fear & Greed Index'].between(0, 100).all()


def test_ensure_fear_greed_defaults_to_neutral(synthetic_ohlc):
    """Absent sentiment becomes a constant 50 - never a price-derived fake."""
    data = synthetic_ohlc.drop(columns=['Fear & Greed Index'])
    filled = ensure_fear_greed(data)
    assert (filled['Fear & Greed Index'] == 50.0).all()


def test_validate_data_rejects_bad_bars(synthetic_ohlc):
    broken = synthetic_ohlc.copy()
    broken.loc[broken.index[3], 'high'] = broken['low'].iloc[3] - 1

    with pytest.raises(ValueError):
        validate_data(broken)


def test_validate_data_rejects_non_positive_prices(synthetic_ohlc):
    broken = synthetic_ohlc.copy()
    broken.loc[broken.index[2], 'close'] = 0.0

    with pytest.raises(ValueError):
        validate_data(broken)
