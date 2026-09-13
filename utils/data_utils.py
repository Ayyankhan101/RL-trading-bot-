"""
Data handling utilities for the RL Trading Bot.

Loading is deliberately strict: the environment walks the dataframe forwards in
time, so a file that is sorted newest-first (as data/BTC.csv is on disk) must be
reordered before anything else touches it.
"""

import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd

FEAR_GREED_COLUMN = 'Fear & Greed Index'


def download_bitcoin_data(symbol: str = "BTC-USD",
                          period: str = "5y",
                          interval: str = "1d") -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.

    yfinance is imported lazily so that the rest of this module - and therefore
    the whole training/backtesting pipeline - works without it installed.
    """
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for download_bitcoin_data(). "
            "Install it with: pip install yfinance"
        ) from exc

    ticker = yf.Ticker(symbol)
    data = ticker.history(period=period, interval=interval)
    if data.empty:
        raise ValueError(f"No data returned for {symbol} (period={period}, interval={interval})")

    data.columns = [str(col).lower() for col in data.columns]
    data.reset_index(inplace=True)
    for candidate in ('Date', 'date', 'Datetime', 'datetime', 'index'):
        if candidate in data.columns:
            data.rename(columns={candidate: 'timestamp'}, inplace=True)
            break

    data['timestamp'] = pd.to_datetime(data['timestamp'], utc=True).dt.tz_localize(None)
    data = data.sort_values('timestamp').set_index('timestamp')
    print(f"Downloaded {len(data)} rows of {symbol} data")
    return data


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load an OHLC(V) CSV into a chronologically-sorted, DatetimeIndex-ed frame.

    Raises rather than returning an empty frame: silently continuing with no data
    is how a backtest ends up reporting numbers for a run that never happened.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    data = pd.read_csv(file_path)

    if 'timestamp' not in data.columns:
        raise KeyError(f"{file_path} has no 'timestamp' column (found: {list(data.columns)})")

    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = (data
            .drop_duplicates(subset='timestamp', keep='last')
            .sort_values('timestamp')
            .set_index('timestamp'))

    missing = {'open', 'high', 'low', 'close'} - set(data.columns)
    if missing:
        raise KeyError(f"{file_path} is missing OHLC columns: {sorted(missing)}")

    if not data.index.is_monotonic_increasing:
        raise ValueError("Data index is not monotonically increasing after sorting")

    print(f"Loaded {len(data)} rows from {file_path} "
          f"({data.index[0]} -> {data.index[-1]})")
    return data


def infer_periods_per_year(index: pd.DatetimeIndex) -> float:
    """
    Periods per year implied by the median spacing of ``index``.

    Annualization factors must come from the data. The previous code hardcoded
    252 (daily) while the bundled dataset is 4-hourly, overstating every
    annualized figure by roughly 2.9x.
    """
    if len(index) < 3:
        return 365.0

    median_delta = pd.Series(index).diff().median()
    seconds = median_delta.total_seconds()
    if not seconds or seconds <= 0:
        return 365.0

    # Crypto trades continuously: 365 calendar days, not 252 trading days.
    return (365.0 * 24 * 60 * 60) / seconds


def save_data(data: pd.DataFrame, file_path: str) -> None:
    """Write a dataframe to CSV, creating parent directories when needed."""
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    data.to_csv(file_path)
    print(f"Saved {len(data)} rows to {file_path}")


def split_data(data: pd.DataFrame, train_fraction: float = 0.8,
               purge: int = 0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split with an optional purge gap.

    The gap drops ``purge`` bars between the two sets so that no indicator
    lookback window straddles the boundary and leaks train data into test.
    """
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")

    split_idx = int(len(data) * train_fraction)
    train = data.iloc[:split_idx]
    test = data.iloc[split_idx + purge:]
    return train, test


def normalize_causal(series: pd.Series, window: int = 200) -> pd.Series:
    """
    Rolling z-score using only past observations.

    Deliberately rolling rather than whole-series: normalizing by the full
    dataframe's mean and standard deviation - as the previous implementation
    did - leaks future information into every historical row.
    """
    mean = series.rolling(window=window, min_periods=window).mean()
    std = series.rolling(window=window, min_periods=window).std(ddof=0)
    return (series - mean) / std.replace(0, np.nan)


def ensure_fear_greed(data: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee a usable Fear & Greed column.

    data/BTC.csv ships the real index, so the normal path is a forward-fill of
    the handful of missing readings. When the column is absent entirely the
    column is filled with the neutral value 50 - a flat constant that carries no
    information, rather than a synthetic index derived from the price series,
    which would masquerade as sentiment while being a function of the very
    prices the agent already observes.
    """
    out = data.copy()

    if FEAR_GREED_COLUMN in out.columns:
        out[FEAR_GREED_COLUMN] = (out[FEAR_GREED_COLUMN]
                                  .ffill()
                                  .fillna(50.0)
                                  .astype(float)
                                  .clip(0, 100))
    else:
        out[FEAR_GREED_COLUMN] = 50.0

    return out


def validate_data(data: pd.DataFrame) -> None:
    """Raise on the data defects that would silently corrupt a backtest."""
    problems = []

    if data.empty:
        problems.append("dataframe is empty")
    if not isinstance(data.index, pd.DatetimeIndex):
        problems.append("index is not a DatetimeIndex")
    elif not data.index.is_monotonic_increasing:
        problems.append("index is not sorted ascending")

    for col in ('open', 'high', 'low', 'close'):
        if col not in data.columns:
            problems.append(f"missing column '{col}'")
        elif (data[col] <= 0).any():
            problems.append(f"column '{col}' contains non-positive prices")

    if {'high', 'low'} <= set(data.columns) and (data['high'] < data['low']).any():
        problems.append("high < low on at least one bar")

    if problems:
        raise ValueError("Invalid market data: " + "; ".join(problems))


def prepare_dataset(file_path: str, add_indicators: bool = True) -> pd.DataFrame:
    """Load, validate, enrich and clean a dataset in one call."""
    from features.technical_indicators import add_technical_indicators

    data = load_data(file_path)
    validate_data(data)
    data = ensure_fear_greed(data)

    if add_indicators:
        data = add_technical_indicators(data)
        data = data.dropna(subset=[c for c in data.columns if c != 'Fear & Greed Classification'])

    print(f"Prepared dataset: {len(data)} rows, {len(data.columns)} columns")
    return data
