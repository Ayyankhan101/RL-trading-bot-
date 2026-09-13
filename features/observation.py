"""
Observation construction, shared by the backtest environment and the live trader.

This module exists so the two can never drift. If the live runner built its
features even slightly differently from the environment the agent trained in,
the policy would be reading a different input than it learned on - and the live
results would be meaningless while still looking plausible.

All market features are scale-free (ratios or bounded oscillators), so a policy
learned at $6k BTC stays meaningful at $118k.
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

PORTFOLIO_FEATURES: List[str] = [
    'cash_fraction',
    'position_fraction',
    'unrealized_pnl',
    'drawdown',
    'in_position',
]


def build_market_features(data: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
    """
    Build the scale-free market feature matrix for an indicator-enriched frame.

    Returns the feature names and a ``(n_bars, n_features)`` float32 matrix.
    Row ``i`` depends only on rows ``0..i`` of the input, because every column it
    reads is already causal.
    """
    close = data['close']
    features: Dict[str, pd.Series] = {}

    for col in ('SMA_10', 'SMA_20', 'SMA_50', 'SMA_200', 'EMA_12', 'EMA_26', 'BB_middle'):
        if col in data.columns:
            features[f'close_over_{col}'] = close / data[col] - 1.0

    if 'RSI' in data.columns:
        features['rsi'] = data['RSI'] / 100.0
    if 'STOCH_k' in data.columns:
        features['stoch_k'] = data['STOCH_k'] / 100.0
    if 'STOCH_d' in data.columns:
        features['stoch_d'] = data['STOCH_d'] / 100.0

    for col in ('MACD', 'MACD_signal', 'MACD_histogram'):
        if col in data.columns:
            features[col.lower()] = data[col] / close

    if {'BB_upper', 'BB_lower'} <= set(data.columns):
        band = (data['BB_upper'] - data['BB_lower']).replace(0, np.nan)
        features['bb_position'] = (close - data['BB_lower']) / band
        features['bb_width'] = band / close

    if 'ATR' in data.columns:
        features['atr_pct'] = data['ATR'] / close
    if 'Volatility' in data.columns:
        features['volatility'] = data['Volatility']
    if 'Volume_ratio' in data.columns:
        features['volume_ratio'] = data['Volume_ratio'] - 1.0

    features['return_1'] = close.pct_change(1)
    for col in ('Price_momentum_5', 'Price_momentum_10',
                'Price_momentum_96', 'Price_momentum_384'):
        if col in data.columns:
            features[col.lower()] = data[col]

    if 'Fear & Greed Index' in data.columns:
        features['fear_greed'] = data['Fear & Greed Index'] / 100.0

    frame = pd.DataFrame(features, index=data.index)
    # A NaN reaching the network would poison it silently; the indicator warm-up
    # is already trimmed upstream, so anything left here is a genuine gap.
    matrix = np.nan_to_num(frame.to_numpy(dtype=np.float32),
                           nan=0.0, posinf=0.0, neginf=0.0)
    return list(frame.columns), matrix


def build_portfolio_features(balance: float, position_value: float, equity: float,
                             entry_price: float | None, price: float,
                             peak_equity: float) -> np.ndarray:
    """Portfolio half of the observation, identical for backtest and live."""
    unrealized = 0.0
    if position_value > 0 and entry_price:
        unrealized = (price / entry_price) - 1.0

    drawdown = (equity - peak_equity) / peak_equity if peak_equity > 0 else 0.0

    return np.array([
        balance / equity if equity > 0 else 0.0,
        position_value / equity if equity > 0 else 0.0,
        unrealized,
        drawdown,
        1.0 if position_value > 0 else 0.0,
    ], dtype=np.float32)


def full_feature_names(market_names: List[str]) -> List[str]:
    return list(market_names) + list(PORTFOLIO_FEATURES)
