import os
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.technical_indicators import add_technical_indicators  # noqa: E402


@pytest.fixture(scope='session')
def config():
    with open('config.yaml') as file:
        return yaml.safe_load(file)


@pytest.fixture
def synthetic_ohlc():
    """
    Deterministic 15-minute price series with a known shape.

    Long enough to survive the indicator warm-up: SMA_200 and the 384-bar
    momentum feature between them consume the first ~580 bars.
    """
    n = 1500
    rng = np.random.default_rng(7)
    index = pd.date_range('2026-01-01', periods=n, freq='15min')

    returns = rng.normal(0.00005, 0.002, n)     # 15m-scale volatility
    close = 10_000 * np.exp(np.cumsum(returns))
    spread = close * 0.004

    data = pd.DataFrame({
        'open': close * (1 + rng.normal(0, 0.001, n)),
        'close': close,
        'high': close + spread,
        'low': close - spread,
        'Fear & Greed Index': rng.integers(10, 90, n).astype(float),
    }, index=index)
    data.index.name = 'timestamp'
    return data


@pytest.fixture
def enriched_ohlc(synthetic_ohlc):
    return add_technical_indicators(synthetic_ohlc).dropna()


@pytest.fixture
def real_data():
    """The bundled dataset, or skip when it is unavailable."""
    from utils.data_utils import prepare_dataset

    if not os.path.exists('data/BTC.csv'):
        pytest.skip('data/BTC.csv not present')
    return prepare_dataset('data/BTC.csv')


@pytest.fixture
def state_size(enriched_ohlc):
    """
    Observation width for the current feature set.

    Derived rather than hardcoded: adding a feature should not break unrelated
    tests, it should just widen the network they build.
    """
    from features.observation import PORTFOLIO_FEATURES, build_market_features

    names, _ = build_market_features(enriched_ohlc)
    return len(names) + len(PORTFOLIO_FEATURES)


@pytest.fixture
def n_actions(enriched_ohlc, config):
    """Action count for the configured sizing levels, derived not hardcoded."""
    from environment.trading_env import BitcoinTradingEnv

    return BitcoinTradingEnv(enriched_ohlc, config=config).n_actions
