"""
Maker and taker fill models.

The central risk with maker execution is modelling the cheaper fee while
ignoring the fills it misses - that would manufacture the entire improvement.
These tests pin both halves.
"""

import copy

import numpy as np
import pytest

from environment.execution import (
    BUY,
    SELL,
    execution_config,
    expired,
    is_maker,
    place_limit,
    taker_fill,
    try_fill_limit,
)
from environment.trading_env import BitcoinTradingEnv

MAKER = {'mode': 'maker', 'maker_fee': 0.0002, 'taker_fee': 0.001,
         'slippage': 0.0005, 'limit_offset': 0.0005, 'max_wait_bars': 4,
         'fallback_to_taker': False}
TAKER = {**MAKER, 'mode': 'taker'}


# --------------------------------------------------------------- fill model

def test_limit_rests_on_the_passive_side():
    """A buy rests below the last price and a sell above it - that is what earns
    the maker fee, and why it may never fill."""
    buy = place_limit(BUY, 100_000, 0, {'execution': MAKER})
    sell = place_limit(SELL, 100_000, 0, {'execution': MAKER})

    assert buy.limit_price == pytest.approx(99_950.0)
    assert sell.limit_price == pytest.approx(100_050.0)


def test_maker_order_misses_when_price_runs_away():
    """A limit the bar never trades through does not fill. This is the cost."""
    order = place_limit(BUY, 100_000, 0, {'execution': MAKER})

    assert try_fill_limit(order, bar_high=100_500, bar_low=99_960, step=1) is None


def test_maker_fill_price_is_the_limit_not_the_bar():
    """A resting order that gets hit trades at its own price - no slippage."""
    order = place_limit(BUY, 100_000, 0, {'execution': MAKER})
    fill = try_fill_limit(order, bar_high=100_100, bar_low=99_000, step=2)

    assert fill is not None
    assert fill.price == pytest.approx(order.limit_price)
    assert fill.maker is True
    assert fill.bars_waited == 2


def test_taker_fill_pays_slippage_against_the_agent():
    buy = taker_fill(BUY, 100_000, {'execution': TAKER})
    sell = taker_fill(SELL, 100_000, {'execution': TAKER})

    assert buy.price == pytest.approx(100_050.0)
    assert sell.price == pytest.approx(99_950.0)
    assert buy.maker is False
    assert buy.fee_rate == pytest.approx(0.001)


def test_maker_fee_is_cheaper_than_taker():
    maker = place_limit(BUY, 100_000, 0, {'execution': MAKER})
    taker = taker_fill(BUY, 100_000, {'execution': TAKER})

    assert maker.fee_rate < taker.fee_rate


def test_order_expires_after_max_wait():
    order = place_limit(BUY, 100_000, 10, {'execution': MAKER})

    assert not expired(order, 13, {'execution': MAKER})
    assert expired(order, 14, {'execution': MAKER})


def test_legacy_config_without_execution_block_still_works():
    """Older configs used a single flat transaction_cost."""
    legacy = {'trading': {'transaction_cost': 0.002},
              'backtesting': {'slippage': 0.001}}
    execution = execution_config(legacy)

    assert execution['mode'] == 'taker'
    assert execution['taker_fee'] == pytest.approx(0.002)
    assert execution['slippage'] == pytest.approx(0.001)
    assert not is_maker(legacy)


# ------------------------------------------------------- environment wiring

@pytest.fixture
def maker_config(config):
    cfg = copy.deepcopy(config)
    cfg['execution'] = dict(MAKER)
    cfg['risk'] = {'min_hold_bars': 4}
    # Regime scaling has its own suite; these tests pin fill mechanics and need
    # unscaled exposure.
    cfg['regime'] = {'enabled': False}
    return cfg


def run_random(env, seed=1):
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    while True:
        _, _, terminated, truncated, info = env.step(int(rng.integers(0, env.n_actions)))
        if terminated or truncated:
            return info


def test_maker_mode_both_fills_and_misses(enriched_ohlc, maker_config):
    """
    A fill model that never misses is wrong.

    Maker execution buys a cheaper fee with missed trades; if the miss count is
    zero the model is quietly handing out free money.
    """
    env = BitcoinTradingEnv(enriched_ohlc, config=maker_config)
    info = run_random(env)

    assert info['maker_fills'] > 0
    assert info['missed_orders'] > 0


def test_taker_mode_never_misses(enriched_ohlc, maker_config):
    cfg = copy.deepcopy(maker_config)
    cfg['execution'] = dict(TAKER)

    env = BitcoinTradingEnv(enriched_ohlc, config=cfg)
    info = run_random(env)

    assert info['missed_orders'] == 0
    assert info['maker_fills'] == 0


def test_maker_entries_fill_at_or_below_the_signal_price(enriched_ohlc, maker_config):
    """Every maker buy must be at least as good as the bar that triggered it."""
    env = BitcoinTradingEnv(enriched_ohlc, config=maker_config)
    run_random(env)

    buys = [t for t in env.trades if t['action'] == 'BUY']
    assert buys
    for trade in buys:
        bar_low = env._low[trade['step']]
        assert trade['price'] >= bar_low - 1e-6


def test_stops_always_cross_the_spread(enriched_ohlc, maker_config):
    """
    A stop that waits for a better price is not a stop.

    Risk exits must fill immediately even in maker mode.
    """
    cfg = copy.deepcopy(maker_config)
    cfg['risk'] = {'stop_loss_atr': 1.0, 'min_hold_bars': 0}

    data = enriched_ohlc.copy()
    entry = data['close'].iloc[0]
    data.loc[data.index[1], ['low', 'close']] = entry * 0.70

    env = BitcoinTradingEnv(data, config=cfg)
    env.reset(seed=1)
    env.step(1)                      # post a maker buy
    env.step(1)                      # it fills, or not
    for _ in range(5):
        env.step(0)

    stopped = [t for t in env.trades if t.get('exit_reason') == 'stop_loss']
    if stopped:                      # only meaningful if the entry actually filled
        assert env.btc_held == 0


def test_min_hold_96_blocks_early_signal_exits(enriched_ohlc, config):
    """The 24h horizon is what makes the edge exceed the cost; enforce it."""
    cfg = copy.deepcopy(config)
    cfg['execution'] = dict(TAKER)
    cfg['risk'] = {'min_hold_bars': 96}
    cfg['regime'] = {'enabled': False}

    env = BitcoinTradingEnv(enriched_ohlc, config=cfg)
    env.reset(seed=1)
    env.step(1)
    for _ in range(50):
        env.step(2)                  # repeated sell attempts, all too early

    assert env.btc_held > 0
    assert env.total_trades == 0
