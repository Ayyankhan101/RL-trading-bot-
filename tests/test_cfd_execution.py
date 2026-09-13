"""CFD execution: gold spreads, per-lot commission, and overnight financing."""

import copy

import numpy as np
import pandas as pd
import pytest

from environment.execution import (
    BUY,
    SELL,
    cfd_fill,
    execution_config,
    financing_cost,
    is_cfd,
    taker_fill,
)

GOLD = {'mode': 'cfd', 'spread_points': 0.15, 'commission_per_lot': 7.0,
        'contract_size': 100.0, 'swap_long_daily': -0.00005,
        'swap_short_daily': -0.00002}
PRICE = 4408.90


def test_buy_pays_the_ask_and_sell_receives_the_bid():
    buy = cfd_fill(BUY, PRICE, {'execution': GOLD})
    sell = cfd_fill(SELL, PRICE, {'execution': GOLD})

    assert buy.price == pytest.approx(PRICE + 0.075)
    assert sell.price == pytest.approx(PRICE - 0.075)
    assert buy.price - sell.price == pytest.approx(GOLD['spread_points'])


def test_spread_is_a_price_offset_not_a_percentage():
    """
    The reason CFDs need their own mode. A crypto fee scales with notional; a
    gold spread is dollars per ounce and does not. Modelling one as the other
    misprices every trade by an amount that depends on position size.
    """
    cheap = cfd_fill(BUY, 1_000.0, {'execution': GOLD})
    dear = cfd_fill(BUY, 10_000.0, {'execution': GOLD})

    assert cheap.price - 1_000.0 == pytest.approx(dear.price - 10_000.0)


def test_round_trip_matches_broker_arithmetic():
    """0.01 lot of gold: $0.15 of spread plus $0.07 of commission = $0.22."""
    ounces = 1.0                                  # 0.01 standard lot
    buy = cfd_fill(BUY, PRICE, {'execution': GOLD})
    sell = cfd_fill(SELL, PRICE, {'execution': GOLD})

    spread_cost = (buy.price - sell.price) * ounces
    commission = 2 * buy.fee_rate * (ounces * PRICE)

    assert spread_cost + commission == pytest.approx(0.22, abs=1e-6)


def test_cfd_round_trip_is_far_cheaper_than_crypto():
    """
    The whole reason gold is viable at 15m where BTC is not: 0.005% against
    0.30%, a 60x difference, on an asset that moves 0.84x as much.
    """
    buy = cfd_fill(BUY, PRICE, {'execution': GOLD})
    sell = cfd_fill(SELL, PRICE, {'execution': GOLD})
    cfd_cost = ((buy.price - sell.price) / PRICE) + 2 * buy.fee_rate

    crypto_cost = 2 * (0.001 + 0.0005)

    assert cfd_cost < crypto_cost / 20


def test_taker_fill_routes_to_cfd_when_configured():
    routed = taker_fill(BUY, PRICE, {'execution': GOLD})
    direct = cfd_fill(BUY, PRICE, {'execution': GOLD})

    assert is_cfd({'execution': GOLD})
    assert routed.price == pytest.approx(direct.price)


def test_financing_is_charged_and_scales_with_time_held():
    """Overnight financing is not free, and a longer hold pays more of it."""
    hour = financing_cost(10_000.0, 60, True, {'execution': GOLD})
    day = financing_cost(10_000.0, 1440, True, {'execution': GOLD})

    assert hour < 0
    assert day == pytest.approx(hour * 24)
    assert day == pytest.approx(10_000.0 * GOLD['swap_long_daily'])


def test_financing_differs_by_direction():
    long_cost = financing_cost(10_000.0, 1440, True, {'execution': GOLD})
    short_cost = financing_cost(10_000.0, 1440, False, {'execution': GOLD})

    assert long_cost != short_cost


def test_crypto_config_is_unaffected_by_cfd_defaults():
    crypto = {'execution': {'mode': 'taker', 'taker_fee': 0.001, 'slippage': 0.0005}}
    fill = taker_fill(BUY, 100_000.0, crypto)

    assert not is_cfd(crypto)
    assert fill.price == pytest.approx(100_050.0)


def test_env_charges_financing_on_a_held_position(enriched_ohlc, config):
    from environment.trading_env import BitcoinTradingEnv

    cfg = copy.deepcopy(config)
    cfg['execution'] = dict(GOLD)
    cfg['risk'] = {'min_hold_bars': 0}
    cfg['regime'] = {'enabled': False}

    env = BitcoinTradingEnv(enriched_ohlc, config=cfg)
    env.reset(seed=1)
    env.step(1)
    for _ in range(30):
        env.step(0)

    assert env.financing_paid < 0
