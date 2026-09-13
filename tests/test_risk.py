"""ATR-scaled stops and the minimum holding period."""

import copy

import numpy as np
import pandas as pd
import pytest

from environment.risk import (
    ExitLevels,
    can_exit_on_signal,
    check_exit,
    min_hold_bars,
    stop_and_target,
)
from environment.trading_env import BUY, HOLD, SELL, BitcoinTradingEnv
from live.paper_broker import PaperBroker

ATR_CONFIG = {'risk': {'stop_loss_atr': 3.0, 'take_profit_atr': 6.0, 'min_hold_bars': 24}}
PCT_CONFIG = {'risk': {'stop_loss': 0.05, 'take_profit': 0.15}}


# ------------------------------------------------------------------- levels

def test_atr_stops_scale_with_volatility():
    """Double the ATR at entry, double the stop distance."""
    quiet = stop_and_target(100_000, 100, ATR_CONFIG)
    wild = stop_and_target(100_000, 200, ATR_CONFIG)

    assert 100_000 - quiet.stop == pytest.approx(300.0)
    assert 100_000 - wild.stop == pytest.approx(600.0)
    assert wild.target - 100_000 == pytest.approx(2 * (quiet.target - 100_000))


def test_atr_levels_clear_the_round_trip_cost_at_15m():
    """
    A 15m ATR is ~0.19% of price. The resulting stop and target must sit outside
    the ~0.30% round-trip cost, or every exit is a guaranteed loss.
    """
    price, atr = 100_000.0, 190.0
    levels = stop_and_target(price, atr, ATR_CONFIG)
    round_trip_cost = 2 * (0.001 + 0.0005)

    assert (price - levels.stop) / price > round_trip_cost
    assert (levels.target - price) / price > round_trip_cost


def test_percentage_settings_still_work_without_atr():
    """Existing 4h configs keep working unchanged."""
    levels = stop_and_target(100_000, None, PCT_CONFIG)

    assert levels.stop == pytest.approx(95_000)
    assert levels.target == pytest.approx(115_000)


def test_atr_falls_back_to_percentages_during_warmup():
    config = {'risk': {**ATR_CONFIG['risk'], **PCT_CONFIG['risk']}}
    warmup = stop_and_target(100_000, None, config)

    assert warmup.stop == pytest.approx(95_000)


def test_distance_clamps_bound_the_stop():
    config = {'risk': {'stop_loss_atr': 3.0, 'min_stop_pct': 0.004, 'max_stop_pct': 0.02}}

    tiny_atr = stop_and_target(100_000, 1.0, config)      # would be 0.003%
    huge_atr = stop_and_target(100_000, 5000.0, config)   # would be 15%

    assert 100_000 - tiny_atr.stop == pytest.approx(400.0)    # floored at 0.4%
    assert 100_000 - huge_atr.stop == pytest.approx(2000.0)   # capped at 2%


def test_no_levels_configured_means_no_exits():
    assert stop_and_target(100, 1, {'risk': {}}) == ExitLevels(None, None)
    assert check_exit(200, 1, ExitLevels(None, None)) is None


def test_stop_beats_target_inside_one_bar():
    """The bar's internal path is unknown; the optimistic read invents profit."""
    levels = stop_and_target(100_000, 190, ATR_CONFIG)
    reason, fill = check_exit(high=102_000, low=99_000, levels=levels)

    assert reason == 'stop_loss'
    assert fill == pytest.approx(levels.stop)


def test_exit_fills_at_the_level_not_the_bar_extreme():
    levels = stop_and_target(100_000, 190, ATR_CONFIG)
    _, fill = check_exit(high=100_100, low=50_000, levels=levels)

    assert fill == pytest.approx(levels.stop)


# ----------------------------------------------------------------- min hold

def test_min_hold_gate():
    assert min_hold_bars(ATR_CONFIG) == 24
    assert not can_exit_on_signal(23, ATR_CONFIG)
    assert can_exit_on_signal(24, ATR_CONFIG)


def test_min_hold_absent_means_no_restriction():
    assert can_exit_on_signal(0, {'risk': {}})


# ------------------------------------------------------- env / broker parity

@pytest.fixture
def risk_config(config):
    cfg = copy.deepcopy(config)
    cfg['backtesting']['slippage'] = 0.0
    cfg['risk'] = {'stop_loss_atr': 3.0, 'take_profit_atr': 6.0, 'min_hold_bars': 24}
    # Taker mode: these tests compare env and broker entries directly, which
    # requires both to fill immediately at the same price.
    cfg['execution'] = {'mode': 'taker', 'taker_fee': cfg['trading']['transaction_cost'],
                        'maker_fee': cfg['trading']['transaction_cost'], 'slippage': 0.0}
    # Regime scaling is exercised in tests/test_regime.py; these pin raw
    # execution mechanics and need full, unscaled exposure.
    cfg['regime'] = {'enabled': False}
    return cfg


def test_min_hold_blocks_a_signal_exit(enriched_ohlc, risk_config):
    env = BitcoinTradingEnv(enriched_ohlc, config=risk_config)
    env.reset(seed=1)
    env.step(BUY)

    for _ in range(5):
        env.step(SELL)

    assert env.btc_held > 0          # still long: 5 bars < 24
    assert env.total_trades == 0


def test_signal_exit_allowed_once_the_hold_elapses(enriched_ohlc, risk_config):
    risk_config['risk'] = {'min_hold_bars': 5}       # no stops, isolate the hold
    env = BitcoinTradingEnv(enriched_ohlc, config=risk_config)
    env.reset(seed=1)
    env.step(BUY)

    for _ in range(6):
        env.step(HOLD)
    env.step(SELL)

    assert env.btc_held == 0
    assert env.total_trades == 1


def test_min_hold_never_blocks_a_stop(enriched_ohlc, risk_config):
    """Risk control is not a trading signal; the hold must not suppress it."""
    data = enriched_ohlc.copy()
    entry = data['close'].iloc[0]
    data.loc[data.index[1], ['low', 'close']] = entry * 0.80

    env = BitcoinTradingEnv(data, config=risk_config)
    env.reset(seed=1)
    _, _, _, _, info = env.step(BUY)

    assert info.get('exit_reason') == 'stop_loss'
    assert env.btc_held == 0


def test_env_and_broker_agree_on_exit_levels(enriched_ohlc, risk_config, tmp_path):
    """
    Both implementations must resolve the same bar to the same exit.

    They share environment/risk.py precisely so a live fill matches what the
    backtest claimed it would be.
    """
    env = BitcoinTradingEnv(enriched_ohlc, config=risk_config)
    env.reset(seed=1)
    env.step(BUY)

    broker = PaperBroker(risk_config, str(tmp_path / 'account.json'))
    broker.buy(env.entry_price, '2026-01-01', atr=env.entry_atr)

    env_levels = stop_and_target(env.entry_price, env.entry_atr, risk_config)
    broker_levels = stop_and_target(broker.state.entry_price,
                                    broker.state.entry_atr, risk_config)

    assert env_levels.stop == pytest.approx(broker_levels.stop)
    assert env_levels.target == pytest.approx(broker_levels.target)


def test_broker_min_hold_survives_restart(risk_config, tmp_path):
    """bars_held belongs to the position, not the process."""
    broker = PaperBroker(risk_config, str(tmp_path / 'account.json'))
    broker.buy(1000, '2026-01-01', atr=2.0)
    for _ in range(10):
        broker.advance_bar()
    broker.save()

    reloaded = PaperBroker(risk_config, str(tmp_path / 'account.json'))

    assert reloaded.state.bars_held == 10
    assert reloaded.act(SELL, 1000, '2026-01-02') is None      # 10 < 24


def test_broker_bars_held_only_counts_while_in_position(risk_config, tmp_path):
    broker = PaperBroker(risk_config, str(tmp_path / 'account.json'))
    for _ in range(5):
        broker.advance_bar()

    assert broker.state.bars_held == 0


def test_entry_atr_is_frozen_for_the_life_of_the_trade(enriched_ohlc, risk_config):
    """A later volatility spike must not widen an open position's stop."""
    env = BitcoinTradingEnv(enriched_ohlc, config=risk_config)
    env.reset(seed=1)
    env.step(BUY)

    frozen = env.entry_atr
    for _ in range(10):
        env.step(HOLD)
        if env.btc_held == 0:
            pytest.skip('position exited before the check')

    assert env.entry_atr == frozen
