"""Environment mechanics: costs, risk exits, accounting, and no lookahead."""

import copy

import numpy as np
import pandas as pd
import pytest

from environment.trading_env import BUY, HOLD, SELL, BitcoinTradingEnv


@pytest.fixture
def env_config(config):
    cfg = copy.deepcopy(config)
    cfg['risk'] = {}                       # risk exits enabled per-test
    cfg['backtesting']['slippage'] = 0.0
    # These tests pin immediate-fill mechanics, so they pin taker mode
    # explicitly. Maker execution is covered in tests/test_execution.py.
    cfg['execution'] = {'mode': 'taker', 'taker_fee': cfg['trading']['transaction_cost'],
                        'maker_fee': cfg['trading']['transaction_cost'], 'slippage': 0.0}
    # Regime scaling is exercised in tests/test_regime.py; these pin raw
    # execution mechanics and need full, unscaled exposure.
    cfg['regime'] = {'enabled': False}
    return cfg


def build_env(data, cfg):
    return BitcoinTradingEnv(data, config=cfg)


def test_reset_returns_observation_and_info(enriched_ohlc, env_config):
    env = build_env(enriched_ohlc, env_config)
    obs, info = env.reset(seed=1)

    assert obs.shape == (env.n_features,)
    assert np.isfinite(obs).all()
    assert info['portfolio_value'] == pytest.approx(env.initial_balance)


def test_constructor_does_not_require_reset_first(enriched_ohlc, env_config):
    """The old env called _get_observation() at step 0 and raised IndexError."""
    env = BitcoinTradingEnv(enriched_ohlc, config=env_config)
    assert env.n_features > 0


def test_episode_reaches_the_last_bar(enriched_ohlc, env_config):
    """max_steps was a count compared against an index, leaving bars untraded."""
    env = build_env(enriched_ohlc, env_config)
    env.reset(seed=1)

    while True:
        _, _, terminated, truncated, _ = env.step(HOLD)
        if terminated or truncated:
            break

    assert env.current_step == env.last_index


def test_buy_charges_fee_and_slippage(enriched_ohlc, env_config):
    env_config['trading']['transaction_cost'] = 0.001
    env_config['execution'].update({'taker_fee': 0.001, 'slippage': 0.002})
    env = build_env(enriched_ohlc, env_config)
    env.reset(seed=1)

    price = env._close[env.current_step]
    env.step(BUY)

    fill = env.trades[0]['price']
    assert fill == pytest.approx(price * 1.002)          # slippage against the agent
    assert env.trades[0]['fee'] > 0

    spent = env.initial_balance - env.balance
    notional = env.btc_held * fill
    assert spent == pytest.approx(notional * 1.001, rel=1e-9)


def test_sell_slippage_is_also_adverse(enriched_ohlc, env_config):
    env_config['execution']['slippage'] = 0.002
    env = build_env(enriched_ohlc, env_config)
    env.reset(seed=1)
    env.step(BUY)

    price_before_sell = env._close[env.current_step]
    env.step(SELL)

    sell_fill = [t for t in env.trades if t['action'] == 'SELL'][0]['price']
    assert sell_fill == pytest.approx(price_before_sell * 0.998)


def test_round_trip_with_zero_price_move_loses_the_fees(enriched_ohlc, env_config):
    """A flat round trip must be a loss, not a win."""
    flat = enriched_ohlc.copy()
    for col in ('open', 'high', 'low', 'close'):
        flat[col] = 20_000.0

    env_config['trading']['transaction_cost'] = 0.001
    env_config['execution']['taker_fee'] = 0.001
    env = build_env(flat, env_config)
    env.reset(seed=1)
    env.step(BUY)
    env.step(SELL)

    assert len(env.round_trips) == 1
    assert env.round_trips[0]['pnl'] < 0
    assert env.winning_trades == 0


def test_total_trades_counts_round_trips_not_orders(enriched_ohlc, env_config):
    """Counting both legs doubled the win-rate denominator."""
    env = build_env(enriched_ohlc, env_config)
    env.reset(seed=1)
    env.step(BUY)
    env.step(SELL)

    assert env.total_trades == 1
    assert env.open_orders == 1
    assert len([t for t in env.trades if t['action'] == 'BUY']) == 1


def test_sell_trade_records_the_amount_sold(enriched_ohlc, env_config):
    """The old env logged 'amount' after zeroing btc_held, so it was always 0."""
    env = build_env(enriched_ohlc, env_config)
    env.reset(seed=1)
    env.step(BUY)
    held = env.btc_held
    env.step(SELL)

    sell = [t for t in env.trades if t['action'] == 'SELL'][0]
    assert sell['amount'] == pytest.approx(held)
    assert sell['amount'] > 0


def test_repeated_buys_do_not_churn(enriched_ohlc, env_config):
    """
    The old env re-bought every bar with its leftover cash buffer, inflating the
    trade count with dust orders. min_position_size must stop that.
    """
    env = build_env(enriched_ohlc, env_config)
    env.reset(seed=1)
    for _ in range(30):
        env.step(BUY)

    assert env.open_orders <= 2


def test_exposure_respects_max_position_size(enriched_ohlc, env_config):
    env_config['trading']['max_position_size'] = 0.5
    env = build_env(enriched_ohlc, env_config)
    env.reset(seed=1)
    env.step(BUY)

    price = env._close[env.current_step]
    equity = env._portfolio_value(price)
    assert env.btc_held * price / equity <= 0.51


def test_stop_loss_fires_on_the_bar_that_breaches_it(enriched_ohlc, env_config):
    env_config['risk'] = {'stop_loss': 0.05}
    data = enriched_ohlc.copy().iloc[:10]
    data.loc[data.index[1], ['low', 'close']] = data['close'].iloc[0] * 0.90

    env = build_env(data, env_config)
    env.reset(seed=1)
    # The buy fills on bar 0; the stop is checked against bar 1 inside the same
    # step call, which is exactly the bar that breaches it.
    _, _, _, _, info = env.step(BUY)

    assert info.get('exit_reason') == 'stop_loss'
    assert env.btc_held == 0
    assert env.round_trips[0]['return_pct'] == pytest.approx(-0.05, abs=1e-9)


def test_take_profit_fires_on_the_bar_that_breaches_it(enriched_ohlc, env_config):
    env_config['risk'] = {'take_profit': 0.10}
    data = enriched_ohlc.copy().iloc[:10]
    data.loc[data.index[1], ['high', 'close']] = data['close'].iloc[0] * 1.20

    env = build_env(data, env_config)
    env.reset(seed=1)
    _, _, _, _, info = env.step(BUY)

    assert info.get('exit_reason') == 'take_profit'
    assert env.round_trips[0]['return_pct'] == pytest.approx(0.10, abs=1e-9)


def test_stop_wins_when_both_levels_sit_in_one_bar(enriched_ohlc, env_config):
    """Pessimistic ordering: ambiguity must not be resolved in the agent's favour."""
    env_config['risk'] = {'stop_loss': 0.05, 'take_profit': 0.05}
    data = enriched_ohlc.copy().iloc[:10]
    entry = data['close'].iloc[0]
    data.loc[data.index[1], 'high'] = entry * 1.20
    data.loc[data.index[1], 'low'] = entry * 0.80

    env = build_env(data, env_config)
    env.reset(seed=1)
    _, _, _, _, info = env.step(BUY)

    assert info.get('exit_reason') == 'stop_loss'


def test_drawdown_kill_switch_halts_trading(enriched_ohlc, env_config):
    env_config['risk'] = {'max_drawdown_threshold': 0.10}
    data = enriched_ohlc.copy().iloc[:10]
    data.loc[data.index[1]:, ['open', 'high', 'low', 'close']] *= 0.5

    env = build_env(data, env_config)
    env.reset(seed=1)
    _, _, _, _, info = env.step(BUY)

    assert info['halted']
    assert env.btc_held == 0

    # Halted means "sit in cash", not "stop the clock": the episode runs on so
    # the equity curve still covers every bar the benchmarks are scored over.
    trades_at_halt = len(env.trades)
    while True:
        _, _, terminated, truncated, _ = env.step(BUY)
        if terminated or truncated:
            break

    assert env.current_step == env.last_index
    assert len(env.trades) == trades_at_halt


def test_equity_curve_reconciles_with_round_trip_pnl(enriched_ohlc, env_config):
    """Realized P&L must explain the final equity exactly."""
    env = build_env(enriched_ohlc, env_config)
    env.reset(seed=1)

    rng = np.random.default_rng(3)
    while True:
        _, _, terminated, truncated, _ = env.step(int(rng.integers(0, 3)))
        if terminated or truncated:
            break

    realized = sum(t['pnl'] for t in env.round_trips)
    assert env.btc_held == 0                       # flattened at episode end
    assert env.portfolio_values[-1] == pytest.approx(env.initial_balance + realized, rel=1e-6)


def test_reward_sums_to_log_return_of_equity(enriched_ohlc, env_config):
    # Only the log-return reward is additive over the episode by construction;
    # the differential-Sharpe reward deliberately is not.
    env_config['reward'] = {'mode': 'log_return', 'scale': 100.0, 'clip': 10.0}
    env = build_env(enriched_ohlc, env_config)
    env.reset(seed=1)

    rng = np.random.default_rng(5)
    while True:
        _, _, terminated, truncated, _ = env.step(int(rng.integers(0, 3)))
        if terminated or truncated:
            break

    # The last step force-closes the position, which moves equity after the
    # reward for that step was computed; compare up to the penultimate value.
    expected = np.log(env.portfolio_values[-2] / env.portfolio_values[0]) * 100
    assert sum(env.rewards_history[:-1]) == pytest.approx(expected, rel=1e-6)


def test_observations_do_not_depend_on_future_bars(enriched_ohlc, env_config):
    """
    Replace every bar after step k with garbage; observations and rewards up to
    k must be bit-identical. This is the no-lookahead guarantee.
    """
    k = 60
    tampered = enriched_ohlc.copy()
    tampered.iloc[k + 1:] = tampered.iloc[k + 1:] * 3.0

    actions = [BUY, HOLD, HOLD, SELL, HOLD] * 20

    def run(data):
        env = build_env(data, env_config)
        obs, _ = env.reset(seed=11)
        observations, rewards = [obs], []
        for action in actions[:k]:
            obs, reward, terminated, truncated, _ = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            if terminated or truncated:
                break
        return np.array(observations), np.array(rewards)

    base_obs, base_rewards = run(enriched_ohlc)
    tampered_obs, tampered_rewards = run(tampered)

    np.testing.assert_array_equal(base_obs, tampered_obs)
    np.testing.assert_array_equal(base_rewards, tampered_rewards)


def test_rejects_unsorted_data(enriched_ohlc, env_config):
    with pytest.raises(ValueError):
        build_env(enriched_ohlc.iloc[::-1], env_config)


def test_rejects_invalid_action(enriched_ohlc, env_config):
    env = build_env(enriched_ohlc, env_config)
    env.reset(seed=1)
    with pytest.raises(ValueError):
        env.step(7)
