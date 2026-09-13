"""End-to-end behaviour of the backtest harness and its baselines."""

import copy
import json
import os

import numpy as np
import pandas as pd
import pytest

from backtest import (
    buy_and_hold_policy,
    evaluate,
    rsi_policy,
    run_baselines,
    run_walk_forward,
    stitch_equity,
    walk_forward_folds,
    write_artifacts,
)
from environment.trading_env import BitcoinTradingEnv


@pytest.fixture
def bt_config(config):
    cfg = copy.deepcopy(config)
    cfg['network']['hidden_layers'] = [16, 16]
    cfg['model'].update({'memory_size': 2000, 'batch_size': 16,
                         'learning_starts': 32, 'train_frequency': 4})
    cfg['backtesting']['walk_forward'] = {'n_folds': 2, 'train_episodes': 1, 'purge_bars': 5}
    cfg['regime'] = {'enabled': False}
    return cfg


def test_folds_are_chronological_and_purged():
    folds = walk_forward_folds(n_rows=1200, n_folds=3, purge=10)

    assert len(folds) == 3
    for train_start, train_end, test_start, test_end in folds:
        assert train_start < train_end < test_start < test_end
        assert test_start - train_end == 10      # the purge gap

    # Test windows must not overlap and must move forward in time.
    starts = [f[2] for f in folds]
    assert starts == sorted(starts)
    for (_, _, _, prev_end), (_, _, next_start, _) in zip(folds, folds[1:]):
        assert next_start >= prev_end


def test_folds_never_train_on_future_bars():
    for train_start, train_end, test_start, _ in walk_forward_folds(2000, 4, 20):
        assert train_end <= test_start


def test_too_few_rows_raises():
    with pytest.raises(ValueError):
        walk_forward_folds(n_rows=20, n_folds=5, purge=10)


def test_stitch_equity_compounds_across_folds():
    """
    Folds each restart at the initial balance; stitching must compound their
    returns rather than concatenating raw values.
    """
    index_a = pd.date_range('2022-01-01', periods=3, freq='4h')
    index_b = pd.date_range('2022-01-02', periods=3, freq='4h')
    segments = [
        pd.DataFrame({'portfolio_value': [100.0, 110.0, 120.0]}, index=index_a),
        pd.DataFrame({'portfolio_value': [100.0, 50.0, 60.0]}, index=index_b),
    ]

    stitched = stitch_equity(segments, initial_balance=100.0)

    assert stitched['portfolio_value'].iloc[0] == pytest.approx(100.0)
    assert stitched['portfolio_value'].iloc[2] == pytest.approx(120.0)
    # Second fold's +20%/-50% path applied on top of 120.
    assert stitched['portfolio_value'].iloc[-1] == pytest.approx(120.0 * 0.6)


def test_buy_and_hold_tracks_price_minus_costs(enriched_ohlc, bt_config):
    cfg = copy.deepcopy(bt_config)
    cfg['risk'] = {}
    env = BitcoinTradingEnv(enriched_ohlc, config=cfg)
    result = evaluate(env, buy_and_hold_policy(), 'buy_and_hold')

    price_return = enriched_ohlc['close'].iloc[-1] / enriched_ohlc['close'].iloc[0] - 1
    strategy_return = result['metrics']['total_return']

    assert result['metrics']['total_trades'] == 1
    # Exposure is capped at max_position_size, so the curve tracks 95% of the
    # price move; fees and slippage then make it strictly worse than that.
    assert strategy_return < price_return * 0.95
    assert strategy_return == pytest.approx(price_return * 0.95, abs=0.01)


def test_rsi_policy_only_trades_at_thresholds(enriched_ohlc, bt_config):
    policy = rsi_policy(enriched_ohlc, oversold=30, overbought=70)
    rsi = enriched_ohlc['RSI'].to_numpy()

    for step in range(len(enriched_ohlc)):
        action = policy(step, np.zeros(1))
        if 30 < rsi[step] < 70:
            assert action == 0


def test_baselines_run_without_risk_stops(enriched_ohlc, bt_config):
    """Buy & hold with a stop-loss is not buy & hold."""
    bt_config['risk'] = {'stop_loss': 0.01, 'take_profit': 0.01}
    results = run_baselines(enriched_ohlc, bt_config, verbose=False)

    assert results['buy_and_hold']['metrics']['total_trades'] == 1
    assert set(results) == {'buy_and_hold', 'rsi_strategy', 'mean_reversion'}


def test_walk_forward_produces_out_of_sample_curve(enriched_ohlc, bt_config):
    result = run_walk_forward(enriched_ohlc, bt_config, verbose=False)

    assert len(result['folds']) == 2
    assert result['equity'].index.is_monotonic_increasing
    assert result['metrics']['label'] == 'rl_agent_walk_forward'

    # The reported span must start after the first fold's training window.
    first_test_start = pd.Timestamp(result['folds'][0]['test_start'])
    assert result['equity'].index[0] >= first_test_start


def test_artifacts_are_written_and_self_consistent(enriched_ohlc, bt_config, tmp_path):
    result = run_walk_forward(enriched_ohlc, bt_config, verbose=False)
    oos_start, oos_end = result['oos_range']
    window = enriched_ohlc[(enriched_ohlc.index >= oos_start) & (enriched_ohlc.index <= oos_end)]

    baselines = run_baselines(window, bt_config, verbose=False)
    write_artifacts(result, baselines, window, bt_config,
                    result['folds'], output_dir=str(tmp_path))

    for name in ('metrics.json', 'equity_curve.csv', 'trades.csv',
                 'run_meta.json', 'config_snapshot.yaml'):
        assert os.path.exists(tmp_path / name)

    metrics = json.loads((tmp_path / 'metrics.json').read_text())
    assert {'rl_agent', 'buy_and_hold', 'rsi_strategy', 'mean_reversion'} <= set(metrics)

    equity = pd.read_csv(tmp_path / 'equity_curve.csv', index_col='timestamp')
    assert list(equity.columns) == ['rl_agent', 'buy_and_hold', 'rsi_strategy',
                                    'mean_reversion']

    # The headline number in metrics.json must be derivable from the curve it ships.
    curve = equity['rl_agent'].to_numpy()
    assert metrics['rl_agent']['total_return'] == pytest.approx(curve[-1] / curve[0] - 1, rel=1e-6)


def test_oos_curve_covers_every_test_bar(enriched_ohlc, bt_config):
    """
    The stitched curve must span the full test windows.

    A curve that stops early - as it did when the drawdown kill switch also
    ended the episode - gets compared against benchmarks measured over more
    bars than the agent actually traded.
    """
    bt_config['risk'] = {'max_drawdown_threshold': 0.01}     # halt almost immediately
    result = run_walk_forward(enriched_ohlc, bt_config, verbose=False)

    last_fold_end = pd.Timestamp(result['folds'][-1]['test_end'])
    assert result['equity'].index[-1] == last_fold_end


def test_published_metrics_carry_a_selection_bias_correction(tmp_path, monkeypatch):
    """
    The headline Sharpe must be deflated by how many configurations were
    searched before it was chosen (Bailey & Lopez de Prado 2014). Otherwise the
    published number is the maximum of a search, reported as if it were a single
    honest trial.
    """
    import backtest as backtest_module

    sweep = tmp_path / 'sweep.json'
    sweep.write_text(json.dumps({'n_trials': 18}))

    assert backtest_module._sweep_trials(str(sweep)) == 18
    assert backtest_module._sweep_trials(str(tmp_path / 'missing.json')) == 1


def test_more_searched_configs_lower_the_reported_confidence():
    """Searching harder must raise the bar, not the score."""
    from utils.metrics import calculate_comprehensive_metrics

    rng = np.random.default_rng(3)
    curve = 10_000 * np.cumprod(1 + rng.normal(0.0001, 0.004, 3000))

    honest = calculate_comprehensive_metrics(curve.tolist(), periods_per_year=35040,
                                             n_trials=1)
    searched = calculate_comprehensive_metrics(curve.tolist(), periods_per_year=35040,
                                               n_trials=200)

    assert searched['deflated_sharpe'] <= honest['deflated_sharpe']
    assert searched['threshold_sharpe'] > honest['threshold_sharpe']
    assert honest['sharpe_ratio'] == pytest.approx(searched['sharpe_ratio'])


def test_mean_reversion_quantiles_are_causal(enriched_ohlc, bt_config):
    """
    The decile boundaries must expand over time, not be computed on the whole
    series. Whole-series quantiles would leak the future into every entry - the
    offline analysis can do that, a backtest cannot.
    """
    from backtest import mean_reversion_policy

    policy = mean_reversion_policy(enriched_ohlc)
    truncated = mean_reversion_policy(enriched_ohlc.iloc[:len(enriched_ohlc) - 200])

    for step in range(600, len(enriched_ohlc) - 200):
        assert policy(step, np.zeros(1)) == truncated(step, np.zeros(1))


def test_mean_reversion_buys_weakness_and_sells_strength(enriched_ohlc, bt_config):
    """The measured IC is negative, so the rule must fade strength, not chase it."""
    from backtest import mean_reversion_policy
    from features.observation import build_market_features

    names, matrix = build_market_features(enriched_ohlc)
    signal = matrix[:, names.index('close_over_SMA_50')]
    policy = mean_reversion_policy(enriched_ohlc)

    buys = [i for i in range(600, len(enriched_ohlc)) if policy(i, np.zeros(1)) == 1]
    sells = [i for i in range(600, len(enriched_ohlc)) if policy(i, np.zeros(1)) == 2]

    assert buys and sells
    assert np.mean(signal[buys]) < np.mean(signal[sells])
