"""
Backtesting for the Bitcoin RL trading bot.

Two modes:

* ``--model PATH``   evaluate an existing checkpoint on a date range.
* ``--walk-forward`` (default) train and test across sequential folds, then
  concatenate the out-of-sample segments into one continuous equity curve.

Walk-forward is the headline protocol. A single train/test split on one asset is
one draw from a very noisy distribution; reporting it as *the* result is how a
backtest ends up flattering itself. Every fold trains only on bars that precede
its test window, with a purge gap between them.

Baselines (buy & hold, an RSI threshold rule) run through the same environment
with the same fees and slippage, so the comparison table is apples to apples.

Artifacts land in ``results/`` and are the single source of truth for the README
table and the dashboard - no figure is ever typed by hand.
"""

import argparse
import json
import os
import subprocess
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from agents.torch_dqn import DQNAgent
from environment.execution import execution_config, is_maker
from environment.trading_env import BUY, HOLD, SELL, BitcoinTradingEnv
from utils.data_utils import infer_periods_per_year, prepare_dataset
from utils.metrics import calculate_comprehensive_metrics, print_performance_report

RESULTS_DIR = 'results'


# --------------------------------------------------------------------- helpers

def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _sweep_trials(path: str = 'results/sweep.json') -> int:
    """How many configurations were tried before the published one was picked."""
    if not os.path.exists(path):
        return 1
    try:
        with open(path) as file:
            return int(json.load(file).get('n_trials', 1))
    except (ValueError, KeyError, OSError):
        return 1


def _risk_free_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Same fees and slippage, no stops or kill switch.

    Buy & hold with a 5% stop-loss is not buy & hold; the baselines have to be
    the strategies they claim to be.
    """
    baseline = yaml.safe_load(yaml.safe_dump(config))
    # Keep the minimum hold: it is an execution rule that the cost arithmetic
    # requires, not a risk overlay, and dropping it would let the rule-based
    # baselines churn at a horizon the edge cannot pay for.
    baseline['risk'] = {'min_hold_bars': config.get('risk', {}).get('min_hold_bars', 0)}
    # Baselines must be the strategies they claim to be. Buy & hold scaled by a
    # volatility regime is not buy & hold, and comparing against it would credit
    # the agent with an overlay the benchmark never had.
    baseline['regime'] = {'enabled': False}
    return baseline


def run_policy(env: BitcoinTradingEnv,
               policy: Callable[[int, np.ndarray], int]) -> Dict[str, Any]:
    """Drive an environment with an arbitrary ``(step, observation) -> action`` policy."""
    state, _ = env.reset()
    steps = 0

    while True:
        action = policy(env.current_step, state)
        state, _, terminated, truncated, _ = env.step(action)
        steps += 1
        if terminated or truncated:
            break

    return {'steps': steps}


def evaluate(env: BitcoinTradingEnv, policy: Callable[[int, np.ndarray], int],
             label: str) -> Dict[str, Any]:
    """Run a policy and return its equity curve, round trips and metrics."""
    run_policy(env, policy)
    periods_per_year = infer_periods_per_year(env.data.index)

    metrics = calculate_comprehensive_metrics(
        env.portfolio_values,
        trades=env.round_trips,
        periods_per_year=periods_per_year,
    )
    metrics['label'] = label

    return {
        'label': label,
        'metrics': metrics,
        'equity': env.equity_curve(),
        'trades': env.trades,
        'round_trips': env.round_trips,
    }


# ------------------------------------------------------------------- policies

def agent_policy(agent: DQNAgent) -> Callable[[int, np.ndarray], int]:
    """Greedy policy: no exploration at evaluation time."""
    return lambda step, obs: agent.act(obs, training=False)


def buy_and_hold_policy() -> Callable[[int, np.ndarray], int]:
    """Buy on the first bar, then never trade again."""
    state = {'entered': False}

    def policy(step: int, obs: np.ndarray) -> int:
        if not state['entered']:
            state['entered'] = True
            return BUY
        return HOLD

    return policy


def mean_reversion_policy(data: pd.DataFrame, signal: str = 'close_over_SMA_50',
                          low_quantile: float = 0.10,
                          high_quantile: float = 0.90) -> Callable[[int, np.ndarray], int]:
    """
    The edge the signal analysis actually found, expressed as a rule.

    `tools/signal_analysis.py` measured a stable mean-reversion signal: features
    with consistently negative information coefficients, meaning strength
    predicts weakness. This buys the bottom decile and exits at the top, with no
    model in between.

    It exists as a baseline because a learned policy that cannot beat the rule it
    was supposed to learn is not adding anything - and in the sweep the RL agent
    failed to capture the edge the rule captures.

    Quantiles are computed on an **expanding** window, so the decile boundary at
    bar t uses only bars up to t. Using whole-series quantiles - which is how the
    offline analysis computed them - would leak the future into every entry.
    """
    from features.observation import build_market_features

    names, matrix = build_market_features(data)
    if signal not in names:
        raise ValueError(f"{signal!r} not in the feature set")

    series = pd.Series(matrix[:, names.index(signal)], index=data.index)
    expanding = series.expanding(min_periods=500)
    low = expanding.quantile(low_quantile).to_numpy()
    high = expanding.quantile(high_quantile).to_numpy()
    values = series.to_numpy()

    def policy(step: int, obs: np.ndarray) -> int:
        value, floor, ceiling = values[step], low[step], high[step]
        if not (np.isfinite(value) and np.isfinite(floor) and np.isfinite(ceiling)):
            return HOLD
        if value <= floor:
            return BUY
        if value >= ceiling:
            return SELL
        return HOLD

    return policy


def rsi_policy(data: pd.DataFrame, oversold: float = 30.0,
               overbought: float = 70.0) -> Callable[[int, np.ndarray], int]:
    """Classic RSI threshold rule: buy oversold, sell overbought."""
    rsi = data['RSI'].to_numpy(dtype=np.float64)

    def policy(step: int, obs: np.ndarray) -> int:
        value = rsi[step]
        if not np.isfinite(value):
            return HOLD
        if value <= oversold:
            return BUY
        if value >= overbought:
            return SELL
        return HOLD

    return policy


# --------------------------------------------------------------- walk-forward

def walk_forward_folds(n_rows: int, n_folds: int, purge: int,
                       max_train_bars: Optional[int] = None) -> List[Tuple[int, int, int, int]]:
    """
    Sequential folds as ``(train_start, train_end, test_start, test_end)``.

    Train windows expand by default, so each fold uses all history available at
    that point in time - which is what a live system would have. ``max_train_bars``
    caps that into a sliding window: at 15m the final fold would otherwise train
    on ~58,000 bars, and the run stops being affordable while the extra history
    is mostly stale regime.
    """
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")

    test_size = n_rows // (n_folds + 1)
    if test_size <= purge + 2:
        raise ValueError(f"Not enough data for {n_folds} folds ({n_rows} rows)")

    folds = []
    for fold in range(n_folds):
        train_end = test_size * (fold + 1)
        test_start = train_end + purge
        test_end = min(test_start + test_size, n_rows)
        if test_end - test_start < 2:
            break

        train_start = 0
        if max_train_bars and train_end - train_start > max_train_bars:
            train_start = train_end - max_train_bars

        folds.append((train_start, train_end, test_start, test_end))

    return folds


def stitch_equity(segments: List[pd.DataFrame], initial_balance: float) -> pd.DataFrame:
    """
    Chain per-fold equity curves into one continuous compounded curve.

    Each fold restarts at the configured initial balance, so the segments are
    rebased onto the running equity by their period returns - not concatenated
    raw, which would reset the curve at every fold boundary.
    """
    running = initial_balance
    frames = []

    for segment in segments:
        values = segment['portfolio_value'].to_numpy(dtype=np.float64)
        if len(values) < 2 or values[0] <= 0:
            continue
        rebased = running * (values / values[0])
        frames.append(pd.DataFrame({'portfolio_value': rebased[1:]},
                                   index=segment.index[1:]))
        running = float(rebased[-1])

    if not frames:
        return pd.DataFrame({'portfolio_value': [initial_balance]})

    first_index = segments[0].index[0]
    head = pd.DataFrame({'portfolio_value': [initial_balance]}, index=[first_index])
    return pd.concat([head] + frames)


def run_walk_forward(data: pd.DataFrame, config: Dict[str, Any],
                     verbose: bool = True,
                     log_dir: Optional[str] = None) -> Dict[str, Any]:
    """Train and evaluate across sequential folds; report the stitched OOS result."""
    from train_agent import train  # local import: train_agent imports this module's siblings

    wf_config = config.get('backtesting', {}).get('walk_forward', {})
    n_folds = int(wf_config.get('n_folds', 5))
    purge = int(wf_config.get('purge_bars', 60))
    train_episodes = int(wf_config.get('train_episodes', 40))

    fold_config = yaml.safe_load(yaml.safe_dump(config))
    fold_config['training']['episodes'] = train_episodes

    max_train_bars = wf_config.get('max_train_bars')
    folds = walk_forward_folds(len(data), n_folds, purge, max_train_bars)
    if verbose:
        print(f"\nWalk-forward: {len(folds)} folds, purge {purge} bars, "
              f"{train_episodes} episodes per fold")

    segments: List[pd.DataFrame] = []
    all_round_trips: List[Dict[str, Any]] = []
    all_trades: List[Dict[str, Any]] = []
    fold_reports: List[Dict[str, Any]] = []

    for i, (train_start, train_end, test_start, test_end) in enumerate(folds, start=1):
        train_slice = data.iloc[train_start:train_end]
        test_slice = data.iloc[test_start:test_end]

        if verbose:
            print(f"\n--- Fold {i}/{len(folds)} ---")
            print(f"train {train_slice.index[0]} -> {train_slice.index[-1]} ({len(train_slice)} bars)")
            print(f"test  {test_slice.index[0]} -> {test_slice.index[-1]} ({len(test_slice)} bars)")

        fold_config['training']['seed'] = int(config['training'].get('seed', 42)) + i
        # Per-fold training logs make the in-sample/out-of-sample gap auditable:
        # an agent whose training equity explodes while its test fold loses money
        # is memorizing, and the two files sitting side by side show it.
        fold_log = os.path.join(log_dir, f'training_log_fold{i}.csv') if log_dir else None
        agent, _ = train(train_slice, fold_config, verbose=verbose, log_path=fold_log)

        test_env = BitcoinTradingEnv(test_slice, config=config)
        result = evaluate(test_env, agent_policy(agent), label=f'fold_{i}')

        segments.append(result['equity'])
        all_round_trips.extend(result['round_trips'])
        all_trades.extend(result['trades'])
        fold_reports.append({
            'fold': i,
            'train_start': str(train_slice.index[0]),
            'train_end': str(train_slice.index[-1]),
            'test_start': str(test_slice.index[0]),
            'test_end': str(test_slice.index[-1]),
            'total_return': result['metrics']['total_return'],
            'sharpe_ratio': result['metrics']['sharpe_ratio'],
            'max_drawdown': result['metrics']['max_drawdown'],
            'total_trades': result['metrics'].get('total_trades', 0),
            'win_rate': result['metrics'].get('win_rate', 0.0),
        })

        if verbose:
            print(f"Fold {i} OOS return: {result['metrics']['total_return']:.2%} "
                  f"| Sharpe {result['metrics']['sharpe_ratio']:.2f} "
                  f"| trades {result['metrics'].get('total_trades', 0)}")

    equity = stitch_equity(segments, float(config['trading']['initial_balance']))
    # n_trials deflates the Sharpe by how many configurations were searched
    # before this one was chosen (Bailey & Lopez de Prado 2014). Read from the
    # sweep artifact when present, so the published number carries its own
    # selection-bias correction.
    metrics = calculate_comprehensive_metrics(
        equity['portfolio_value'].tolist(),
        trades=all_round_trips,
        periods_per_year=infer_periods_per_year(data.index),
        n_trials=_sweep_trials(),
    )
    metrics['label'] = 'rl_agent_walk_forward'

    oos_start, oos_end = equity.index[0], equity.index[-1]

    return {
        'label': 'rl_agent_walk_forward',
        'metrics': metrics,
        'equity': equity,
        'trades': all_trades,
        'round_trips': all_round_trips,
        'folds': fold_reports,
        'oos_range': (oos_start, oos_end),
    }


# --------------------------------------------------------------- baselines

def run_baselines(data: pd.DataFrame, config: Dict[str, Any],
                  verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """Buy & hold and RSI, over the same bars, fees and slippage as the agent."""
    baseline_config = _risk_free_config(config)

    results = {}
    for label, policy_factory in (
        ('buy_and_hold', lambda: buy_and_hold_policy()),
        ('rsi_strategy', lambda: rsi_policy(data)),
        ('mean_reversion', lambda: mean_reversion_policy(data)),
    ):
        env = BitcoinTradingEnv(data, config=baseline_config)
        results[label] = evaluate(env, policy_factory(), label=label)
        if verbose:
            print(f"{label}: {results[label]['metrics']['total_return']:.2%} return")

    return results


# ----------------------------------------------------------------- artifacts

def write_artifacts(strategy: Dict[str, Any], baselines: Dict[str, Dict[str, Any]],
                    data: pd.DataFrame, config: Dict[str, Any],
                    folds: Optional[List[Dict[str, Any]]] = None,
                    output_dir: str = RESULTS_DIR,
                    full_data: Optional[pd.DataFrame] = None) -> None:
    """Write metrics, equity curves and trades. These files ARE the results."""
    os.makedirs(output_dir, exist_ok=True)

    equity = strategy['equity'].rename(columns={'portfolio_value': 'rl_agent'})
    for label, result in baselines.items():
        series = result['equity']['portfolio_value']
        # Baselines run over the full range; align them to the agent's OOS bars.
        equity[label] = series.reindex(equity.index).ffill()

    equity.index.name = 'timestamp'
    equity.to_csv(os.path.join(output_dir, 'equity_curve.csv'))

    round_trips = pd.DataFrame(strategy['round_trips'])
    round_trips.to_csv(os.path.join(output_dir, 'trades.csv'), index=False)

    metrics = {
        'rl_agent': strategy['metrics'],
        **{label: result['metrics'] for label, result in baselines.items()},
    }
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as file:
        json.dump(metrics, file, indent=2, default=str)

    # The dataset span and the out-of-sample span are different things: folds
    # consume the earliest bars for training, so reporting the trimmed window as
    # "the data" would understate what the run was built on.
    dataset = full_data if full_data is not None else data

    meta = {
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'git_sha': _git_sha(),
        'seed': config['training'].get('seed'),
        'variant': config['model'].get('variant'),
        'data_path': config.get('_data_path') or config.get('data', {}).get('path'),
        'data_rows': int(len(dataset)),
        'data_start': str(dataset.index[0]),
        'data_end': str(dataset.index[-1]),
        'oos_start': str(equity.index[0]),
        'oos_end': str(equity.index[-1]),
        'periods_per_year': infer_periods_per_year(data.index),
        'execution_mode': execution_config(config)['mode'],
        'fee_per_fill': (execution_config(config)['maker_fee'] if is_maker(config)
                         else execution_config(config)['taker_fee']),
        'slippage': (0.0 if is_maker(config)
                     else execution_config(config)['slippage']),
        'transaction_cost': config['trading']['transaction_cost'],
        'purge_bars': config['backtesting'].get('walk_forward', {}).get('purge_bars'),
        'train_episodes': config['backtesting'].get('walk_forward', {}).get('train_episodes'),
        'folds': folds or [],
    }
    with open(os.path.join(output_dir, 'run_meta.json'), 'w') as file:
        json.dump(meta, file, indent=2)

    with open(os.path.join(output_dir, 'config_snapshot.yaml'), 'w') as file:
        yaml.safe_dump(config, file, sort_keys=False)

    print(f"\nArtifacts written to {output_dir}/: "
          "metrics.json, equity_curve.csv, trades.csv, run_meta.json, config_snapshot.yaml")


def print_comparison(metrics_by_label: Dict[str, Dict[str, Any]]) -> None:
    """One table, one metrics implementation, every row measured the same way."""
    header = f"{'Strategy':<22}{'Return':>12}{'Sharpe':>9}{'Max DD':>10}{'Calmar':>9}{'Trades':>8}{'Win':>8}"
    print("\n" + header)
    print("-" * len(header))

    for label, metrics in metrics_by_label.items():
        pf = metrics.get('total_trades', 0)
        print(f"{label:<22}"
              f"{metrics['total_return'] * 100:>11.2f}%"
              f"{metrics['sharpe_ratio']:>9.2f}"
              f"{metrics['max_drawdown'] * 100:>9.2f}%"
              f"{metrics['calmar_ratio']:>9.2f}"
              f"{pf:>8d}"
              f"{metrics.get('win_rate', 0) * 100:>7.1f}%")
    print("-" * len(header))


# ---------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description='Backtest the Bitcoin RL trading bot')
    parser.add_argument('--data', type=str, default='data/BTC.csv')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--model', type=str, help='Evaluate this checkpoint instead of walk-forward')
    parser.add_argument('--walk-forward', action='store_true', default=False,
                        help='Walk-forward train/test across folds (default when --model is absent)')
    parser.add_argument('--folds', type=int, help='Override backtesting.walk_forward.n_folds')
    parser.add_argument('--episodes', type=int, help='Override walk_forward.train_episodes')
    parser.add_argument('--start-date', type=str)
    parser.add_argument('--end-date', type=str)
    parser.add_argument('--output-dir', type=str, default=RESULTS_DIR)
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    if args.folds:
        config['backtesting']['walk_forward']['n_folds'] = args.folds
    if args.episodes:
        config['backtesting']['walk_forward']['train_episodes'] = args.episodes

    # Record the dataset actually used, which the --data flag may override.
    config['_data_path'] = args.data

    data = prepare_dataset(args.data)
    if args.start_date:
        data = data[data.index >= pd.to_datetime(args.start_date)]
    if args.end_date:
        data = data[data.index <= pd.to_datetime(args.end_date)]
    if len(data) < 100:
        raise ValueError(f"Only {len(data)} bars after filtering - not enough to backtest")

    full_data = data
    folds = None
    if args.model:
        env = BitcoinTradingEnv(data, config=config)
        agent = DQNAgent.from_checkpoint(args.model)
        strategy = evaluate(env, agent_policy(agent), label='rl_agent')
        print("\nNote: a single-split evaluation of a pre-trained checkpoint. "
              "The headline numbers in README.md come from --walk-forward.")
    else:
        strategy = run_walk_forward(data, config, log_dir=args.output_dir)
        folds = strategy['folds']
        # Baselines must cover exactly the agent's out-of-sample span, or the
        # comparison silently credits the agent with bars it never traded.
        oos_start, oos_end = strategy['oos_range']
        data = data[(data.index >= oos_start) & (data.index <= oos_end)]

    baselines = run_baselines(data, config)

    print_performance_report(strategy['metrics'], title="RL AGENT (out-of-sample)")
    print_comparison({
        'RL Agent': strategy['metrics'],
        'Buy & Hold': baselines['buy_and_hold']['metrics'],
        'RSI Strategy': baselines['rsi_strategy']['metrics'],
        'Mean Reversion': baselines['mean_reversion']['metrics'],
    })

    write_artifacts(strategy, baselines, data, config, folds, args.output_dir,
                    full_data=full_data)


if __name__ == "__main__":
    main()
