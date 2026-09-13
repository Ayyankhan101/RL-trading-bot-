"""
Configuration sweep on a validation split.

Searching over configurations is how backtests get overfit: try enough reward
functions and stop widths and one will look good by luck. Two guards here:

1. Every candidate is trained and scored on a **validation** window only. The
   walk-forward test folds are never touched, so the final number is not the
   number that was optimized.
2. The count of candidates is recorded, so the winner's Sharpe can be deflated
   by it - Bailey & Lopez de Prado (2014). A Sharpe that does not beat
   ``expected_max_sharpe(n_trials)`` is indistinguishable from luck.

    python tools/sweep.py --data data/BTC_15m.csv --episodes 4
"""

import argparse
import copy
import itertools
import json
import os
import sys
import time
from typing import Any, Dict, List

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.learner import evaluate_agent  # noqa: E402
from utils.data_utils import infer_periods_per_year, prepare_dataset  # noqa: E402
from utils.metrics import deflated_sharpe_ratio, expected_max_sharpe  # noqa: E402

# Each entry is a dotted config path and the values to try.
# The signal analysis settled the reward and the stop family; what is open now is
# execution mode and holding horizon, which is where the edge-versus-cost
# arithmetic actually lives.
GRID: Dict[str, List[Any]] = {
    'execution.mode': ['maker', 'taker'],
    'risk.min_hold_bars': [24, 96, 192],
    'risk.stop_loss_atr': [10.0, 16.0],
}


def set_path(config: Dict[str, Any], path: str, value: Any) -> None:
    section, key = path.split('.')
    config.setdefault(section, {})[key] = value


def candidate_configs(base: Dict[str, Any], grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid)
    out = []
    for combination in itertools.product(*(grid[k] for k in keys)):
        config = copy.deepcopy(base)
        label = {}
        for key, value in zip(keys, combination):
            set_path(config, key, value)
            # The target moves with the stop so the payoff ratio stays fixed and
            # the sweep measures stop width, not a changed risk/reward.
            if key == 'risk.stop_loss_atr':
                set_path(config, 'risk.take_profit_atr', float(value) * 2.0)
            if key == 'risk.min_hold_bars':
                # n-step should reach roughly half the hold, or a delayed payoff
                # never reaches the decision that caused it.
                set_path(config, 'model.n_step', max(4, int(value) // 2))
            label[key] = value
        config['_label'] = label
        out.append(config)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/BTC_15m.csv')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--episodes', type=int, default=4)
    parser.add_argument('--train-bars', type=int, default=8000)
    parser.add_argument('--val-bars', type=int, default=4000)
    parser.add_argument('--out', default='results/sweep.json')
    args = parser.parse_args()

    with open(args.config) as file:
        base = yaml.safe_load(file)

    data = prepare_dataset(args.data)
    periods_per_year = infer_periods_per_year(data.index)

    # Validation sits strictly before the walk-forward test region so tuning
    # cannot see the bars the final result is measured on.
    reserve = int(len(data) * 0.45)
    window = data.iloc[:len(data) - reserve]
    train_slice = window.iloc[-(args.train_bars + args.val_bars):-args.val_bars]
    val_slice = window.iloc[-args.val_bars:]

    candidates = candidate_configs(base, GRID)
    print(f"Sweeping {len(candidates)} configurations")
    print(f"  train {train_slice.index[0]} -> {train_slice.index[-1]}")
    print(f"  valid {val_slice.index[0]} -> {val_slice.index[-1]}\n")

    from train_agent import train

    results = []
    for i, config in enumerate(candidates, start=1):
        label = config.pop('_label')
        config['training']['episodes'] = args.episodes
        config['training']['seed'] = 42

        started = time.time()
        agent, _ = train(train_slice, config, verbose=False)
        metrics = evaluate_agent(agent, val_slice, config)

        row = {
            'index': i,
            'label': label,
            'total_return': metrics.get('total_return', 0.0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0.0),
            'max_drawdown': metrics.get('max_drawdown', 0.0),
            'total_trades': metrics.get('total_trades', 0),
            'win_rate': metrics.get('win_rate', 0.0),
            'profit_factor': metrics.get('profit_factor'),
            'seconds': round(time.time() - started, 1),
        }
        results.append(row)

        print(f"[{i}/{len(candidates)}] {label} -> "
              f"return {row['total_return']:+7.2%}  Sharpe {row['sharpe_ratio']:6.2f}  "
              f"trades {row['total_trades']:4d}  ({row['seconds']:.0f}s)")

    results.sort(key=lambda r: r['sharpe_ratio'], reverse=True)
    best = results[0]

    n_trials = len(candidates)
    threshold = expected_max_sharpe(n_trials, sharpe_variance=1.0 / max(args.val_bars - 1, 1))
    threshold_annual = threshold * (periods_per_year ** 0.5)

    print(f"\nBest: {best['label']}  Sharpe {best['sharpe_ratio']:.2f}")
    print(f"Luck threshold for {n_trials} trials: annualized Sharpe "
          f"{threshold_annual:.2f}")
    if best['sharpe_ratio'] <= threshold_annual:
        print("=> The winner does NOT clear what searching this many configs "
              "produces by chance. Treat it as noise.")
    else:
        print("=> The winner clears the selection-bias threshold on validation.")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as file:
        json.dump({
            'n_trials': n_trials,
            'threshold_annual_sharpe': threshold_annual,
            'train_window': [str(train_slice.index[0]), str(train_slice.index[-1])],
            'val_window': [str(val_slice.index[0]), str(val_slice.index[-1])],
            'results': results,
        }, file, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
