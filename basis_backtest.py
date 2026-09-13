"""
Walk-forward backtest for the funding-rate carry strategy.

    python basis_backtest.py --data data/BTC_basis.csv

Parameters are chosen on each fold's training window and scored on the untouched
test window that follows it, so the reported number is never the number that was
optimized. The Deflated Sharpe Ratio corrects for how many parameter sets were
searched (Bailey & Lopez de Prado 2014), exactly as in the price-based backtest.

Artifacts land in ``results/basis/``.
"""

import argparse
import itertools
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

from strategies.basis_carry import (
    FUNDING_INTERVALS_PER_YEAR,
    BasisConfig,
    liquidation_move,
    run_basis_carry,
    weekly_statistics,
)
from utils.metrics import calculate_comprehensive_metrics, print_performance_report

RESULTS_DIR = 'results/basis'

GRID = {
    'leverage': [1.0, 2.0, 3.0, 5.0],
    'entry_threshold': [0.0, 0.00002, 0.00005],
}


def candidates() -> List[Dict[str, Any]]:
    keys = list(GRID)
    return [dict(zip(keys, values)) for values in itertools.product(*GRID.values())]


def folds(n_rows: int, n_folds: int) -> List[Tuple[int, int, int, int]]:
    """Sequential expanding-train / fixed-test folds over the funding series."""
    size = n_rows // (n_folds + 1)
    out = []
    for i in range(n_folds):
        train_end = size * (i + 1)
        test_end = min(train_end + size, n_rows)
        if test_end - train_end < 10:
            break
        out.append((0, train_end, train_end, test_end))
    return out


def score(data: pd.DataFrame, params: Dict[str, Any], base: BasisConfig) -> Dict[str, Any]:
    config = BasisConfig(**{**base.__dict__, **params})
    result = run_basis_carry(data, config)

    equity = result.equity
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0 if len(equity) else -1.0

    return {
        'params': params,
        'total_return': float(total_return),
        'final_equity': float(equity.iloc[-1]) if len(equity) else 0.0,
        'liquidations': result.liquidations,
        'funding_collected': result.funding_collected,
        'basis_pnl': result.basis_pnl,
        'costs_paid': result.costs_paid,
        'result': result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/BTC_basis.csv')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument('--output-dir', default=RESULTS_DIR)
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)
    base = BasisConfig.from_dict(config)

    data = pd.read_csv(args.data, parse_dates=['timestamp'], index_col='timestamp')
    print(f"Loaded {len(data)} funding intervals, "
          f"{data.index[0]} -> {data.index[-1]}\n")

    grid = candidates()
    fold_list = folds(len(data), args.folds)
    print(f"Walk-forward: {len(fold_list)} folds, {len(grid)} parameter sets per fold\n")

    segments: List[pd.Series] = []
    reports: List[Dict[str, Any]] = []

    for i, (_, train_end, test_start, test_end) in enumerate(fold_list, start=1):
        train = data.iloc[:train_end]
        test = data.iloc[test_start:test_end]

        # Pick on training, then never look at that choice again.
        ranked = sorted((score(train, p, base) for p in grid),
                        key=lambda r: r['total_return'], reverse=True)
        chosen = ranked[0]['params']

        out_of_sample = score(test, chosen, base)
        segments.append(out_of_sample['result'].equity)

        reports.append({
            'fold': i,
            'train_end': str(train.index[-1]),
            'test_start': str(test.index[0]),
            'test_end': str(test.index[-1]),
            'chosen_params': chosen,
            'total_return': out_of_sample['total_return'],
            'liquidations': out_of_sample['liquidations'],
            'funding_collected': out_of_sample['funding_collected'],
        })

        print(f"Fold {i}: chose {chosen} -> OOS return "
              f"{out_of_sample['total_return']:+.2%}  "
              f"liquidations {out_of_sample['liquidations']}")

    # Chain the out-of-sample segments into one compounded curve.
    running = base.initial_capital
    frames = []
    for segment in segments:
        values = segment.to_numpy(dtype=np.float64)
        if len(values) < 2 or values[0] <= 0:
            continue
        rebased = running * (values / values[0])
        frames.append(pd.Series(rebased[1:], index=segment.index[1:]))
        running = float(rebased[-1])

    equity = pd.concat(frames) if frames else pd.Series([base.initial_capital])
    equity.name = 'equity'

    metrics = calculate_comprehensive_metrics(
        equity.tolist(),
        periods_per_year=FUNDING_INTERVALS_PER_YEAR,
        n_trials=len(grid) * len(fold_list),
    )
    weekly = weekly_statistics(equity)

    print_performance_report(metrics, title="FUNDING CARRY (out-of-sample, walk-forward)")

    print("\nWEEKLY DISTRIBUTION")
    print("-" * 44)
    print(f"{'Weeks':<22}{weekly['weeks']:>21}")
    print(f"{'Positive weeks':<22}{weekly['positive_weeks']:>12} "
          f"({weekly['positive_week_rate'] * 100:.1f}%)")
    print(f"{'Mean weekly return':<22}{weekly['mean_weekly_return'] * 100:>20.4f}%")
    print(f"{'Median weekly return':<22}{weekly['median_weekly_return'] * 100:>20.4f}%")
    print(f"{'Worst week':<22}{weekly['worst_week'] * 100:>20.4f}%")
    print(f"{'Best week':<22}{weekly['best_week'] * 100:>20.4f}%")
    print("-" * 44)

    print(f"\nLiquidation distance at each leverage "
          f"(maintenance {base.maintenance_margin:.1%}):")
    for lev in GRID['leverage'] + [10.0, 107.0]:
        print(f"  {lev:6.0f}x  liquidated by a "
              f"{liquidation_move(lev, base.maintenance_margin) * 100:5.2f}% adverse move")

    os.makedirs(args.output_dir, exist_ok=True)
    equity.to_csv(os.path.join(args.output_dir, 'equity_curve.csv'))
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as file:
        json.dump({'metrics': metrics, 'weekly': weekly, 'folds': reports},
                  file, indent=2, default=str)
    with open(os.path.join(args.output_dir, 'run_meta.json'), 'w') as file:
        json.dump({
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'data': args.data,
            'intervals': int(len(data)),
            'data_start': str(data.index[0]),
            'data_end': str(data.index[-1]),
            'n_trials': len(grid) * len(fold_list),
            'fee_per_leg': base.fee_per_leg,
            'maintenance_margin': base.maintenance_margin,
        }, file, indent=2)

    print(f"\nArtifacts written to {args.output_dir}/")


if __name__ == "__main__":
    main()
