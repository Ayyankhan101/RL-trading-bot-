"""
Walk-forward backtest for the multi-sleeve carry book.

    python carry_backtest.py --data data/perp_funding.csv

Parameters are chosen on each fold's training window and scored on the untouched
test window. Leverage is not chosen by the search - it is solved from a **ruin
budget** against the empirical distribution of adverse moves across every symbol
traded, because the standard deviation of a carry return series does not contain
the jump that actually ends it.

The Deflated Sharpe Ratio corrects the published figure for how many parameter
sets were searched (Bailey & Lopez de Prado 2014).
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

from strategies.basis_carry import FUNDING_INTERVALS_PER_YEAR, weekly_statistics
from strategies.carry_book import (
    CarryBookConfig,
    run_carry_book,
    solve_per_symbol_leverage,
    to_panels,
)
from strategies.risk_budget import (
    INTERVALS_PER_YEAR,
    format_ruin_table,
    max_leverage,
    ruin_probability,
    ruin_table,
)
from utils.metrics import calculate_comprehensive_metrics, print_performance_report

RESULTS_DIR = 'results/carry_book'

GRID = {
    'max_sleeves': [3, 5, 8],
    'entry_threshold': [0.00001, 0.00002, 0.00005],
    'allow_negative_funding': [True, False],
}


def candidates() -> List[Dict[str, Any]]:
    keys = list(GRID)
    return [dict(zip(keys, values)) for values in itertools.product(*GRID.values())]


def folds(n_rows: int, n_folds: int) -> List[Tuple[int, int, int]]:
    size = n_rows // (n_folds + 1)
    out = []
    for i in range(n_folds):
        train_end = size * (i + 1)
        test_end = min(train_end + size, n_rows)
        if test_end - train_end < 10:
            break
        out.append((train_end, train_end, test_end))
    return out


def adverse_moves(data: pd.DataFrame) -> np.ndarray:
    """
    Pooled per-interval adverse moves across every symbol.

    Pooled deliberately: the book holds altcoins, which jump far harder than
    BTC, and sizing the book on BTC's tail would understate the risk of the
    sleeves that actually carry it.
    """
    perp = to_panels(data)['perp']
    returns = perp.pct_change().abs().to_numpy().flatten()
    return returns[np.isfinite(returns)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/perp_funding.csv')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--folds', type=int, default=4)
    parser.add_argument('--ruin-budget', type=float, default=0.01,
                        help='Acceptable probability of losing a sleeve in a year')
    parser.add_argument('--output-dir', default=RESULTS_DIR)
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)
    base = CarryBookConfig.from_dict(config)

    data = pd.read_csv(args.data, parse_dates=['timestamp'])
    panels = to_panels(data)
    index = panels['funding_rate'].index
    print(f"{data.symbol.nunique()} symbols, {len(index)} intervals, "
          f"{index[0]} -> {index[-1]}\n")

    # ---- leverage comes from a ruin budget, never from the parameter search
    moves = adverse_moves(data)
    pooled = max_leverage(moves, base.maintenance_margin,
                          budget=args.ruin_budget, horizon_intervals=INTERVALS_PER_YEAR)
    print(f"Ruin budget {args.ruin_budget:.1%}/year over {len(moves):,} pooled "
          f"adverse moves -> {pooled:.1f}x if the book were sized as one block")
    print(f"  worst observed 8h move across all symbols: {moves.max() * 100:.2f}%")
    print("  sizing each sleeve against its own tail instead, solved per fold "
          "on training data only\n")

    grid = candidates()
    fold_list = folds(len(index), args.folds)
    print(f"Walk-forward: {len(fold_list)} folds, {len(grid)} parameter sets per fold\n")

    def evaluate(window_index, params, leverages) -> Dict[str, Any]:
        window = data[data.timestamp.isin(window_index)]
        cfg = CarryBookConfig(**{**base.__dict__, **params})
        cfg.per_symbol_leverage = leverages
        result = run_carry_book(window, cfg)
        equity = result.equity
        total = equity.iloc[-1] / equity.iloc[0] - 1.0 if len(equity) > 1 else -1.0
        return {'params': params, 'total_return': float(total), 'result': result}

    segments, reports = [], []
    for i, (train_end, test_start, test_end) in enumerate(fold_list, start=1):
        train_index, test_index = index[:train_end], index[test_start:test_end]

        # Leverage is solved on the training window only. Fitting it to the full
        # sample would let each sleeve be sized by a crash it had not yet seen.
        train_window = data[data.timestamp.isin(train_index)]
        leverages = solve_per_symbol_leverage(
            train_window, base.maintenance_margin, args.ruin_budget,
            cap=base.max_leverage_cap)

        ranked = sorted((evaluate(train_index, p, leverages) for p in grid),
                        key=lambda r: r['total_return'], reverse=True)
        chosen = ranked[0]['params']

        oos = evaluate(test_index, chosen, leverages)
        segments.append(oos['result'].equity)
        reports.append({
            'fold': i,
            'test_start': str(test_index[0]), 'test_end': str(test_index[-1]),
            'chosen_params': chosen,
            'leverage': {k: round(v, 1) for k, v in leverages.items()},
            'total_return': oos['total_return'],
            'liquidations': oos['result'].liquidations,
            'funding_collected': oos['result'].funding_collected,
        })
        print(f"Fold {i}: {chosen} -> OOS {oos['total_return']:+.2%}  "
              f"liquidations {oos['result'].liquidations}")

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
        equity.tolist(), periods_per_year=FUNDING_INTERVALS_PER_YEAR,
        n_trials=len(grid) * len(fold_list))
    weekly = weekly_statistics(equity)

    print_performance_report(metrics, title="CARRY BOOK (out-of-sample, walk-forward)")

    print("\nWEEKLY DISTRIBUTION")
    print("-" * 44)
    print(f"{'Weeks':<24}{weekly['weeks']:>19}")
    print(f"{'Positive weeks':<24}{weekly['positive_weeks']:>10} "
          f"({weekly['positive_week_rate'] * 100:.1f}%)")
    print(f"{'Mean weekly return':<24}{weekly['mean_weekly_return'] * 100:>18.4f}%")
    print(f"{'Worst week':<24}{weekly['worst_week'] * 100:>18.4f}%")
    print("-" * 44)

    # Unlevered edge implied by the book, for the ruin table below.
    weekly_edge = weekly['mean_weekly_return'] / max(base.max_leverage_cap, 1e-9)
    print("\nWHAT EACH WEEKLY TARGET COSTS")
    print(format_ruin_table(ruin_table(moves, weekly_edge,
                                       maintenance_margin=base.maintenance_margin)))

    os.makedirs(args.output_dir, exist_ok=True)
    equity.to_csv(os.path.join(args.output_dir, 'equity_curve.csv'))
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as file:
        json.dump({'metrics': metrics, 'weekly': weekly, 'folds': reports},
                  file, indent=2, default=str)
    with open(os.path.join(args.output_dir, 'run_meta.json'), 'w') as file:
        json.dump({
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'data': args.data, 'symbols': int(data.symbol.nunique()),
            'intervals': int(len(index)),
            'data_start': str(index[0]), 'data_end': str(index[-1]),
            'n_trials': len(grid) * len(fold_list),
            'ruin_budget': args.ruin_budget,
            'pooled_leverage': pooled,
            'per_symbol_leverage': {k: round(v, 1) for k, v in
                                    solve_per_symbol_leverage(
                                        data, base.maintenance_margin,
                                        args.ruin_budget,
                                        cap=base.max_leverage_cap).items()},
            'ruin_at_pooled_leverage': ruin_probability(
                moves, pooled, base.maintenance_margin, INTERVALS_PER_YEAR),
            'worst_observed_move': float(moves.max()),
        }, file, indent=2)

    print(f"\nArtifacts written to {args.output_dir}/")


if __name__ == "__main__":
    main()
