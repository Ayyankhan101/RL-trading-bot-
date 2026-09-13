"""
Is there an edge, and is it bigger than the cost of trading it?

Two measurements, run before any model is fit, because a model cannot extract
information that is not in the features:

**Information coefficient** - the Spearman correlation between each feature and
the forward return. Standard practice in quantitative equity research (see
Grinold & Kahn, *Active Portfolio Management*, where IC ties directly to the
information ratio). |IC| around 0.03 is considered usable; below 0.01 is noise.
Reported per horizon, and split-half, because an IC that flips sign between the
two halves of the data is not a signal.

**Gross edge vs cost** - for each candidate signal, enter on decile extremes,
exit after a fixed horizon, and compare the average gross return per trade with
what the round trip actually costs. An edge smaller than its cost is untradeable
no matter how good the model is.

    python tools/signal_analysis.py --data data/BTC_15m.csv
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.observation import build_market_features  # noqa: E402
from utils.data_utils import infer_periods_per_year, prepare_dataset  # noqa: E402


def information_coefficients(features: pd.DataFrame, close: pd.Series,
                             horizons: List[int]) -> Dict[str, Any]:
    out = {}
    for horizon in horizons:
        forward = close.pct_change(horizon).shift(-horizon)
        ics = {col: float(features[col].corr(forward, method='spearman'))
               for col in features.columns}
        finite = {k: v for k, v in ics.items() if np.isfinite(v)}
        out[str(horizon)] = {
            'mean_abs_ic': float(np.mean([abs(v) for v in finite.values()])),
            'ics': finite,
            'top': sorted(finite.items(), key=lambda kv: -abs(kv[1]))[:5],
        }
    return out


def split_half_stability(features: pd.DataFrame, close: pd.Series,
                         horizon: int, threshold: float = 0.02) -> Dict[str, Any]:
    """
    A feature whose IC flips sign between the first and second half of the data
    is not a signal, however large it looks overall.
    """
    half = len(features) // 2
    forward = close.pct_change(horizon).shift(-horizon)

    stable = {}
    for col in features.columns:
        first = features[col].iloc[:half].corr(forward.iloc[:half], method='spearman')
        second = features[col].iloc[half:].corr(forward.iloc[half:], method='spearman')
        if not (np.isfinite(first) and np.isfinite(second)):
            continue
        if np.sign(first) == np.sign(second) and min(abs(first), abs(second)) > threshold:
            stable[col] = {'first_half': float(first), 'second_half': float(second)}

    return {
        'horizon': horizon,
        'threshold': threshold,
        'n_stable': len(stable),
        'n_features': int(features.shape[1]),
        'stable': stable,
    }


def edge_versus_cost(features: pd.DataFrame, close: pd.Series, signals: List[str],
                     horizons: List[int], taker_cost: float, maker_cost: float,
                     periods_per_year: float) -> List[Dict[str, Any]]:
    """
    Decile-extreme entry, fixed-horizon exit. The simplest possible expression of
    the signal, so the number is not a property of any model.
    """
    rows = []
    for name in signals:
        if name not in features.columns:
            continue
        z = features[name]
        low, high = z.quantile(0.10), z.quantile(0.90)

        for horizon in horizons:
            forward = close.pct_change(horizon).shift(-horizon)
            # Negative IC means mean reversion: buy the low decile, sell the high.
            longs = forward[z <= low].dropna()
            shorts = -forward[z >= high].dropna()
            combined = pd.concat([longs, shorts])
            if combined.empty:
                continue

            gross = float(combined.mean())
            trades_per_year = periods_per_year / horizon

            rows.append({
                'signal': name,
                'horizon_bars': horizon,
                'n_observations': int(len(combined)),
                'gross_per_trade': gross,
                'long_gross': float(longs.mean()) if len(longs) else None,
                'short_gross': float(shorts.mean()) if len(shorts) else None,
                'net_taker_per_trade': gross - taker_cost,
                'net_maker_per_trade': gross - maker_cost,
                'annualized_net_maker': (gross - maker_cost) * trades_per_year,
                'tradeable_taker': bool(gross > taker_cost),
                'tradeable_maker': bool(gross > maker_cost),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/BTC_15m.csv')
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--out', default='results/signal_analysis.json')
    parser.add_argument('--horizons', type=int, nargs='+', default=[4, 24, 96, 192])
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    execution = config.get('execution', {})

    if str(execution.get('mode', 'taker')).lower() == 'cfd':
        # A CFD round trip is the spread (a price offset) plus per-lot
        # commission, expressed against notional - not a percentage fee.
        import pandas as _pd
        reference = float(_pd.read_csv(args.data, nrows=200)['close'].median())
        spread = float(execution.get('spread_points', 0.15))
        commission = float(execution.get('commission_per_lot', 7.0))
        contract = float(execution.get('contract_size', 100.0))
        taker_cost = maker_cost = (spread + commission / contract) / reference
    else:
        taker_cost = 2 * (float(execution.get('taker_fee', 0.001))
                          + float(execution.get('slippage', 0.0005)))
        maker_cost = 2 * float(execution.get('maker_fee', 0.0002))

    data = prepare_dataset(args.data)
    periods_per_year = infer_periods_per_year(data.index)
    names, matrix = build_market_features(data)
    features = pd.DataFrame(matrix, columns=names, index=data.index)

    ic = information_coefficients(features, data['close'], args.horizons)
    stability = split_half_stability(features, data['close'], horizon=24)
    signals = [name for name, _ in ic[str(args.horizons[1])]['top']]
    edges = edge_versus_cost(features, data['close'], signals, args.horizons,
                             taker_cost, maker_cost, periods_per_year)

    print(f"\nRound-trip cost: taker {taker_cost * 100:.3f}%  maker {maker_cost * 100:.3f}%\n")
    print("Information coefficient (Spearman feature vs forward return)")
    for horizon in args.horizons:
        entry = ic[str(horizon)]
        top = ', '.join(f"{k}={v:+.4f}" for k, v in entry['top'][:3])
        print(f"  {horizon:4d} bars  mean|IC| {entry['mean_abs_ic']:.4f}   {top}")

    print(f"\nSign-stable features across both halves (h=24, |IC|>0.02): "
          f"{stability['n_stable']}/{stability['n_features']}")

    print("\nGross edge vs cost (decile entry, fixed-horizon exit)")
    print(f"  {'signal':22s} {'bars':>5s} {'gross':>9s} {'net taker':>10s} "
          f"{'net maker':>10s} {'ann maker':>10s}")
    for row in edges:
        print(f"  {row['signal']:22s} {row['horizon_bars']:5d} "
              f"{row['gross_per_trade'] * 100:+8.4f}% "
              f"{row['net_taker_per_trade'] * 100:+9.4f}% "
              f"{row['net_maker_per_trade'] * 100:+9.4f}% "
              f"{row['annualized_net_maker'] * 100:+9.1f}%")

    tradeable = [r for r in edges if r['tradeable_maker']]
    print(f"\n{len(tradeable)}/{len(edges)} signal/horizon pairs clear even maker costs.")
    if not tradeable:
        print("=> No configuration of this feature set is tradeable. "
              "The edge is smaller than the cost of capturing it.")

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as file:
        json.dump({
            'data': args.data,
            'n_bars': int(len(data)),
            'periods_per_year': periods_per_year,
            'taker_round_trip': taker_cost,
            'maker_round_trip': maker_cost,
            'information_coefficients': ic,
            'split_half_stability': stability,
            'edge_versus_cost': edges,
        }, file, indent=2, default=str)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
