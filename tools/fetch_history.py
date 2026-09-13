"""
Fetch OHLC history from Binance into a CSV the backtest can read.

The bundled data/BTC.csv is 4-hourly and ends in 2025. Any other interval - or a
fresher window - has to be pulled from the exchange:

    python tools/fetch_history.py --interval 15m --years 2 --out data/BTC_15m.csv

Output matches data/BTC.csv's shape: a timestamp index, OHLC columns and the real
Fear & Greed Index, oldest bar first.
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.feed import INTERVALS, attach_fear_greed, backfill_binance  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', default='15m', choices=sorted(INTERVALS))
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--years', type=float, help='How far back to fetch')
    parser.add_argument('--start', help='Explicit start date (overrides --years)')
    parser.add_argument('--out', default=None, help='Output CSV path')
    parser.add_argument('--keep-volume', action='store_true',
                        help='Keep the volume column (off by default so the feature '
                             'set matches the bundled dataset)')
    args = parser.parse_args()

    if args.start:
        start = pd.Timestamp(args.start)
    elif args.years:
        start = pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(
            days=int(round(365 * args.years)))
    else:
        start = None

    bars = backfill_binance(symbol=args.symbol, interval=args.interval, start=start)
    if not args.keep_volume:
        bars = bars.drop(columns=['volume'], errors='ignore')

    bars = attach_fear_greed(bars)

    out = args.out or f"data/{args.symbol.replace('USDT', '')}_{args.interval}.csv"
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    bars.to_csv(out)

    gaps = bars.index.to_series().diff().dropna()
    expected = pd.Timedelta(INTERVALS[args.interval]['pandas'])
    missing = int((gaps > expected).sum())

    print(f"\nWrote {len(bars):,} bars to {out}")
    print(f"  range   {bars.index[0]} -> {bars.index[-1]}")
    print(f"  gaps    {missing} (bars missing from the exchange's own history)")
    print(f"  columns {list(bars.columns)}")


if __name__ == "__main__":
    main()
