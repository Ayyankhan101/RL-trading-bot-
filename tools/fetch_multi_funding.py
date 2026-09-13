"""
Build the multi-asset carry dataset: funding, spot and perp for N perpetuals.

    python tools/fetch_multi_funding.py --out data/perp_funding.csv

Long format: one row per (timestamp, symbol). Both legs are priced at the bar
**open** - the price at the timestamp itself. Pairing an 8h perp close with a 4h
spot close compares prices hours apart and turns raw price movement into fake
basis volatility; that bug previously inflated basis std from 0.023% to 0.97%.
"""

import argparse
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.feed import backfill_binance, fetch_funding_history, fetch_perp_klines  # noqa: E402

DEFAULT_SYMBOLS = [
    'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT', 'DOGEUSDT', 'BNBUSDT',
    'ADAUSDT', 'LINKUSDT', 'AVAXUSDT', 'DOTUSDT', 'LTCUSDT', 'NEARUSDT',
    'APTUSDT', 'ARBUSDT', 'OPUSDT', 'SUIUSDT',
]


def build_symbol(symbol: str, pages: int) -> pd.DataFrame:
    funding = fetch_funding_history(symbol, pages=pages)
    start = funding.index[0]

    perp = fetch_perp_klines(symbol, interval='8h', start=start)
    spot = backfill_binance(symbol, interval='4h', start=start, verbose=False)

    frame = pd.DataFrame(index=funding.index)
    frame['funding_rate'] = funding['funding_rate']
    frame['perp'] = perp['perp_open'].reindex(frame.index, method='nearest',
                                              tolerance=pd.Timedelta('1h'))
    frame['spot'] = spot['open'].reindex(frame.index, method='nearest',
                                         tolerance=pd.Timedelta('1h'))
    frame = frame.dropna()
    frame['basis'] = frame['perp'] / frame['spot'] - 1.0
    frame['symbol'] = symbol
    frame.index.name = 'timestamp'
    return frame.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbols', nargs='+', default=DEFAULT_SYMBOLS)
    parser.add_argument('--pages', type=int, default=2)
    parser.add_argument('--out', default='data/perp_funding.csv')
    args = parser.parse_args()

    frames = []
    for symbol in args.symbols:
        try:
            frame = build_symbol(symbol, args.pages)
            frames.append(frame)
            print(f"  {symbol:10s} {len(frame):5d} intervals  "
                  f"funding {frame.funding_rate.mean() * 3 * 365 * 100:+6.2f}%/yr  "
                  f"basis std {frame.basis.std() * 100:.4f}%")
        except Exception as exc:                       # noqa: BLE001
            print(f"  {symbol:10s} skipped ({exc})")
        time.sleep(0.15)

    if not frames:
        raise SystemExit('No symbols fetched')

    data = pd.concat(frames, ignore_index=True).sort_values(['timestamp', 'symbol'])
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    data.to_csv(args.out, index=False)

    print(f"\nWrote {len(data):,} rows ({data.symbol.nunique()} symbols) to {args.out}")
    print(f"  range {data.timestamp.min()} -> {data.timestamp.max()}")


if __name__ == "__main__":
    main()
