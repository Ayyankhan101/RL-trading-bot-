"""
Build the basis-trade dataset: funding rate, spot price and perpetual price,
aligned on the 8-hour funding schedule.

    python tools/fetch_funding.py --out data/BTC_basis.csv

Columns: funding_rate, spot_close, perp_close, basis (perp/spot - 1).
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from live.feed import backfill_binance, fetch_funding_history, fetch_perp_klines  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol', default='BTCUSDT')
    parser.add_argument('--pages', type=int, default=4, help='1000 funding intervals each')
    parser.add_argument('--out', default='data/BTC_basis.csv')
    args = parser.parse_args()

    funding = fetch_funding_history(args.symbol, pages=args.pages)
    start = funding.index[0]
    print(f"funding: {len(funding)} intervals, {funding.index[0]} -> {funding.index[-1]}")

    perp = fetch_perp_klines(args.symbol, interval='8h', start=start)
    spot = backfill_binance(args.symbol, interval='4h', start=start, verbose=False)
    print(f"perp:    {len(perp)} bars\nspot:    {len(spot)} bars")

    # Align onto the funding timestamps: funding is the clock.
    #
    # Both legs must be priced at the SAME INSTANT. A kline's `close` is the
    # price at the END of its bar, so pairing an 8h perp close with a 4h spot
    # close compares prices hours apart and injects raw price movement into what
    # is supposed to be a spread. Using each bar's `open` - the price at the
    # timestamp itself - keeps the two legs synchronous.
    frame = pd.DataFrame(index=funding.index)
    frame['funding_rate'] = funding['funding_rate']
    frame['perp_close'] = perp['perp_open'].reindex(frame.index, method='nearest',
                                                    tolerance=pd.Timedelta('1h'))
    frame['spot_close'] = spot['open'].reindex(frame.index, method='nearest',
                                               tolerance=pd.Timedelta('1h'))
    frame = frame.dropna()

    # Basis: what the perpetual trades at relative to spot. A short-perp leg
    # gains as this narrows and loses as it widens.
    frame['basis'] = frame['perp_close'] / frame['spot_close'] - 1.0
    frame.index.name = 'timestamp'

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    frame.to_csv(args.out)

    print(f"\nWrote {len(frame)} rows to {args.out}")
    print(f"  range        {frame.index[0]} -> {frame.index[-1]}")
    print(f"  funding/8h   mean {frame.funding_rate.mean() * 100:+.5f}%  "
          f"positive {(frame.funding_rate > 0).mean() * 100:.1f}%")
    print(f"  basis        mean {frame.basis.mean() * 100:+.4f}%  "
          f"std {frame.basis.std() * 100:.4f}%")


if __name__ == "__main__":
    main()
