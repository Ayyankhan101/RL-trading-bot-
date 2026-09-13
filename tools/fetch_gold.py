"""
Fetch gold (COMEX front-month futures, GC=F) into the project's CSV format.

    python tools/fetch_gold.py

Yahoo caps intraday history by interval, which shapes what can be validated:

    1h   730 days  (~12,000 bars)  full walk-forward
    15m   60 days  (~3,800 bars)   persistence check only
    5m    60 days  (~11,000 bars)  persistence check only

Session gaps - weekends and the daily CME break - are preserved rather than
forward-filled. Interpolating across a closed market would let indicators smear
price action that never happened.
"""

import argparse
import json
import os
import sys
import urllib.request
from typing import Dict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RANGES: Dict[str, str] = {'1h': '730d', '15m': '60d', '5m': '60d', '1d': '10y'}


def fetch(symbol: str, interval: str, period: str) -> pd.DataFrame:
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
           f"?interval={interval}&range={period}")
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    payload = json.loads(urllib.request.urlopen(request, timeout=30).read())

    result = payload['chart']['result'][0]
    quote = result['indicators']['quote'][0]

    frame = pd.DataFrame({
        'open': quote['open'], 'high': quote['high'],
        'low': quote['low'], 'close': quote['close'],
    }, index=pd.to_datetime(result['timestamp'], unit='s'))

    frame.index.name = 'timestamp'
    # Drop rather than fill: a bar with no trade did not happen.
    frame = frame.dropna()
    frame = frame[~frame.index.duplicated(keep='last')].sort_index()
    return frame


def session_gap_report(frame: pd.DataFrame, interval: str) -> Dict[str, float]:
    """
    Describe the gaps, because gold does not trade continuously.

    The pipeline's annualization infers periods per year from the *median* bar
    spacing, which stays correct in the presence of weekend gaps precisely
    because it is a median and not a mean.
    """
    deltas = frame.index.to_series().diff().dropna()
    expected = pd.Timedelta(interval.replace('m', 'min'))
    gaps = deltas[deltas > expected * 1.5]

    return {
        'bars': int(len(frame)),
        'median_spacing_minutes': float(deltas.median().total_seconds() / 60),
        'gaps': int(len(gaps)),
        'largest_gap_hours': float(gaps.max().total_seconds() / 3600) if len(gaps) else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symbol', default='GC=F', help='Yahoo symbol')
    parser.add_argument('--name', default='GOLD', help='Output file prefix')
    parser.add_argument('--intervals', nargs='+', default=['1h', '15m', '5m'])
    parser.add_argument('--out-dir', default='data')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    for interval in args.intervals:
        period = RANGES.get(interval, '60d')
        frame = fetch(args.symbol, interval, period)

        # The pipeline expects a Fear & Greed column; gold has no such index, so
        # a flat neutral value keeps the feature present and uninformative
        # rather than absent and shape-changing.
        frame['Fear & Greed Index'] = 50.0

        path = os.path.join(args.out_dir, f"{args.name}_{interval}.csv")
        frame.to_csv(path)

        report = session_gap_report(frame, interval)
        print(f"{path}")
        print(f"  {report['bars']:>6,} bars   {frame.index[0]} -> {frame.index[-1]}")
        print(f"  median spacing {report['median_spacing_minutes']:.0f} min, "
              f"{report['gaps']} session gaps, largest {report['largest_gap_hours']:.1f}h")
        print(f"  median bar range "
              f"{((frame.high - frame.low) / frame.close).median() * 100:.4f}%\n")


if __name__ == "__main__":
    main()
