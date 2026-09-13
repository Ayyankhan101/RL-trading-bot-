"""
Live gold bars from Yahoo Finance (COMEX front-month futures, GC=F).

Two rules, both of which would silently corrupt a live record if broken:

1. **Only closed bars.** Yahoo's most recent bar is still forming - its high,
   low and close will change. Acting on it is the live equivalent of reading
   bar t+1 in a backtest, and the bot would appear to trade beautifully.
2. **Session gaps stay gaps.** Gold stops trading at weekends and through the
   daily CME break. Forward-filling those would smear indicators across price
   action that never happened.

Bars are cached to disk so a restart does not depend on the feed being up, and
so the indicator warm-up survives across runs.
"""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

import pandas as pd

CHART_URL = ('https://query1.finance.yahoo.com/v8/finance/chart/{symbol}'
             '?interval={interval}&range={period}')
CACHE_DIR = 'data/live'

# Yahoo caps intraday history by interval.
MAX_RANGE = {'5m': '60d', '15m': '60d', '30m': '60d', '1h': '730d', '1d': '10y'}


class GoldFeedError(RuntimeError):
    """Raised when gold bars cannot be fetched."""


def cache_path(symbol: str, interval: str, cache_dir: str = CACHE_DIR) -> str:
    safe = symbol.replace('=', '').replace('/', '')
    return os.path.join(cache_dir, f"{safe}_{interval}.csv")


def fetch_bars(symbol: str = 'GC=F', interval: str = '1h',
               period: Optional[str] = None, timeout: float = 30.0) -> pd.DataFrame:
    """Fetch bars, dropping the one still forming."""
    period = period or MAX_RANGE.get(interval, '60d')
    url = CHART_URL.format(symbol=symbol, interval=interval, period=period)
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

    try:
        payload = json.loads(urllib.request.urlopen(request, timeout=timeout).read())
    except (urllib.error.URLError, ValueError, TimeoutError) as exc:
        raise GoldFeedError(f"could not fetch {symbol}: {exc}") from exc

    results = (payload.get('chart') or {}).get('result') or []
    if not results:
        raise GoldFeedError(f"{symbol}: chart response contained no result")

    result = results[0]
    quote = result['indicators']['quote'][0]

    frame = pd.DataFrame({
        'open': quote['open'], 'high': quote['high'],
        'low': quote['low'], 'close': quote['close'],
    }, index=pd.to_datetime(result['timestamp'], unit='s'))

    frame.index.name = 'timestamp'
    frame = frame.dropna()
    frame = frame[~frame.index.duplicated(keep='last')].sort_index()

    frame = drop_forming_bar(frame, interval)
    if frame.empty:
        raise GoldFeedError(f"{symbol}: no closed bars available")
    return frame


def drop_forming_bar(frame: pd.DataFrame, interval: str) -> pd.DataFrame:
    """
    Remove the final bar if its interval has not elapsed.

    Yahoo stamps a bar with its *opening* time, so the last row is still open
    until one full interval has passed since that stamp.
    """
    if frame.empty:
        return frame

    step = pd.Timedelta(interval.replace('m', 'min'))
    now = pd.Timestamp.utcnow().tz_localize(None)

    if frame.index[-1] + step > now:
        return frame.iloc[:-1]
    return frame


def update_cache(symbol: str = 'GC=F', interval: str = '1h',
                 cache_dir: str = CACHE_DIR) -> pd.DataFrame:
    """
    Merge freshly fetched bars into the on-disk history.

    Yahoo serves only a rolling window, so the cache is what gives the bot a
    long enough warm-up for its indicators across restarts.
    """
    fresh = fetch_bars(symbol, interval)
    path = cache_path(symbol, interval, cache_dir)

    if os.path.exists(path):
        existing = pd.read_csv(path, parse_dates=['timestamp']).set_index('timestamp')
        combined = pd.concat([existing, fresh])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    else:
        combined = fresh

    os.makedirs(cache_dir, exist_ok=True)
    combined.to_csv(path)
    return combined


def session_gaps(frame: pd.DataFrame, interval: str) -> int:
    """Count gaps larger than one interval - weekends and exchange breaks."""
    deltas = frame.index.to_series().diff().dropna()
    return int((deltas > pd.Timedelta(interval.replace('m', 'min')) * 1.5).sum())
