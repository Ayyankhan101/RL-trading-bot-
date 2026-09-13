"""
Live market data feed.

Pulls closed OHLC bars from a public exchange REST endpoint, caches them to
disk, and hands back a frame shaped exactly like ``data/BTC.csv`` so the rest of
the pipeline cannot tell the difference between live and historical bars.

Two rules this module enforces, because breaking either would quietly invalidate
every live result:

1. **Only closed bars.** The most recent kline an exchange returns is still
   forming; its high, low and close will change. Acting on it is lookahead in
   the live setting, exactly as reading bar ``t+1`` would be in a backtest.
2. **No volume by default.** The bundled training data has no volume column, so
   a model trained on it expects a feature vector without volume. Silently
   adding volume features live would change the observation the policy sees.
"""

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

import pandas as pd

USER_AGENT = 'rl-trading-bot/1.0'
DEFAULT_CACHE = 'data/live'

# Interval -> (binance code, kraken minutes, coinbase seconds, pandas offset,
#              default years of history to backfill)
#
# The default history shrinks as the interval does: two years of 15m bars is
# ~70,000 rows and 70 API pages, while two years of 4h bars is ~4,400 rows.
INTERVALS: Dict[str, Dict[str, Any]] = {
    '1m': {'binance': '1m', 'kraken': 1, 'coinbase': 60, 'pandas': '1min', 'years': 0.25},
    '5m': {'binance': '5m', 'kraken': 5, 'coinbase': 300, 'pandas': '5min', 'years': 1.0},
    '15m': {'binance': '15m', 'kraken': 15, 'coinbase': 900, 'pandas': '15min', 'years': 2.0},
    '1h': {'binance': '1h', 'kraken': 60, 'coinbase': 3600, 'pandas': '1h', 'years': 4.0},
    '4h': {'binance': '4h', 'kraken': 240, 'coinbase': 21600, 'pandas': '4h', 'years': 6.0},
    '1d': {'binance': '1d', 'kraken': 1440, 'coinbase': 86400, 'pandas': '1D', 'years': 8.0},
}

BARS_PER_PAGE = 1000


def default_start(interval: str) -> pd.Timestamp:
    """How far back to backfill an interval by default."""
    years = INTERVALS[interval]['years']
    # Integer days: a float here trips pandas' generic-unit deprecation.
    return pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta(
        days=int(round(365 * years)))


def pages_needed(interval: str, start: pd.Timestamp) -> int:
    """
    Pages required to reach ``start``, since one request caps at 1000 bars.

    At 15m a page covers only ~10 days, so a fixed page limit that was fine for
    4h bars would silently truncate the history and leave the model training on
    a few weeks of data while believing it had years.
    """
    span = pd.Timestamp.utcnow().tz_localize(None) - start
    bar_seconds = pd.Timedelta(INTERVALS[interval]['pandas']).total_seconds()
    bars = max(1.0, span.total_seconds() / bar_seconds)
    return int(bars // BARS_PER_PAGE) + 2      # +1 for the remainder, +1 slack


class FeedError(RuntimeError):
    """Raised when no data source could supply usable bars."""


def _get_json(url: str, timeout: float = 15.0) -> Any:
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _frame(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame['timestamp'] = pd.to_datetime(frame['timestamp'], utc=True).dt.tz_localize(None)
    return (frame
            .drop_duplicates(subset='timestamp', keep='last')
            .sort_values('timestamp')
            .set_index('timestamp'))


def fetch_binance(symbol: str = 'BTCUSDT', interval: str = '4h', limit: int = 1000) -> pd.DataFrame:
    code = INTERVALS[interval]['binance']
    url = (f"https://api.binance.com/api/v3/klines"
           f"?symbol={symbol}&interval={code}&limit={min(limit, 1000)}")
    payload = _get_json(url)

    now_ms = time.time() * 1000
    rows = []
    for kline in payload:
        open_time, open_, high, low, close, volume, close_time = kline[:7]
        # close_time is the last millisecond of the bar; a bar whose close_time
        # is in the future is still forming.
        if close_time >= now_ms:
            continue
        rows.append({
            'timestamp': pd.Timestamp(open_time, unit='ms'),
            'open': float(open_), 'high': float(high),
            'low': float(low), 'close': float(close),
            'volume': float(volume),
        })

    if not rows:
        raise FeedError('Binance returned no closed bars')
    return _frame(rows)


def backfill_binance(symbol: str = 'BTCUSDT', interval: str = '4h',
                     start: Optional[pd.Timestamp] = None,
                     max_pages: Optional[int] = None, pause: float = 0.25,
                     verbose: bool = True) -> pd.DataFrame:
    """
    Page backwards through Binance klines to build a long, gap-free history.

    Exchanges cap a single request at 1000 bars (~166 days at 4h), which is not
    enough to cover the gap between the bundled dataset and today. Stitching a
    stale CSV onto recent bars would leave a hole in the middle, and indicators
    computed across that hole are garbage - so the live cache is backfilled from
    one source instead.
    """
    code = INTERVALS[interval]['binance']
    if start is None:
        start = default_start(interval)
    # Derive the page budget from the requested span rather than a fixed number:
    # one page is ~166 days at 4h but only ~10 days at 15m.
    if max_pages is None:
        max_pages = pages_needed(interval, start)

    frames: List[Dict[str, Any]] = []
    end_ms = int(time.time() * 1000)
    start_ms = int(start.timestamp() * 1000)

    if verbose:
        print(f"[feed] backfilling {symbol} {interval} from {start.date()} "
              f"(up to {max_pages} pages)")

    for page in range(max_pages):
        url = (f"https://api.binance.com/api/v3/klines"
               f"?symbol={symbol}&interval={code}&limit=1000&endTime={end_ms}")
        payload = _get_json(url)
        if not payload:
            break

        now_ms = time.time() * 1000
        for kline in payload:
            open_time, open_, high, low, close, volume, close_time = kline[:7]
            if close_time >= now_ms:
                continue
            frames.append({
                'timestamp': pd.Timestamp(open_time, unit='ms'),
                'open': float(open_), 'high': float(high),
                'low': float(low), 'close': float(close),
                'volume': float(volume),
            })

        earliest = int(payload[0][0])
        if verbose and page and page % 10 == 0:
            print(f"[feed]   page {page}/{max_pages}, back to "
                  f"{pd.Timestamp(earliest, unit='ms').date()}")

        if earliest <= start_ms:
            break
        if len(payload) < BARS_PER_PAGE:
            break

        end_ms = earliest - 1
        time.sleep(pause)

    if not frames:
        raise FeedError('Binance backfill returned no bars')

    frame = _frame(frames)
    frame = frame[frame.index >= start]
    frame.attrs['source'] = 'binance'
    return frame


def fetch_kraken(pair: str = 'XBTUSD', interval: str = '4h') -> pd.DataFrame:
    minutes = INTERVALS[interval]['kraken']
    payload = _get_json(f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={minutes}")

    if payload.get('error'):
        raise FeedError(f"Kraken error: {payload['error']}")

    series = next(value for key, value in payload['result'].items() if key != 'last')
    last_open = payload['result'].get('last')

    rows = []
    for entry in series:
        open_time = int(entry[0])
        # Kraken's `last` marks the still-forming bar.
        if last_open is not None and open_time >= int(last_open):
            continue
        rows.append({
            'timestamp': pd.Timestamp(open_time, unit='s'),
            'open': float(entry[1]), 'high': float(entry[2]),
            'low': float(entry[3]), 'close': float(entry[4]),
            'volume': float(entry[6]),
        })

    if not rows:
        raise FeedError('Kraken returned no closed bars')
    return _frame(rows)


def fetch_coinbase(product: str = 'BTC-USD', interval: str = '4h') -> pd.DataFrame:
    granularity = INTERVALS[interval]['coinbase']
    payload = _get_json(
        f"https://api.exchange.coinbase.com/products/{product}/candles"
        f"?granularity={granularity}")

    now = time.time()
    rows = []
    for entry in payload:
        start, low, high, open_, close, volume = entry[:6]
        if start + granularity > now:
            continue
        rows.append({
            'timestamp': pd.Timestamp(int(start), unit='s'),
            'open': float(open_), 'high': float(high),
            'low': float(low), 'close': float(close),
            'volume': float(volume),
        })

    if not rows:
        raise FeedError('Coinbase returned no closed bars')
    return _frame(rows)


SOURCES = {
    'binance': fetch_binance,
    'kraken': fetch_kraken,
    'coinbase': fetch_coinbase,
}


def fetch_bars(interval: str = '4h', source: str = 'auto') -> pd.DataFrame:
    """
    Fetch closed bars, falling back across venues.

    A single exchange being unreachable (geoblocked, rate limited, down) should
    pause the bot, not end its run.
    """
    if interval not in INTERVALS:
        raise ValueError(f"interval must be one of {sorted(INTERVALS)}")

    order = list(SOURCES) if source == 'auto' else [source]
    errors = []

    for name in order:
        try:
            frame = SOURCES[name](interval=interval)
            frame.attrs['source'] = name
            return frame
        except (urllib.error.URLError, FeedError, KeyError, ValueError, TimeoutError) as exc:
            errors.append(f"{name}: {exc}")

    raise FeedError('No data source reachable -> ' + '; '.join(errors))


def cache_path(symbol: str, interval: str, cache_dir: str = DEFAULT_CACHE) -> str:
    return os.path.join(cache_dir, f"{symbol}_{interval}.csv")


def update_cache(symbol: str = 'BTCUSD', interval: str = '4h',
                 cache_dir: str = DEFAULT_CACHE, source: str = 'auto',
                 keep_volume: bool = False) -> pd.DataFrame:
    """
    Merge freshly fetched bars into the on-disk history and return the result.

    Exchanges only serve a window of recent klines, so the cache is what gives
    the bot enough history to warm up its indicators across restarts.
    """
    fresh = fetch_bars(interval=interval, source=source)
    if not keep_volume:
        # Match the training feature set exactly - see the module docstring.
        fresh = fresh.drop(columns=['volume'], errors='ignore')

    path = cache_path(symbol, interval, cache_dir)
    if os.path.exists(path):
        existing = pd.read_csv(path, parse_dates=['timestamp']).set_index('timestamp')
        combined = pd.concat([existing, fresh])
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
    else:
        combined = fresh

    os.makedirs(cache_dir, exist_ok=True)
    combined.to_csv(path)
    combined.attrs['source'] = fresh.attrs.get('source', source)
    return combined


def build_cache(symbol: str = 'BTCUSD', interval: str = '4h',
                cache_dir: str = DEFAULT_CACHE, start: Optional[str] = None,
                keep_volume: bool = False, force: bool = False) -> pd.DataFrame:
    """
    Create or extend a single-source, gap-free live cache.

    Called once before live trading starts. Subsequent cycles only need
    ``update_cache`` to append the handful of bars that closed since.
    """
    path = cache_path(symbol, interval, cache_dir)
    start_ts = pd.Timestamp(start) if start else default_start(interval)

    if os.path.exists(path) and not force:
        existing = pd.read_csv(path, parse_dates=['timestamp']).set_index('timestamp')
        gaps = existing.index.to_series().diff().dropna()
        expected = pd.Timedelta(INTERVALS[interval]['pandas'])
        if not gaps.empty and gaps.max() <= expected * 2 and existing.index[0] <= start_ts:
            return existing
        print('[feed] cache has gaps or starts late; rebuilding from the exchange')

    bars = backfill_binance(interval=interval, start=start_ts)
    if not keep_volume:
        bars = bars.drop(columns=['volume'], errors='ignore')

    bars = attach_fear_greed(bars)

    os.makedirs(cache_dir, exist_ok=True)
    bars.to_csv(path)
    print(f"[feed] cache built: {len(bars)} bars, {bars.index[0]} -> {bars.index[-1]}")
    return bars


def fetch_fear_greed(limit: int = 0) -> pd.Series:
    """
    Daily Fear & Greed Index from alternative.me.

    The training data carries the real index, so the live bot fetches the real
    index too. Substituting a neutral constant would hand the policy a different
    input than the one it learned on for this feature.

    ``limit=0`` asks for the full history (2018 onwards).
    """
    payload = _get_json(f"https://api.alternative.me/fng/?limit={limit}&format=json")
    if payload.get('metadata', {}).get('error'):
        raise FeedError(f"Fear & Greed error: {payload['metadata']['error']}")

    entries = payload.get('data') or []
    if not entries:
        raise FeedError('Fear & Greed returned no data')

    index = pd.to_datetime([int(entry['timestamp']) for entry in entries], unit='s')
    values = [float(entry['value']) for entry in entries]
    return pd.Series(values, index=index, name='Fear & Greed Index').sort_index()


def attach_fear_greed(bars: pd.DataFrame, sentiment: Optional[pd.Series] = None) -> pd.DataFrame:
    """
    Join the daily sentiment reading onto intraday bars.

    The index publishes once a day, so each bar takes the most recent reading at
    or before its own timestamp - never a later one, which would be lookahead.
    """
    out = bars.copy()

    if sentiment is None:
        try:
            sentiment = fetch_fear_greed()
        except (urllib.error.URLError, FeedError, TimeoutError, ValueError) as exc:
            print(f"[feed] Fear & Greed unavailable ({exc}); keeping existing values")
            sentiment = None

    if sentiment is None or sentiment.empty:
        return out

    aligned = sentiment.reindex(sentiment.index.union(out.index)).ffill().reindex(out.index)

    if 'Fear & Greed Index' in out.columns:
        out['Fear & Greed Index'] = out['Fear & Greed Index'].fillna(aligned)
    else:
        out['Fear & Greed Index'] = aligned

    return out


# --------------------------------------------------------------- funding rates

FUTURES_BASE = 'https://fapi.binance.com/fapi/v1'


def fetch_funding_history(symbol: str = 'BTCUSDT', pages: int = 4,
                          pause: float = 0.2) -> pd.DataFrame:
    """
    Perpetual funding rate history from Binance USD-M futures.

    Funding settles every 8 hours. A **positive** rate means longs pay shorts, so
    a short perpetual leg *receives* it. This is the entire return source of the
    basis trade: no price forecast is involved.

    One request caps at 1000 intervals (~333 days), so pages are walked
    backwards the same way klines are.
    """
    rows: List[Dict[str, Any]] = []
    end_ms = int(time.time() * 1000)

    for _ in range(pages):
        payload = _get_json(f"{FUTURES_BASE}/fundingRate"
                            f"?symbol={symbol}&limit=1000&endTime={end_ms}")
        if not payload:
            break
        rows.extend(payload)
        end_ms = int(payload[0]['fundingTime']) - 1
        time.sleep(pause)

    if not rows:
        raise FeedError('Binance returned no funding history')

    frame = pd.DataFrame(rows).drop_duplicates('fundingTime')
    frame['timestamp'] = pd.to_datetime(frame['fundingTime'], unit='ms')
    frame['funding_rate'] = frame['fundingRate'].astype(float)

    return (frame[['timestamp', 'funding_rate']]
            .sort_values('timestamp')
            .set_index('timestamp'))


def fetch_perp_klines(symbol: str = 'BTCUSDT', interval: str = '8h',
                      start: Optional[pd.Timestamp] = None,
                      max_pages: int = 20, pause: float = 0.2) -> pd.DataFrame:
    """
    Perpetual futures klines, for the short leg's mark price.

    The basis trade needs both legs: spot for the long and perp for the short.
    Their difference is the basis, and changes in it are the trade's only
    price-related P&L.
    """
    rows: List[Dict[str, Any]] = []
    end_ms = int(time.time() * 1000)
    start_ms = int(start.timestamp() * 1000) if start is not None else None

    for _ in range(max_pages):
        payload = _get_json(f"{FUTURES_BASE}/klines"
                            f"?symbol={symbol}&interval={interval}"
                            f"&limit=1000&endTime={end_ms}")
        if not payload:
            break

        now_ms = time.time() * 1000
        for kline in payload:
            open_time, open_, high, low, close, volume, close_time = kline[:7]
            if close_time >= now_ms:
                continue
            rows.append({
                'timestamp': pd.Timestamp(open_time, unit='ms'),
                'perp_open': float(open_), 'perp_high': float(high),
                'perp_low': float(low), 'perp_close': float(close),
            })

        earliest = int(payload[0][0])
        if start_ms is not None and earliest <= start_ms:
            break
        if len(payload) < BARS_PER_PAGE:
            break
        end_ms = earliest - 1
        time.sleep(pause)

    if not rows:
        raise FeedError('Binance returned no perpetual klines')

    frame = pd.DataFrame(rows).drop_duplicates('timestamp')
    frame['timestamp'] = pd.to_datetime(frame['timestamp'])
    return frame.sort_values('timestamp').set_index('timestamp')
