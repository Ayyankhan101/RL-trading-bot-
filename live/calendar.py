"""
Economic event calendar, for standing aside when the market is about to gap.

A strategy cannot predict what a central bank will say. What it can do is know
*when* the announcement lands and refuse to be holding a leveraged position
through it.

This matters far more on a CFD account than on an exchange. Broker spreads on
XAUUSD widen by 10-50x through an FOMC or NFP print, and stop orders fill deep
past their level. A retail account is rarely killed by a bad forecast; it is
killed by being in the market at 14:00 on a Fed day with a 2-pip stop.

Source: ForexFactory's weekly JSON feed - free, no key, impact-rated. Events are
matched to instruments by currency: gold and BTC both respond to USD events,
gold additionally to major central-bank decisions.
"""

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

FEED_URL = 'https://nfs.faireconomy.media/ff_calendar_thisweek.json'
NASDAQ_URL = 'https://api.nasdaq.com/api/calendar/economicevents?date={date}'
CACHE_PATH = 'data/live/economic_calendar.json'
CACHE_TTL_SECONDS = 6 * 60 * 60      # the weekly schedule changes rarely

# Gold is a dollar-denominated safe haven: US data and Fed decisions move it
# most, but any major central bank shifts the rate-differential story.
GOLD_CURRENCIES = ('USD', 'EUR', 'GBP', 'JPY', 'CNY')
CRYPTO_CURRENCIES = ('USD',)


class CalendarError(RuntimeError):
    """Raised when the calendar cannot be fetched or parsed."""


def _get(url: str, timeout: float) -> Any:
    request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    return json.loads(urllib.request.urlopen(request, timeout=timeout).read())


def _fetch_forexfactory(url: str, timeout: float) -> list:
    """Primary source: impact-rated, one call for the whole week."""
    return _get(url, timeout)


def _fetch_nasdaq(_url: str, timeout: float) -> list:
    """
    Fallback: Nasdaq's calendar, queried per day.

    ForexFactory rate-limits hard - a bot polling on a short cycle is refused
    within a few requests - so a second source keeps the blackout working when
    the first is unavailable.
    """
    events = []
    today = datetime.now(timezone.utc).date()

    for offset in range(7):
        day = today + timedelta(days=offset)
        payload = _get(NASDAQ_URL.format(date=day.isoformat()), timeout)
        rows = ((payload or {}).get('data') or {}).get('rows') or []
        for row in rows:
            gmt = (row.get('gmt') or '').strip()
            if not gmt:
                continue
            events.append({
                'title': row.get('eventName', ''),
                'country': (row.get('country') or '').strip()[:3].upper() or 'USD',
                'date': f"{day.isoformat()}T{gmt}:00+00:00",
                # Nasdaq has no impact rating; treat everything as Medium so it
                # never silently suppresses more than ForexFactory would.
                'impact': 'Medium',
            })
    if not events:
        raise CalendarError('nasdaq calendar returned no events')
    return events


def _read_cache(path: str, ttl: float) -> Optional[list]:
    if not os.path.exists(path):
        return None
    if time.time() - os.path.getmtime(path) > ttl:
        return None
    try:
        with open(path) as file:
            return json.load(file)
    except (ValueError, OSError):
        return None


def _write_cache(path: str, payload: list) -> None:
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as file:
        json.dump(payload, file)


def fetch_events(url: str = FEED_URL, timeout: float = 20.0,
                 cache_path: str = CACHE_PATH,
                 ttl: float = CACHE_TTL_SECONDS) -> pd.DataFrame:
    """
    This week's scheduled economic events, cached to disk.

    The feed rate-limits aggressively - a live bot polling every minute is
    refused with HTTP 429 within a few requests. The weekly schedule changes
    rarely, so it is cached and the stale copy is preferred over no calendar at
    all: trading blind through an FOMC is worse than using a six-hour-old
    timetable.
    """
    payload = _read_cache(cache_path, ttl)

    if payload is None:
        errors = []
        for source in (_fetch_forexfactory, _fetch_nasdaq):
            try:
                payload = source(url, timeout)
                _write_cache(cache_path, payload)
                break
            except Exception as exc:                   # noqa: BLE001
                errors.append(f"{source.__name__}: {exc}")

        if payload is None:
            # A stale schedule beats no schedule: trading blind through an FOMC
            # is worse than using last week's timetable.
            stale = _read_cache(cache_path, ttl=float('inf'))
            if stale:
                print(f"[calendar] live fetch failed ({'; '.join(errors)}); "
                      "using cached schedule")
                payload = stale
            else:
                raise CalendarError('could not fetch calendar -> ' + '; '.join(errors))

    if not payload:
        raise CalendarError('calendar feed returned no events')

    frame = pd.DataFrame(payload)
    frame['timestamp'] = pd.to_datetime(frame['date'], utc=True, format='mixed')
    return frame[['timestamp', 'country', 'title', 'impact']].sort_values('timestamp')


def relevant_events(events: pd.DataFrame, currencies: Sequence[str],
                    impacts: Sequence[str] = ('High',)) -> pd.DataFrame:
    """Filter to the events that actually move the instrument being traded."""
    return events[events['country'].isin(currencies) & events['impact'].isin(impacts)]


def blackout_windows(events: pd.DataFrame, before_minutes: int = 30,
                     after_minutes: int = 30) -> List[Dict[str, Any]]:
    """
    Windows during which no new position should be opened.

    Asymmetric by design is possible, but the default is symmetric: the spread
    starts widening in anticipation and takes time to normalize afterwards.
    """
    windows = []
    for _, event in events.iterrows():
        windows.append({
            'start': event['timestamp'] - timedelta(minutes=before_minutes),
            'end': event['timestamp'] + timedelta(minutes=after_minutes),
            'title': event['title'],
            'country': event['country'],
            'impact': event['impact'],
        })
    return windows


def in_blackout(moment: pd.Timestamp, windows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The window covering ``moment``, or None. Returns the window so it can be logged."""
    if moment.tzinfo is None:
        moment = moment.tz_localize('UTC')

    for window in windows:
        if window['start'] <= moment <= window['end']:
            return window
    return None


def blackout_mask(index: pd.DatetimeIndex, windows: Sequence[Dict[str, Any]]) -> pd.Series:
    """Boolean mask over a bar index: True where trading should be suppressed."""
    if index.tz is None:
        localized = index.tz_localize('UTC')
    else:
        localized = index

    mask = pd.Series(False, index=index)
    for window in windows:
        mask |= (localized >= window['start']) & (localized <= window['end'])
    return mask


def gold_blackouts(before_minutes: int = 30, after_minutes: int = 30) -> List[Dict[str, Any]]:
    """High-impact windows for gold. Returns an empty list if the feed is down."""
    try:
        events = fetch_events()
    except CalendarError as exc:
        print(f"[calendar] unavailable ({exc}); trading without event blackouts")
        return []

    return blackout_windows(relevant_events(events, GOLD_CURRENCIES),
                            before_minutes, after_minutes)
