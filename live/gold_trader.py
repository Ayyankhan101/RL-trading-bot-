"""
Live paper trading for gold.

Reuses `live.trader.LiveTrader` wholesale - the persisted account, the
never-reprocess-a-bar guarantee, the shadow learner and its promotion gate - and
changes only what gold requires:

* **Bars come from Yahoo (GC=F)** rather than a crypto exchange, closed bars
  only, with session gaps preserved.
* **No Fear & Greed index.** Gold has no such series, so the feature is held at
  a constant neutral value: present so the feature vector keeps its shape,
  uninformative so it cannot invent signal.
* **Scheduled-event blackouts.** The one case where the timing of a shock is
  known in advance. Broker spreads on XAUUSD widen 10-50x through an FOMC or NFP
  print, so no new position is opened in the window around one. Exits stay
  allowed - being unable to leave is worse than not entering.

Execution costs come from the ``cfd`` mode in ``environment/execution.py``:
spread as a price offset plus per-lot commission, which is what a broker
actually charges and is why gold is viable at intraday horizons where crypto is
not.
"""

import os
from typing import Any, Dict, List, Optional

import pandas as pd

from features.technical_indicators import add_technical_indicators
from live.calendar import GOLD_CURRENCIES, blackout_windows, fetch_events, in_blackout, relevant_events
from live.gold_feed import GoldFeedError, session_gaps, update_cache
from live.trader import LiveTrader


class GoldLiveTrader(LiveTrader):
    """Paper-trades gold against live COMEX bars."""

    def __init__(self, config: Dict[str, Any], model_path: Optional[str] = None,
                 state_dir: str = 'results/live_gold', symbol: str = 'GC=F',
                 interval: str = '1h'):
        super().__init__(config=config, model_path=model_path, state_dir=state_dir,
                         symbol=symbol, interval=interval, source='yahoo')

        calendar = config.get('calendar', {}) or {}
        self.calendar_enabled = bool(calendar.get('enabled', True))
        self.before_minutes = int(calendar.get('before_minutes', 30))
        self.after_minutes = int(calendar.get('after_minutes', 30))
        self.impacts = tuple(calendar.get('impacts', ('High',)))
        self.blackouts: List[Dict[str, Any]] = []
        self.blocked_entries = 0

    # ------------------------------------------------------------------ data

    def refresh_data(self, seed_history: bool = True) -> pd.DataFrame:
        """Closed gold bars, enriched with the same indicators as every other path."""
        bars = update_cache(symbol=self.symbol, interval=self.interval)

        # Gold has no sentiment index. A constant keeps the feature's shape
        # without inventing information the market never provided.
        bars['Fear & Greed Index'] = 50.0

        enriched = add_technical_indicators(bars)
        enriched = enriched.dropna(
            subset=[c for c in enriched.columns if c != 'Fear & Greed Classification'])

        if enriched.empty:
            raise GoldFeedError('no usable gold bars after indicator warm-up')

        self._gaps = session_gaps(bars, self.interval)
        self.refresh_calendar()
        return enriched

    def refresh_calendar(self) -> None:
        """Reload the event schedule. Failure leaves the bot trading without it."""
        if not self.calendar_enabled:
            return
        try:
            events = relevant_events(fetch_events(), GOLD_CURRENCIES, self.impacts)
            self.blackouts = blackout_windows(events, self.before_minutes,
                                              self.after_minutes)
        except Exception as exc:                       # noqa: BLE001
            print(f"[gold] calendar unavailable ({exc}); trading without blackouts")
            self.blackouts = []

    # -------------------------------------------------------------- blackout

    def entry_blocked(self, moment: pd.Timestamp) -> Optional[Dict[str, Any]]:
        """Suppress new entries inside a scheduled high-impact event window."""
        if not self.blackouts:
            return None

        window = in_blackout(moment, self.blackouts)
        if window is not None:
            self.blocked_entries += 1
        return window

    # -------------------------------------------------------------- reporting

    def _write_status(self, data: pd.DataFrame, price: float, new_bars: int,
                      promotion: Optional[Dict[str, Any]],
                      idle: bool = False) -> Dict[str, Any]:
        status = super()._write_status(data, price, new_bars, promotion, idle)

        # utcnow() is already tz-aware in current pandas; localizing it raises.
        now = pd.Timestamp.now(tz='UTC')
        upcoming = sorted((w for w in self.blackouts if w['end'] >= now),
                          key=lambda w: w['start'])
        status.update({
            'instrument': 'gold',
            'session_gaps': getattr(self, '_gaps', 0),
            'blocked_entries': self.blocked_entries,
            'blackout_windows': len(self.blackouts),
            'next_event': ({
                'title': upcoming[0]['title'],
                'country': upcoming[0]['country'],
                'starts': str(upcoming[0]['start']),
            } if upcoming else None),
        })

        import json
        with open(self.status_path, 'w') as file:
            json.dump(status, file, indent=2, default=str)
        return status
