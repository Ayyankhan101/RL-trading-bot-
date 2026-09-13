"""
Live paper trading for the funding-rate carry strategy.

Mirrors ``strategies/basis_carry.py`` interval by interval against real funding
payments, real spot and real perpetual prices. No orders are placed and no keys
are needed.

Each cycle:

1. Refresh funding history and both legs' prices from Binance.
2. For every funding interval that has settled since the last run, in order:
   mark the basis, settle funding, then evaluate entry and exit.
3. Persist the account, the equity curve and a status snapshot.

The same rules the backtest used are applied here - the trailing-funding signal
is lagged, the basis is marked every interval rather than at exit, and fees are
charged on all four fills - so a live week is comparable to a backtested one.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from live.feed import backfill_binance, fetch_funding_history, fetch_perp_klines
from strategies.basis_carry import (
    BasisConfig,
    liquidation_move,
    trailing_funding,
    weekly_statistics,
)

CACHE_PATH = 'data/live/BTC_basis.csv'


@dataclass
class CarryState:
    """Persisted account. A restart resumes this rather than starting fresh."""
    equity: float
    initial_capital: float
    in_position: bool = False
    entry_timestamp: Optional[str] = None
    entry_basis: Optional[float] = None
    last_basis: Optional[float] = None
    last_interval: Optional[str] = None
    funding_collected: float = 0.0
    basis_pnl: float = 0.0
    costs_paid: float = 0.0
    intervals_in_position: int = 0
    intervals_seen: int = 0
    liquidations: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)


class LiveBasisTrader:
    """Paper-trades the carry against live funding settlements."""

    def __init__(self, config: Dict[str, Any], state_dir: str = 'results/live_basis',
                 symbol: str = 'BTCUSDT'):
        self.config = config
        self.params = BasisConfig.from_dict(config)
        self.symbol = symbol
        self.state_dir = state_dir
        os.makedirs(state_dir, exist_ok=True)

        self.state_path = os.path.join(state_dir, 'account.json')
        self.equity_path = os.path.join(state_dir, 'equity.csv')
        self.status_path = os.path.join(state_dir, 'status.json')
        self.state = self._load_state()

    # ------------------------------------------------------------- persistence

    def _load_state(self) -> CarryState:
        if os.path.exists(self.state_path):
            with open(self.state_path) as file:
                return CarryState(**json.load(file))
        return CarryState(equity=self.params.initial_capital,
                          initial_capital=self.params.initial_capital)

    def save(self) -> None:
        with open(self.state_path, 'w') as file:
            json.dump(asdict(self.state), file, indent=2, default=str)

    # -------------------------------------------------------------------- data

    def refresh_data(self, pages: int = 2) -> pd.DataFrame:
        """
        Rebuild the aligned funding/spot/perp frame.

        Both legs are priced at the bar's **open** - the price at the timestamp
        itself. Pairing an 8h perp close with a 4h spot close compares prices
        hours apart and turns raw price movement into fake basis volatility.
        """
        funding = fetch_funding_history(self.symbol, pages=pages)
        start = funding.index[0]
        perp = fetch_perp_klines(self.symbol, interval='8h', start=start)
        spot = backfill_binance(self.symbol, interval='4h', start=start, verbose=False)

        frame = pd.DataFrame(index=funding.index)
        frame['funding_rate'] = funding['funding_rate']
        frame['perp_close'] = perp['perp_open'].reindex(
            frame.index, method='nearest', tolerance=pd.Timedelta('1h'))
        frame['spot_close'] = spot['open'].reindex(
            frame.index, method='nearest', tolerance=pd.Timedelta('1h'))
        frame = frame.dropna()
        frame['basis'] = frame['perp_close'] / frame['spot_close'] - 1.0
        frame.index.name = 'timestamp'

        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        frame.to_csv(CACHE_PATH)
        return frame

    # ----------------------------------------------------------------- trading

    def process(self, data: pd.DataFrame, verbose: bool = True) -> List[Dict[str, Any]]:
        """Apply every funding interval that has settled since the last run."""
        signal = trailing_funding(data['funding_rate'], self.params.lookback)
        liq_distance = liquidation_move(self.params.leverage,
                                        self.params.maintenance_margin)

        start = 0
        if self.state.last_interval:
            start = int((data.index <= pd.Timestamp(self.state.last_interval)).sum())

        processed: List[Dict[str, Any]] = []
        rows: List[Dict[str, Any]] = []

        for i in range(start, len(data)):
            timestamp = data.index[i]
            rate = float(data['funding_rate'].iloc[i])
            basis = float(data['basis'].iloc[i])
            edge = signal.iloc[i]

            # --- mark the basis and settle funding on an open position
            if self.state.in_position and self.state.equity > 0:
                if i > 0:
                    adverse = float(data['perp_close'].iloc[i] /
                                    data['perp_close'].iloc[i - 1] - 1.0)
                    if adverse >= liq_distance:
                        self.state.liquidations += 1
                        self.state.events.append({
                            'timestamp': str(timestamp), 'action': 'LIQUIDATED',
                            'equity_lost': self.state.equity})
                        self.state.equity = 0.0
                        self.state.in_position = False
                        if verbose:
                            print(f"[carry] {timestamp} LIQUIDATED")
                        break

                if self.state.last_basis is not None:
                    notional = self.state.equity * self.params.leverage
                    drift = notional * (self.state.last_basis - basis)
                    self.state.equity += drift
                    self.state.basis_pnl += drift

                notional = self.state.equity * self.params.leverage
                funding = notional * rate
                self.state.equity += funding
                self.state.funding_collected += funding
                self.state.intervals_in_position += 1

            # --- exit / entry on the lagged signal
            if np.isfinite(edge):
                if self.state.in_position and edge <= self.params.exit_threshold:
                    self._charge_fees('EXIT', timestamp, verbose)
                    self.state.in_position = False
                elif not self.state.in_position and edge > self.params.entry_threshold:
                    self._charge_fees('ENTER', timestamp, verbose)
                    self.state.in_position = True
                    self.state.entry_timestamp = str(timestamp)
                    self.state.entry_basis = basis

            self.state.last_basis = basis if self.state.in_position else None
            self.state.last_interval = str(timestamp)
            self.state.intervals_seen += 1

            rows.append({'timestamp': timestamp, 'equity': self.state.equity,
                         'funding_rate': rate, 'basis': basis,
                         'in_position': self.state.in_position})
            processed.append(rows[-1])

        if rows:
            frame = pd.DataFrame(rows)
            frame.to_csv(self.equity_path, mode='a', index=False,
                         header=not os.path.exists(self.equity_path))

        return processed

    def _charge_fees(self, action: str, timestamp: Any, verbose: bool) -> None:
        cost = self.state.equity * self.params.leverage * self.params.fee_per_leg * 2
        self.state.equity -= cost
        self.state.costs_paid += cost
        self.state.events.append({'timestamp': str(timestamp), 'action': action,
                                  'cost': cost, 'equity': self.state.equity})
        if verbose:
            print(f"[carry] {timestamp} {action}  fees ${cost:,.2f}  "
                  f"equity ${self.state.equity:,.2f}")

    # ---------------------------------------------------------------- reporting

    def tick(self, verbose: bool = True) -> Dict[str, Any]:
        data = self.refresh_data()
        processed = self.process(data, verbose=verbose)
        self.save()
        return self.write_status(data, len(processed))

    def write_status(self, data: pd.DataFrame, new_intervals: int) -> Dict[str, Any]:
        weekly: Dict[str, Any] = {}
        if os.path.exists(self.equity_path):
            curve = pd.read_csv(self.equity_path, parse_dates=['timestamp'])
            if len(curve) > 2:
                weekly = weekly_statistics(
                    curve.set_index('timestamp')['equity'].astype(float))

        recent = data['funding_rate'].tail(self.params.lookback)
        status = {
            'updated_at': datetime.now().isoformat(timespec='seconds'),
            'strategy': 'funding_carry',
            'mode': 'paper',
            'symbol': self.symbol,
            'leverage': self.params.leverage,
            'equity': self.state.equity,
            'total_return': (self.state.equity / self.state.initial_capital - 1.0
                             if self.state.initial_capital else 0.0),
            'in_position': self.state.in_position,
            'entry_timestamp': self.state.entry_timestamp,
            'funding_collected': self.state.funding_collected,
            'basis_pnl': self.state.basis_pnl,
            'costs_paid': self.state.costs_paid,
            'intervals_in_position': self.state.intervals_in_position,
            'intervals_seen': self.state.intervals_seen,
            'liquidations': self.state.liquidations,
            'new_intervals': new_intervals,
            'last_interval': self.state.last_interval,
            'last_funding_rate': float(data['funding_rate'].iloc[-1]),
            'trailing_funding': float(recent.mean()),
            'current_basis': float(data['basis'].iloc[-1]),
            'annualized_funding': float(data['funding_rate'].iloc[-1]) * 3 * 365,
            'liquidation_move': liquidation_move(self.params.leverage,
                                                 self.params.maintenance_margin),
            'weekly': weekly,
        }

        with open(self.status_path, 'w') as file:
            json.dump(status, file, indent=2, default=str)
        return status
