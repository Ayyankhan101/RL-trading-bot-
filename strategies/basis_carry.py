"""
Delta-neutral funding-rate carry (cash-and-carry basis trade).

Hold long spot and short an equal notional of the perpetual future. Price moves
cancel between the legs, so the position has no directional exposure. Perpetual
longs pay shorts a funding fee every 8 hours; the short leg collects it.

This is structurally different from everything else in this repo: **there is no
forecast**. The RL agent had to be directionally right often enough to clear
costs and could not be. This collects a fee that exists because leveraged longs
want exposure and will pay for it.

P&L has exactly three components:

1. **Funding** - ``notional * funding_rate`` every 8h, received when the rate is
   positive and paid when it is negative.
2. **Basis convergence** - the legs are opened at a spread (perp vs spot) and
   closed at another. A short-perp position gains when that spread narrows. This
   accrues **every interval**, not in a lump at exit: the perpetual leg's
   unrealized P&L moves with the basis continuously, and booking it only on exit
   would hide the drawdown a widening basis causes while the position is open.
3. **Costs** - fees on all four fills (two legs in, two legs out).

Leverage multiplies all three. It also creates the one way this trade actually
fails: if the perpetual rallies far enough against the short leg before the spot
collateral is credited, the position is liquidated. That check is modeled
explicitly rather than assumed away.

References: the mechanics are standard cash-and-carry arbitrage; for the
crypto-specific funding literature see e.g. "Carry" (Koijen, Moskowitz, Pedersen
& Vrugt, Journal of Financial Economics, 2018), which frames funding as the
carry component of expected return.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

FUNDING_INTERVALS_PER_YEAR = 3 * 365      # every 8 hours
FUNDING_INTERVALS_PER_WEEK = 21


@dataclass
class BasisConfig:
    """Parameters of the carry trade."""
    initial_capital: float = 10_000.0
    leverage: float = 3.0
    fee_per_leg: float = 0.0002           # maker on both spot and perp
    entry_threshold: float = 0.0          # trailing funding needed to enter
    exit_threshold: float = -0.00005      # trailing funding that forces an exit
    lookback: int = 21                    # intervals in the trailing average
    maintenance_margin: float = 0.005     # exchange maintenance requirement
    max_hold_intervals: Optional[int] = None

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "BasisConfig":
        basis = dict(config.get('basis', {}) or {})
        return cls(**{k: v for k, v in basis.items() if k in cls.__dataclass_fields__})


@dataclass
class BasisResult:
    equity: pd.Series
    trades: List[Dict[str, Any]] = field(default_factory=list)
    funding_collected: float = 0.0
    basis_pnl: float = 0.0
    costs_paid: float = 0.0
    liquidations: int = 0
    intervals_in_position: int = 0


def trailing_funding(rates: pd.Series, lookback: int) -> pd.Series:
    """
    Mean funding over the trailing window, shifted one interval.

    Shifted because the decision for interval ``t`` may only use rates that have
    already settled. Including the current rate would be reading the payment
    before deciding whether to be there to receive it.
    """
    return rates.rolling(lookback, min_periods=max(2, lookback // 2)).mean().shift(1)


def liquidation_move(leverage: float, maintenance_margin: float) -> float:
    """
    Adverse price move that wipes the margin on the short leg.

    At ``L`` times leverage the position is gone once price moves roughly
    ``1/L`` against it, less the maintenance requirement. This is why the
    leverage that would turn this trade into a weekly double-digit return is not
    survivable: at 107x the liquidation distance is under 1%, and BTC covers
    that in an hour.
    """
    if leverage <= 0:
        return float('inf')
    return max(0.0, (1.0 / leverage) - maintenance_margin)


def run_basis_carry(data: pd.DataFrame, config: BasisConfig) -> BasisResult:
    """
    Simulate the carry trade over an aligned funding/spot/perp frame.

    ``data`` needs columns ``funding_rate``, ``spot_close``, ``perp_close`` and
    ``basis``, indexed by funding timestamp.
    """
    required = {'funding_rate', 'spot_close', 'perp_close', 'basis'}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"basis data missing columns: {sorted(missing)}")

    rates = data['funding_rate']
    signal = trailing_funding(rates, config.lookback)
    basis = data['basis'].to_numpy(dtype=np.float64)
    perp = data['perp_close'].to_numpy(dtype=np.float64)

    equity = config.initial_capital
    in_position = False
    entry_basis = 0.0
    entry_index = 0
    liquidation_threshold = liquidation_move(config.leverage, config.maintenance_margin)

    curve: List[float] = []
    result = BasisResult(equity=pd.Series(dtype=float))

    for i in range(len(data)):
        edge = signal.iloc[i]

        # ---- exit decisions, evaluated before this interval's funding settles
        if in_position:
            adverse = (perp[i] / perp[i - 1] - 1.0) if i > 0 else 0.0
            held = i - entry_index

            liquidated = adverse >= liquidation_threshold
            stale = (config.max_hold_intervals is not None
                     and held >= config.max_hold_intervals)
            unattractive = np.isfinite(edge) and edge <= config.exit_threshold

            if liquidated:
                # The short leg is closed by the exchange at the worst moment.
                loss = equity
                equity = 0.0
                result.liquidations += 1
                result.trades.append({'timestamp': data.index[i], 'action': 'LIQUIDATED',
                                      'pnl': -loss, 'intervals_held': held})
                in_position = False
                curve.append(equity)
                break

            if unattractive or stale:
                notional = equity * config.leverage
                cost = notional * config.fee_per_leg * 2      # closing both legs
                equity -= cost
                result.costs_paid += cost
                result.trades.append({
                    'timestamp': data.index[i], 'action': 'EXIT',
                    'cost': cost, 'intervals_held': held, 'equity': equity,
                })
                in_position = False

        # ---- entry
        if not in_position and np.isfinite(edge) and edge > config.entry_threshold:
            notional = equity * config.leverage
            cost = notional * config.fee_per_leg * 2          # opening both legs
            equity -= cost
            result.costs_paid += cost
            entry_basis = basis[i]
            entry_index = i
            in_position = True
            result.trades.append({
                'timestamp': data.index[i], 'action': 'ENTER',
                'cost': cost, 'basis': entry_basis, 'equity': equity,
            })

        # ---- mark the basis and settle funding on whatever position is open
        if in_position:
            notional = equity * config.leverage

            # Short perp gains as the basis narrows. Marked every interval so a
            # widening basis shows up as drawdown while the position is live.
            if i > entry_index:
                drift = notional * (basis[i - 1] - basis[i])
                equity += drift
                result.basis_pnl += drift
                notional = equity * config.leverage

            funding = notional * rates.iloc[i]
            equity += funding
            result.funding_collected += funding
            result.intervals_in_position += 1

        curve.append(equity)

    # Close any open position at the end. The basis is already marked to the
    # final interval, so only the exit fees remain.
    if in_position and equity > 0:
        cost = equity * config.leverage * config.fee_per_leg * 2
        equity -= cost
        result.costs_paid += cost
        curve[-1] = equity

    result.equity = pd.Series(curve, index=data.index[:len(curve)], name='equity')
    return result


def weekly_statistics(equity: pd.Series) -> Dict[str, Any]:
    """
    Weekly return distribution - the property this strategy is chosen for.

    The RL agent's problem was never only the average return; it was that no
    week was reliably positive. Funding carry is judged on hit rate first.
    """
    weekly = equity.resample('W').last().pct_change().dropna()
    if weekly.empty:
        return {}

    return {
        'weeks': int(len(weekly)),
        'mean_weekly_return': float(weekly.mean()),
        'median_weekly_return': float(weekly.median()),
        'positive_weeks': int((weekly > 0).sum()),
        'positive_week_rate': float((weekly > 0).mean()),
        'worst_week': float(weekly.min()),
        'best_week': float(weekly.max()),
        'weekly_volatility': float(weekly.std(ddof=1)) if len(weekly) > 1 else 0.0,
    }
