"""
Multi-sleeve cross-sectional funding carry.

A book of independent delta-neutral sleeves rather than one concentrated
position. Three things this buys over the single-asset carry in
``strategies/basis_carry.py``:

**Selection.** Each interval the symbols are ranked by their most recent settled
funding and the richest are held. Measured across 16 perpetuals, top-5 selection
returned 3.59%/yr against BTC-alone's 3.20%, at roughly double the Sharpe.

**Both directions.** Positive funding is collected by a short perpetual against
long spot. *Negative* funding is collected by the mirror position - long
perpetual, short spot. In the sample APTUSDT paid -16.43%/yr and DOTUSDT
-6.45%/yr, which the one-directional book could not touch at all.

**Survivable failure.** The decisive property. A single leveraged position that
gets liquidated takes the whole account to zero. Here each sleeve holds 1/k of
capital, so one liquidation costs 1/k of the book and the rest keeps running.
That structural difference permits more leverage at the same ruin budget than
any signal improvement in this project achieved.

**Per-sleeve leverage.** A single book-wide leverage is sized by the worst asset
in the book. Measured here, the worst 8-hour move across 16 perpetuals was
33.67% - an altcoin - which forced every sleeve, including BTC, down to 2.9x and
cost more return than the extra funding added. Each sleeve is therefore sized
against **its own** empirical tail, so BTC is not punished for what SUI does.

Sign convention throughout: ``direction = -1`` is short perpetual (collects
positive funding), ``direction = +1`` is long perpetual (collects negative
funding).
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strategies.basis_carry import FUNDING_INTERVALS_PER_YEAR, trailing_funding
from strategies.risk_budget import liquidation_distance

SHORT_PERP, LONG_PERP = -1, 1


@dataclass
class CarryBookConfig:
    initial_capital: float = 10_000.0
    leverage: float = 3.0
    max_sleeves: int = 5
    fee_per_leg: float = 0.0002
    entry_threshold: float = 0.00002      # |trailing funding| needed to open
    exit_threshold: float = 0.0           # |trailing funding| below this closes
    lookback: int = 21
    maintenance_margin: float = 0.005
    allow_negative_funding: bool = True   # mirror sleeves on negative funding
    per_symbol_leverage: Optional[Dict[str, float]] = None
    max_leverage_cap: float = 12.0

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "CarryBookConfig":
        book = dict(config.get('carry_book', {}) or {})
        return cls(**{k: v for k, v in book.items() if k in cls.__dataclass_fields__})


@dataclass
class Sleeve:
    symbol: str
    direction: int
    capital: float
    entry_basis: float
    last_basis: float
    entry_index: int
    leverage: float


@dataclass
class BookResult:
    equity: pd.Series
    funding_collected: float = 0.0
    basis_pnl: float = 0.0
    costs_paid: float = 0.0
    liquidations: int = 0
    liquidated_symbols: List[str] = field(default_factory=list)
    sleeve_intervals: int = 0
    events: List[Dict[str, Any]] = field(default_factory=list)


def to_panels(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Pivot the long-format dataset into aligned wide panels."""
    required = {'timestamp', 'symbol', 'funding_rate', 'basis', 'perp'}
    missing = required - set(data.columns)
    if missing:
        raise KeyError(f"carry data missing columns: {sorted(missing)}")

    panels = {
        field: data.pivot(index='timestamp', columns='symbol', values=field)
        for field in ('funding_rate', 'basis', 'perp')
    }
    index = panels['funding_rate'].dropna(how='any').index
    return {k: v.loc[index] for k, v in panels.items()}


def solve_per_symbol_leverage(data: pd.DataFrame, maintenance_margin: float = 0.005,
                              budget: float = 0.01, cap: float = 12.0) -> Dict[str, float]:
    """
    Ruin-budgeted leverage for each symbol against its own tail.

    A book-wide leverage is sized by its worst member. Sizing per symbol lets a
    well-behaved asset carry the leverage it can actually support while a jumpy
    one is held small, which is the entire point of running a book rather than a
    position.
    """
    from strategies.risk_budget import INTERVALS_PER_YEAR, max_leverage

    perp = to_panels(data)['perp']
    leverages = {}
    for symbol in perp.columns:
        moves = perp[symbol].pct_change().abs().dropna().to_numpy()
        leverages[symbol] = max_leverage(moves, maintenance_margin, budget,
                                         INTERVALS_PER_YEAR, cap=cap)
    return leverages


def run_carry_book(data: pd.DataFrame, config: CarryBookConfig) -> BookResult:
    """Simulate the book. ``data`` is long format from tools/fetch_multi_funding.py."""
    panels = to_panels(data)
    funding, basis, perp = panels['funding_rate'], panels['basis'], panels['perp']

    # Selection uses the trailing mean of *settled* funding, lagged one interval:
    # ranking on the rate you are about to receive is reading the payment before
    # deciding to be there for it.
    signal = funding.apply(lambda column: trailing_funding(column, config.lookback))

    symbols = list(funding.columns)
    leverages = config.per_symbol_leverage or {}

    def sleeve_leverage(symbol: str) -> float:
        return float(min(leverages.get(symbol, config.leverage), config.max_leverage_cap))

    equity = config.initial_capital
    sleeves: Dict[str, Sleeve] = {}
    curve: List[float] = []
    result = BookResult(equity=pd.Series(dtype=float))

    for i in range(len(funding)):
        timestamp = funding.index[i]

        # ---- accrue on open sleeves: liquidation, basis mark, then funding
        for symbol in list(sleeves):
            sleeve = sleeves[symbol]
            notional = sleeve.capital * sleeve.leverage

            if i > 0:
                move = perp[symbol].iloc[i] / perp[symbol].iloc[i - 1] - 1.0
                adverse = -sleeve.direction * move
                liq_distance = liquidation_distance(sleeve.leverage,
                                                    config.maintenance_margin)
                if np.isfinite(adverse) and adverse >= liq_distance:
                    # Only this sleeve's capital is lost. The book survives.
                    equity -= sleeve.capital
                    result.liquidations += 1
                    result.liquidated_symbols.append(symbol)
                    result.events.append({'timestamp': timestamp, 'symbol': symbol,
                                          'action': 'LIQUIDATED',
                                          'capital_lost': sleeve.capital})
                    del sleeves[symbol]
                    continue

            current_basis = basis[symbol].iloc[i]
            drift = -sleeve.direction * notional * (sleeve.last_basis - current_basis)
            rate = funding[symbol].iloc[i]
            carry = -sleeve.direction * notional * rate

            sleeve.capital += drift + carry
            sleeve.last_basis = current_basis
            equity += drift + carry
            result.basis_pnl += drift
            result.funding_collected += carry
            result.sleeve_intervals += 1

        # ---- close sleeves whose carry has decayed
        for symbol in list(sleeves):
            edge = signal[symbol].iloc[i]
            sleeve = sleeves[symbol]
            still_rich = (np.isfinite(edge)
                          and np.sign(edge) == -sleeve.direction
                          and abs(edge) > config.exit_threshold)
            if not still_rich:
                cost = sleeve.capital * sleeve.leverage * config.fee_per_leg * 2
                sleeve.capital -= cost
                equity -= cost
                result.costs_paid += cost
                result.events.append({'timestamp': timestamp, 'symbol': symbol,
                                      'action': 'EXIT', 'cost': cost})
                del sleeves[symbol]

        # ---- open the richest available sleeves
        open_slots = config.max_sleeves - len(sleeves)
        if open_slots > 0 and equity > 0:
            candidates = []
            for symbol in symbols:
                if symbol in sleeves:
                    continue
                edge = signal[symbol].iloc[i]
                if not np.isfinite(edge) or abs(edge) <= config.entry_threshold:
                    continue
                if edge < 0 and not config.allow_negative_funding:
                    continue
                # Positive funding -> short the perp; negative -> long it.
                candidates.append((abs(edge), symbol,
                                   SHORT_PERP if edge > 0 else LONG_PERP))

            candidates.sort(reverse=True)
            for _, symbol, direction in candidates[:open_slots]:
                leverage = sleeve_leverage(symbol)
                if leverage <= 0:
                    continue
                sleeve_capital = equity / config.max_sleeves
                cost = sleeve_capital * leverage * config.fee_per_leg * 2
                sleeve_capital -= cost
                equity -= cost
                result.costs_paid += cost

                sleeves[symbol] = Sleeve(
                    symbol=symbol, direction=direction, capital=sleeve_capital,
                    entry_basis=basis[symbol].iloc[i],
                    last_basis=basis[symbol].iloc[i], entry_index=i,
                    leverage=leverage)
                result.events.append({'timestamp': timestamp, 'symbol': symbol,
                                      'action': 'ENTER', 'direction': direction,
                                      'cost': cost})

        curve.append(max(equity, 0.0))
        if equity <= 0:
            break

    # Close out so the reported result is realized, not marked.
    for sleeve in sleeves.values():
        cost = sleeve.capital * sleeve.leverage * config.fee_per_leg * 2
        equity -= cost
        result.costs_paid += cost
    if curve:
        curve[-1] = max(equity, 0.0)

    result.equity = pd.Series(curve, index=funding.index[:len(curve)], name='equity')
    return result
