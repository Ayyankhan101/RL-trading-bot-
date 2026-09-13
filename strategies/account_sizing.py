"""
Position sizing for a specific account, in lots.

An edge expressed as "+0.035% per trade" says nothing about whether a given
account can survive trading it. What decides that is the smallest position the
broker will accept, measured against the equity available.

The case that motivates this module: **$20 against a standard 0.01 lot of gold
is 220x leverage.** One ounce at $4,409 is $4,409 of notional; a 0.45% move ends
the account, and the median 15-minute bar is 37% of equity. The same trade on a
cent account - where 0.01 lot is 0.01 oz - is 2.2x leverage and a 45% move to
wipe out. Identical signal, identical broker, completely different outcome, and
the only difference is contract size.

    python -m strategies.account_sizing --equity 20
"""

import argparse
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from strategies.risk_budget import liquidation_distance, ruin_probability

STANDARD_LOT_OZ = 100.0
CENT_LOT_DIVISOR = 100.0


def notional(lots: float, price: float, contract_size: float = STANDARD_LOT_OZ) -> float:
    return lots * contract_size * price


def effective_leverage(equity: float, lots: float, price: float,
                       contract_size: float = STANDARD_LOT_OZ) -> float:
    """Leverage the position actually carries, whatever the broker advertises."""
    if equity <= 0:
        return float('inf')
    return notional(lots, price, contract_size) / equity


def wipeout_move(equity: float, lots: float, price: float,
                 contract_size: float = STANDARD_LOT_OZ) -> float:
    """Adverse move, as a fraction, that takes the account to zero."""
    exposure = notional(lots, price, contract_size)
    if exposure <= 0:
        return float('inf')
    return equity / exposure


def lot_for_risk(equity: float, price: float, stop_pct: float,
                 risk_per_trade: float = 0.02,
                 contract_size: float = STANDARD_LOT_OZ) -> float:
    """
    Lots such that hitting the stop costs ``risk_per_trade`` of equity.

    The standard risk-first sizing rule: choose what a loss may cost, then let
    the stop distance determine the size - never the other way round.
    """
    if stop_pct <= 0 or price <= 0:
        return 0.0
    return (equity * risk_per_trade) / (stop_pct * price * contract_size)


def minimum_viable_equity(price: float, min_lot: float = 0.01,
                          max_leverage: float = 5.0,
                          contract_size: float = STANDARD_LOT_OZ) -> float:
    """
    Equity at which the broker's smallest position stops being reckless.

    Below this, the minimum lot alone exceeds the leverage you would choose, and
    no amount of strategy quality compensates.
    """
    return notional(min_lot, price, contract_size) / max_leverage


def account_table(equity: float, price: float, bar_range: float,
                  lots: Sequence[float] = (0.01,),
                  contract_sizes: Optional[Dict[str, float]] = None) -> List[Dict[str, Any]]:
    """Compare account types at the same nominal lot size."""
    contract_sizes = contract_sizes or {
        'standard (100 oz/lot)': STANDARD_LOT_OZ,
        'cent (1 oz/lot)': STANDARD_LOT_OZ / CENT_LOT_DIVISOR,
    }

    rows = []
    for name, contract in contract_sizes.items():
        for lot in lots:
            exposure = notional(lot, price, contract)
            wipe = wipeout_move(equity, lot, price, contract)
            rows.append({
                'account': name,
                'lots': lot,
                'notional': exposure,
                'leverage': effective_leverage(equity, lot, price, contract),
                'wipeout_move': wipe,
                'bar_pct_of_equity': (bar_range * exposure / equity) if equity else float('inf'),
                'bars_to_zero': (wipe / bar_range) if bar_range > 0 else float('inf'),
            })
    return rows


def format_account_table(rows: List[Dict[str, Any]], equity: float) -> str:
    lines = [
        f"Account of ${equity:,.2f}",
        f"{'account type':24s}{'lots':>7s}{'notional':>12s}{'leverage':>10s}"
        f"{'wipeout':>10s}{'bar % eq':>10s}{'bars to 0':>11s}",
        '-' * 84,
    ]
    for row in rows:
        verdict = '  <-- RECKLESS' if row['leverage'] > 50 else ''
        lines.append(
            f"{row['account']:24s}{row['lots']:>7.3f}${row['notional']:>11,.2f}"
            f"{row['leverage']:>9.1f}x{row['wipeout_move'] * 100:>9.2f}%"
            f"{row['bar_pct_of_equity'] * 100:>9.2f}%{row['bars_to_zero']:>10.1f}{verdict}"
        )
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--equity', type=float, default=20.0)
    parser.add_argument('--price', type=float, default=4408.90)
    parser.add_argument('--bar-range', type=float, default=0.0017,
                        help='Median bar range as a fraction (gold 15m ~0.0017)')
    parser.add_argument('--min-lot', type=float, default=0.01)
    args = parser.parse_args()

    rows = account_table(args.equity, args.price, args.bar_range, lots=(args.min_lot,))
    print(format_account_table(rows, args.equity))

    print(f"\nMinimum equity for the broker's {args.min_lot} lot to be sane:")
    for target in (2.0, 5.0, 10.0, 20.0):
        std = minimum_viable_equity(args.price, args.min_lot, target, STANDARD_LOT_OZ)
        cent = minimum_viable_equity(args.price, args.min_lot, target,
                                     STANDARD_LOT_OZ / CENT_LOT_DIVISOR)
        print(f"  at {target:>4.0f}x leverage   standard ${std:>10,.2f}   cent ${cent:>8,.2f}")

    print("\nRisk-first sizing (size follows the stop, never the reverse):")
    for stop in (0.002, 0.005, 0.01):
        for risk in (0.01, 0.02):
            lots = lot_for_risk(args.equity, args.price, stop, risk)
            print(f"  stop {stop * 100:4.1f}%  risk {risk * 100:3.0f}% of equity"
                  f"  -> {lots:.5f} standard lots "
                  f"({lots * CENT_LOT_DIVISOR:.3f} cent lots)")


if __name__ == "__main__":
    main()
