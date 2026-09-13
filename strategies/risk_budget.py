"""
Leverage sizing from ruin probability.

Carry strategies are killed by how they are sized, not by their signal. The
funding book in this repo has a weekly Sharpe above 11, and any Sharpe-based
rule - Kelly included - would read that as a licence for enormous leverage.
It is not, because the standard deviation of a carry return series does not
contain the thing that actually ends it: a jump past the liquidation price.

So leverage is solved from an explicit **ruin budget** against the *empirical*
distribution of adverse moves. No Gaussian fit: the left tail is exactly where a
normal approximation is most wrong, and it is the only part that matters here.

The arithmetic that motivates this module, measured on 1,997 real 8-hour BTC
intervals:

| Target/week | Leverage | Liquidation move | P(ruin)/week |
|---|---|---|---|
| 0.5% | 7x | 13.30% | 0.00% |
| 2.0% | 29x | 2.95% | 29.5% |
| 4.5% | 65x | 1.03% | 96.8% |

A 1.03% adverse move is 0.79 standard deviations of an 8-hour BTC return. High
leverage on a high-Sharpe carry book is not an aggressive strategy; it is a
near-certain loss on a short horizon.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

INTERVALS_PER_WEEK = 21
INTERVALS_PER_YEAR = 3 * 365


def liquidation_distance(leverage: float, maintenance_margin: float = 0.005) -> float:
    """
    Adverse move that wipes the margin on a leveraged leg.

    At ``L`` times leverage the position is gone once price moves about ``1/L``
    against it, less the maintenance requirement.
    """
    if leverage <= 0:
        return float('inf')
    return max(0.0, (1.0 / leverage) - maintenance_margin)


def ruin_probability(adverse_returns: Sequence[float], leverage: float,
                     maintenance_margin: float = 0.005,
                     horizon_intervals: int = INTERVALS_PER_YEAR) -> float:
    """
    Probability of at least one liquidation over ``horizon_intervals``.

    ``adverse_returns`` are per-interval moves in the direction that hurts the
    position (for a short leg, positive returns). The per-interval hit rate is
    read straight off the empirical sample, then compounded over the horizon.
    """
    moves = np.asarray(adverse_returns, dtype=np.float64)
    moves = moves[np.isfinite(moves)]
    if len(moves) == 0 or leverage <= 0:
        return 0.0

    distance = liquidation_distance(leverage, maintenance_margin)
    per_interval = float((moves >= distance).mean())

    if per_interval <= 0.0:
        return 0.0
    if per_interval >= 1.0:
        return 1.0

    return float(1.0 - (1.0 - per_interval) ** horizon_intervals)


def max_leverage(adverse_returns: Sequence[float], maintenance_margin: float = 0.005,
                 budget: float = 0.01, horizon_intervals: int = INTERVALS_PER_YEAR,
                 cap: float = 20.0, resolution: float = 0.1) -> float:
    """
    Largest leverage whose ruin probability over the horizon stays under budget.

    Searched rather than solved in closed form, because the empirical hit rate
    is a step function of leverage - it only changes when the liquidation
    distance crosses an observed move.
    """
    if budget <= 0:
        return 0.0

    best = 0.0
    steps = int(cap / resolution)
    for i in range(1, steps + 1):
        candidate = i * resolution
        if ruin_probability(adverse_returns, candidate, maintenance_margin,
                            horizon_intervals) <= budget:
            best = candidate
        else:
            break                 # ruin rises monotonically with leverage
    return best


def survival_intervals(adverse_returns: Sequence[float], leverage: float,
                       maintenance_margin: float = 0.005) -> float:
    """Expected number of intervals before liquidation, or inf if never observed."""
    moves = np.asarray(adverse_returns, dtype=np.float64)
    moves = moves[np.isfinite(moves)]
    distance = liquidation_distance(leverage, maintenance_margin)
    rate = float((moves >= distance).mean()) if len(moves) else 0.0
    return float('inf') if rate <= 0 else 1.0 / rate


def leverage_for_target(weekly_target: float, weekly_edge: float) -> float:
    """Leverage a weekly return target implies, given the unlevered edge."""
    if weekly_edge <= 0:
        return float('inf')
    return weekly_target / weekly_edge


def ruin_table(adverse_returns: Sequence[float], weekly_edge: float,
               targets: Optional[Sequence[float]] = None,
               maintenance_margin: float = 0.005) -> List[Dict[str, Any]]:
    """
    What each weekly return target costs in ruin probability.

    This is the table that answers "can we make 4-5% a week": not with an
    opinion, but with the leverage it requires and the survival that buys.
    """
    targets = targets or (0.005, 0.01, 0.02, 0.04, 0.045)
    rows = []

    for target in targets:
        leverage = leverage_for_target(target, weekly_edge)
        intervals = survival_intervals(adverse_returns, leverage, maintenance_margin)

        rows.append({
            'weekly_target': float(target),
            'leverage': float(leverage),
            'liquidation_move': liquidation_distance(leverage, maintenance_margin),
            'ruin_per_week': ruin_probability(adverse_returns, leverage,
                                              maintenance_margin, INTERVALS_PER_WEEK),
            'ruin_per_year': ruin_probability(adverse_returns, leverage,
                                              maintenance_margin, INTERVALS_PER_YEAR),
            'survival_days': (intervals / 3.0) if np.isfinite(intervals) else float('inf'),
        })

    return rows


def format_ruin_table(rows: List[Dict[str, Any]]) -> str:
    """Render the ruin table for a CLI banner."""
    lines = [
        f"{'target/wk':>10s}{'leverage':>10s}{'liq move':>10s}"
        f"{'P(ruin)/wk':>12s}{'P(ruin)/yr':>12s}{'survival':>13s}",
        '-' * 67,
    ]
    for row in rows:
        survival = (f"{row['survival_days']:.1f} days"
                    if np.isfinite(row['survival_days']) and row['survival_days'] < 3650
                    else '>10 years')
        lines.append(
            f"{row['weekly_target'] * 100:>9.1f}%{row['leverage']:>9.0f}x"
            f"{row['liquidation_move'] * 100:>9.2f}%"
            f"{row['ruin_per_week'] * 100:>11.2f}%{row['ruin_per_year'] * 100:>11.2f}%"
            f"{survival:>13s}"
        )
    return '\n'.join(lines)
