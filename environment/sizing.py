"""
Position sizing.

The bot previously traded all-in or all-out: every signal moved 95% of equity.
That makes the strategy's realized volatility a function of whatever the market
happened to be doing, and it means a marginal signal is expressed with the same
conviction as a strong one.

**Volatility targeting** - Zhang, Zohren & Roberts, "Deep Reinforcement Learning
for Trading" (Journal of Financial Data Science, 2020); the same scaling is
standard in managed-futures practice (see Moskowitz, Ooi & Pedersen, "Time Series
Momentum", Journal of Financial Economics, 2012, which sizes each position at a
constant ex-ante volatility).

    position_scale = sigma_target / sigma_realized

with ``sigma_realized`` an ex-ante estimate from an exponentially weighted
standard deviation of past returns, annualized. Exposure rises in calm markets
and falls in violent ones, so the equity curve's volatility is roughly constant
instead of tracking the asset's.

**Fractional Kelly** - Kelly, "A New Interpretation of Information Rate" (Bell
System Technical Journal, 1956); Thorp, "The Kelly Criterion in Blackjack, Sports
Betting and the Stock Market" (2006). For a continuous-outcome bet the growth-
optimal fraction is

    f* = mu / sigma^2

on per-period estimates. Full Kelly is famously unusable in practice - it assumes
the estimates are exact, and overestimating ``mu`` leads to ruin - so a fraction
(typically 0.25 to 0.5) is applied, and the result is capped.
"""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def ewma_volatility(returns: pd.Series, span: int = 60,
                    periods_per_year: float = 35040.0) -> pd.Series:
    """
    Ex-ante annualized volatility from an exponentially weighted std.

    Shifted by one bar so the estimate for bar ``t`` uses only bars up to
    ``t-1``: using the current bar's own return to size the current bar's
    position is lookahead.
    """
    vol = returns.ewm(span=span, min_periods=span // 2).std(bias=False)
    return (vol * np.sqrt(periods_per_year)).shift(1)


def volatility_scale(realized_vol: float, target_vol: float,
                     max_leverage: float = 1.0,
                     min_scale: float = 0.0) -> float:
    """
    ``sigma_target / sigma_realized``, clamped.

    A vanishing realized vol would otherwise demand infinite size, which is
    exactly the regime right before a volatility spike.
    """
    if not np.isfinite(realized_vol) or realized_vol <= 1e-9:
        return min_scale

    scale = float(target_vol) / float(realized_vol)
    return float(np.clip(scale, min_scale, max_leverage))


def fractional_kelly(mean_return: float, variance: float, fraction: float = 0.25,
                     cap: float = 1.0) -> float:
    """
    ``fraction * mu / sigma^2``, clipped to [0, cap].

    Negative edge returns 0 rather than a short: this environment is long/flat.
    """
    if variance <= 1e-12 or not np.isfinite(mean_return):
        return 0.0
    if mean_return <= 0:
        return 0.0

    return float(np.clip(fraction * mean_return / variance, 0.0, cap))


class PositionSizer:
    """
    Turns a discrete action into a target exposure fraction of equity.

    Actions map to conviction levels rather than a single all-in switch, and the
    chosen level is then scaled by the volatility target. A 'full' action in a
    violent market can therefore be a smaller position than a 'half' action in a
    calm one - which is the point.
    """

    def __init__(self, config: Dict[str, Any]):
        sizing = config.get('sizing', {}) or {}
        trading = config.get('trading', {})

        self.enabled = bool(sizing.get('volatility_target', False))
        self.target_vol = float(sizing.get('target_annual_volatility', 0.30))
        self.max_position = float(trading.get('max_position_size', 0.95))
        self.min_position = float(trading.get('min_position_size', 0.02))
        self.min_scale = float(sizing.get('min_scale', 0.0))
        self.kelly_fraction = sizing.get('kelly_fraction')
        self.kelly_cap = float(sizing.get('kelly_cap', 1.0))

        # Conviction levels the action space maps onto.
        self.levels = [float(x) for x in sizing.get('levels', [0.0, 1.0])]

    def n_levels(self) -> int:
        return len(self.levels)

    def target_exposure(self, level_index: int, realized_vol: Optional[float],
                        edge: Optional[float] = None,
                        variance: Optional[float] = None) -> float:
        """Exposure for a conviction level referenced by index."""
        level = self.levels[int(np.clip(level_index, 0, len(self.levels) - 1))]
        return self.target_exposure_for_level(level, realized_vol, edge, variance)

    def target_exposure_for_level(self, level: float, realized_vol: Optional[float],
                                  edge: Optional[float] = None,
                                  variance: Optional[float] = None) -> float:
        """
        Exposure for a conviction level in [0, 1], as a fraction of equity.

        Args:
            level: conviction, where 1.0 means the configured maximum position.
            realized_vol: ex-ante annualized volatility for this bar.
            edge / variance: optional per-period estimates for a Kelly cap.
        """
        if level <= 0:
            return 0.0

        exposure = level * self.max_position

        if self.enabled and realized_vol is not None:
            exposure *= volatility_scale(realized_vol, self.target_vol,
                                         max_leverage=1.0, min_scale=self.min_scale)

        if self.kelly_fraction and edge is not None and variance is not None:
            kelly = fractional_kelly(edge, variance, float(self.kelly_fraction),
                                     self.kelly_cap)
            exposure = min(exposure, kelly * self.max_position)

        if exposure < self.min_position:
            return 0.0
        return float(min(exposure, self.max_position))
