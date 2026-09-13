"""
Regime detection and adaptive position scaling.

**What can and cannot be adapted to.** A strategy cannot anticipate a central
bank decision or a geopolitical shock; nobody can, and any system claiming to is
either fitting noise or selling something. Measured on 2.4 years of hourly gold:

    autocorrelation of returns      -0.029  (1h)   -0.007  (24h)
    autocorrelation of volatility   +0.989  (1h)   +0.636  (24h)

**Direction is unforecastable. Volatility is highly forecastable.** This is
Engle's volatility clustering (ARCH, Nobel 2003) and it is the single most
reliable regularity in financial time series.

So the adaptation implemented here is deliberately not "predict the news". It is:

1. **Size to forecast volatility.** Exposure falls as expected movement rises, so
   risk per trade stays constant while the market's own scale changes. A shock
   therefore shrinks the position automatically, without predicting the shock.
2. **Classify the regime.** Trend versus range, calm versus stressed, from
   realized volatility and efficiency ratio. A mean-reversion edge belongs in a
   range; a trend regime is when it loses money.
3. **Stand aside for scheduled events.** See ``live/calendar.py`` - the one case
   where the timing of a shock genuinely is known in advance.
4. **Refuse a widened spread.** The cost of a trade is observable before taking
   it, and during a shock it is the thing that changes first and most.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

CALM, NORMAL, STRESSED = 'calm', 'normal', 'stressed'
TREND, RANGE = 'trend', 'range'


@dataclass
class RegimeConfig:
    vol_span: int = 48                  # bars in the volatility estimate
    vol_lookback: int = 500             # bars defining what "normal" vol is
    calm_quantile: float = 0.33
    stressed_quantile: float = 0.80
    efficiency_window: int = 24         # bars for the trend/range test
    efficiency_threshold: float = 0.35
    target_volatility: float = 0.15     # annualized, for position scaling
    max_scale: float = 1.0
    min_scale: float = 0.0
    max_spread_multiple: float = 3.0    # refuse a spread this far above normal

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "RegimeConfig":
        section = dict(config.get('regime', {}) or {})
        return cls(**{k: v for k, v in section.items() if k in cls.__dataclass_fields__})


def realized_volatility(close: pd.Series, span: int = 48,
                        periods_per_year: float = 8760.0) -> pd.Series:
    """
    Ex-ante annualized volatility, lagged one bar.

    Lagged because sizing bar ``t`` with bar ``t``'s own return is lookahead -
    the position has to be set before the move it is sized for.
    """
    returns = close.pct_change()
    vol = returns.ewm(span=span, min_periods=span // 2).std(bias=False)
    return (vol * np.sqrt(periods_per_year)).shift(1)


def efficiency_ratio(close: pd.Series, window: int = 24) -> pd.Series:
    """
    Kaufman's efficiency ratio: net move divided by total path travelled.

    Near 1 the market is trending; near 0 it is oscillating. A mean-reversion
    edge belongs in the second case, and a trending regime is precisely when it
    bleeds.
    """
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    return (net / path.replace(0, np.nan)).shift(1)


def classify(close: pd.Series, config: RegimeConfig,
             periods_per_year: float = 8760.0) -> pd.DataFrame:
    """Label each bar with a volatility regime and a trend/range regime."""
    vol = realized_volatility(close, config.vol_span, periods_per_year)
    efficiency = efficiency_ratio(close, config.efficiency_window)

    # Thresholds from a rolling window, so "normal" adapts instead of being a
    # constant fitted to whatever the sample happened to contain.
    calm_line = vol.rolling(config.vol_lookback, min_periods=50).quantile(config.calm_quantile)
    stress_line = vol.rolling(config.vol_lookback, min_periods=50).quantile(config.stressed_quantile)

    volatility_regime = pd.Series(NORMAL, index=close.index, dtype=object)
    volatility_regime[vol <= calm_line] = CALM
    volatility_regime[vol >= stress_line] = STRESSED

    structure = pd.Series(RANGE, index=close.index, dtype=object)
    structure[efficiency >= config.efficiency_threshold] = TREND

    return pd.DataFrame({
        'volatility': vol,
        'efficiency': efficiency,
        'volatility_regime': volatility_regime,
        'structure': structure,
    })


def volatility_scale(current_vol: float, config: RegimeConfig) -> float:
    """
    ``target_vol / realized_vol``, clamped.

    This is the whole adaptation mechanism: when the market's scale doubles, the
    position halves, and the risk taken per trade stays where it was chosen. No
    forecast of the shock is required - only a measurement of the one in
    progress.
    """
    if not np.isfinite(current_vol) or current_vol <= 1e-9:
        return config.min_scale
    return float(np.clip(config.target_volatility / current_vol,
                         config.min_scale, config.max_scale))


def spread_is_acceptable(current_spread: float, normal_spread: float,
                         config: RegimeConfig) -> bool:
    """
    Whether the quoted cost is close enough to normal to trade through.

    Broker spreads on gold widen by 10-50x through an FOMC print. The cost of a
    trade is observable *before* taking it, which makes this the cheapest
    protection available.
    """
    if normal_spread <= 0:
        return True
    return current_spread <= normal_spread * config.max_spread_multiple


def position_scale(row: pd.Series, config: RegimeConfig,
                   favour: str = RANGE) -> float:
    """
    Final exposure multiplier for one bar, combining volatility and structure.

    A mean-reversion strategy is handed a reduced size when the market is
    trending, because that is the regime in which its edge inverts.
    """
    scale = volatility_scale(row.get('volatility', np.nan), config)

    if row.get('structure') != favour:
        scale *= 0.5

    if row.get('volatility_regime') == STRESSED:
        # A stressed market is not an opportunity to press; it is where gaps
        # jump stops and spreads widen.
        scale *= 0.5

    return float(np.clip(scale, config.min_scale, config.max_scale))
