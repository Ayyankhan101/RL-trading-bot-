"""
Risk-exit rules, shared by the backtest environment and the live paper broker.

This module exists for the same reason ``features/observation.py`` does: two
copies of the execution rules drift, and once they drift the live results stop
measuring what the backtest measured.

Two ideas here:

**ATR-scaled stops.** A fixed 5% stop and 15% target were written for 4-hour
bars. On 15-minute bars a 15% move is weeks away, so the target would never fire
and the "risk management" would be decorative. Expressing the levels as multiples
of the Average True Range at entry makes them scale with both the timeframe and
the current volatility regime, so one config works across intervals.

**Minimum holding period.** At 15m the median bar moves ~0.08% while a round trip
costs ~0.30% in fees and slippage. A policy free to trade every bar pays three
times the available opportunity, 96 times a day. The minimum hold forces
positions out to a horizon where the typical move actually exceeds its own cost.

The minimum hold suppresses *discretionary* exits only. A stop-loss is risk
control, not a trading signal, and must always be allowed to fire.
"""

from typing import Any, Dict, NamedTuple, Optional


class ExitLevels(NamedTuple):
    """Absolute prices at which an open position is closed."""
    stop: Optional[float]
    target: Optional[float]


def min_hold_bars(config: Dict[str, Any]) -> int:
    return int(config.get('risk', {}).get('min_hold_bars', 0) or 0)


def can_exit_on_signal(bars_held: int, config: Dict[str, Any]) -> bool:
    """Whether a discretionary SELL is allowed yet."""
    return bars_held >= min_hold_bars(config)


def stop_and_target(entry_price: float, entry_atr: Optional[float],
                    config: Dict[str, Any]) -> ExitLevels:
    """
    Exit levels for a long position, in absolute prices.

    ``entry_atr`` is the ATR *at the moment of entry* and is frozen for the life
    of the trade: recomputing it each bar would let a position's stop drift away
    as volatility rose, which is exactly when it should not move.

    Falls back to the fixed percentage settings when ATR is unavailable (warm-up
    bars) or when only percentage settings are configured, so existing 4h configs
    keep working unchanged.
    """
    risk = config.get('risk', {})

    stop_atr = risk.get('stop_loss_atr')
    target_atr = risk.get('take_profit_atr')
    stop_pct = risk.get('stop_loss')
    target_pct = risk.get('take_profit')

    stop = None
    target = None

    usable_atr = entry_atr is not None and entry_atr > 0

    if stop_atr and usable_atr:
        distance = float(stop_atr) * float(entry_atr)
        # Clamp so an unusually quiet or wild ATR cannot produce a stop that is
        # either inside the spread or effectively absent.
        distance = _clamp_distance(distance, entry_price, risk, 'stop')
        stop = entry_price - distance
    elif stop_pct:
        stop = entry_price * (1 - float(stop_pct))

    if target_atr and usable_atr:
        distance = float(target_atr) * float(entry_atr)
        distance = _clamp_distance(distance, entry_price, risk, 'target')
        target = entry_price + distance
    elif target_pct:
        target = entry_price * (1 + float(target_pct))

    return ExitLevels(stop=stop, target=target)


def _clamp_distance(distance: float, price: float, risk: Dict[str, Any],
                    kind: str) -> float:
    floor_pct = risk.get(f'min_{kind}_pct')
    ceiling_pct = risk.get(f'max_{kind}_pct')

    if floor_pct:
        distance = max(distance, price * float(floor_pct))
    if ceiling_pct:
        distance = min(distance, price * float(ceiling_pct))
    return distance


def check_exit(high: float, low: float, levels: ExitLevels) -> Optional[tuple]:
    """
    Resolve a bar's range against the exit levels.

    Returns ``(reason, fill_price)`` or None. When both levels sit inside the
    same bar the stop is assumed to hit first: the bar's internal path is
    unknown, and the optimistic reading invents profit out of that ambiguity.
    """
    if levels.stop is not None and low <= levels.stop:
        return 'stop_loss', levels.stop

    if levels.target is not None and high >= levels.target:
        return 'take_profit', levels.target

    return None
