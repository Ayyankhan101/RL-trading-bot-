"""
Fill models, shared by the backtest environment and the live paper broker.

Switching from market orders to limit orders cuts the round-trip cost from about
0.30% to 0.06% on Binance - which matters enormously here, because the measured
mean-reversion edge is 0.01% to 0.08% per trade. But a limit order only fills if
the market trades through it. Modeling the cheaper fee **without** the missed
fills would manufacture the entire improvement out of nothing, so this module
models both.

* ``taker``  - a market order at the reference price, moved against the agent by
  ``slippage``. Always fills.
* ``maker``  - a limit resting ``limit_offset`` inside the spread. It fills on a
  later bar only if that bar's range reaches the limit price, at the limit price
  (a resting order has no adverse slippage). If no bar reaches it within
  ``max_wait_bars``, the order is cancelled and the decision is lost.
* ``cfd``    - a broker quote: buy at the ask, sell at the bid, plus a
  round-turn commission per lot and overnight financing. Used for gold, where
  the cost structure is completely different from crypto and is the reason the
  strategy is viable there at all.

**Why CFD costs need their own mode.** A crypto fee is a percentage of notional,
so it scales with position size and cancels out of any return calculation. A
gold spread is quoted in dollars per ounce - it does *not* scale - and the
commission is per lot, not per dollar. Modelling gold with a percentage fee
would misprice every trade, in a direction that depends on position size.

The miss rate is the honest price of maker execution, and
``tests/test_execution.py`` asserts it is non-zero - a fill model that never
misses is wrong.
"""

from typing import Any, Dict, NamedTuple, Optional

BUY, SELL = 1, 2


class Fill(NamedTuple):
    """A completed fill."""
    price: float
    fee_rate: float
    bars_waited: int
    maker: bool


class PendingOrder(NamedTuple):
    """A resting limit order awaiting a bar that reaches its price."""
    side: int
    limit_price: float
    placed_step: int
    fee_rate: float


def execution_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Execution settings, falling back to the legacy flat cost."""
    execution = dict(config.get('execution', {}) or {})
    legacy_fee = float(config.get('trading', {}).get('transaction_cost', 0.001))
    legacy_slippage = float(config.get('backtesting', {}).get('slippage', 0.0))

    execution.setdefault('mode', 'taker')
    execution.setdefault('taker_fee', legacy_fee)
    execution.setdefault('maker_fee', legacy_fee)
    execution.setdefault('slippage', legacy_slippage)
    execution.setdefault('limit_offset', 0.0005)
    execution.setdefault('max_wait_bars', 4)
    execution.setdefault('fallback_to_taker', False)

    # CFD settings (gold and other broker-quoted instruments)
    execution.setdefault('spread_points', 0.15)       # price units, e.g. $/oz
    execution.setdefault('commission_per_lot', 7.0)   # round turn per standard lot
    execution.setdefault('contract_size', 100.0)      # units per standard lot
    execution.setdefault('swap_long_daily', -0.00005)  # fraction of notional per night
    execution.setdefault('swap_short_daily', -0.00002)
    return execution


def is_cfd(config: Dict[str, Any]) -> bool:
    return str(execution_config(config).get('mode', 'taker')).lower() == 'cfd'


def cfd_fill(side: int, mid_price: float, config: Dict[str, Any]) -> Fill:
    """
    Broker quote: buy at the ask, sell at the bid.

    The spread is a **price offset**, not a percentage, so it is added once in
    price terms. The commission is per standard lot, converted to a fraction of
    notional here so the rest of the pipeline can keep treating fees as rates.
    """
    execution = execution_config(config)
    half_spread = float(execution['spread_points']) / 2.0

    price = mid_price + half_spread if side == BUY else mid_price - half_spread

    contract_size = float(execution['contract_size'])
    per_side_commission = float(execution['commission_per_lot']) / 2.0
    notional_per_lot = contract_size * mid_price
    fee_rate = (per_side_commission / notional_per_lot) if notional_per_lot > 0 else 0.0

    return Fill(price=price, fee_rate=fee_rate, bars_waited=0, maker=False)


def financing_cost(notional: float, bar_minutes: float, is_long: bool,
                   config: Dict[str, Any]) -> float:
    """
    Overnight financing, pro-rated per bar.

    A three-hour hold rarely crosses a rollover, but a strategy that drifts
    longer will pay this every night, and omitting it flatters every result that
    holds positions. Pro-rating per bar spreads the charge smoothly rather than
    spiking it at an arbitrary hour.
    """
    execution = execution_config(config)
    daily = float(execution['swap_long_daily'] if is_long else execution['swap_short_daily'])
    return notional * daily * (bar_minutes / 1440.0)


def is_maker(config: Dict[str, Any]) -> bool:
    return str(execution_config(config).get('mode', 'taker')).lower() == 'maker'


def taker_fill(side: int, reference_price: float, config: Dict[str, Any]) -> Fill:
    """Immediate fill under the configured mode."""
    execution = execution_config(config)

    if str(execution.get('mode', 'taker')).lower() == 'cfd':
        return cfd_fill(side, reference_price, config)

    slippage = float(execution['slippage'])

    price = (reference_price * (1 + slippage) if side == BUY
             else reference_price * (1 - slippage))

    return Fill(price=price, fee_rate=float(execution['taker_fee']),
                bars_waited=0, maker=False)


def place_limit(side: int, reference_price: float, step: int,
                config: Dict[str, Any]) -> PendingOrder:
    """
    Post a limit ``limit_offset`` on the passive side of the reference price.

    A buy rests *below* the last price and a sell *above* it: that is what earns
    the maker rebate, and it is also why the order may never fill.
    """
    execution = execution_config(config)
    offset = float(execution['limit_offset'])

    limit = (reference_price * (1 - offset) if side == BUY
             else reference_price * (1 + offset))

    return PendingOrder(side=side, limit_price=limit, placed_step=step,
                        fee_rate=float(execution['maker_fee']))


def try_fill_limit(order: PendingOrder, bar_high: float, bar_low: float,
                   step: int) -> Optional[Fill]:
    """
    Fill a resting order if this bar's range reaches it.

    Returns the fill at the **limit price**, not the bar close: a resting order
    that gets hit trades at its own price.
    """
    if order.side == BUY and bar_low <= order.limit_price:
        return Fill(price=order.limit_price, fee_rate=order.fee_rate,
                    bars_waited=step - order.placed_step, maker=True)

    if order.side == SELL and bar_high >= order.limit_price:
        return Fill(price=order.limit_price, fee_rate=order.fee_rate,
                    bars_waited=step - order.placed_step, maker=True)

    return None


def expired(order: PendingOrder, step: int, config: Dict[str, Any]) -> bool:
    """Whether a resting order has waited past ``max_wait_bars``."""
    return (step - order.placed_step) >= int(execution_config(config)['max_wait_bars'])
