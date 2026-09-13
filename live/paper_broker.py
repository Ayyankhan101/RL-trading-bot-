"""
Paper-trading broker.

Applies the same execution rules as ``environment.trading_env`` - fee, adverse
slippage, capped exposure, minimum order size, stop-loss / take-profit, and a
max-drawdown kill switch - against live prices, and persists its state to disk
so a restart resumes the same account rather than starting a fresh, flattering
one.

No exchange keys, no orders. This simulates fills; it does not place them.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from environment.execution import execution_config, is_maker, taker_fill
from environment.risk import can_exit_on_signal, check_exit, stop_and_target

HOLD, BUY, SELL = 0, 1, 2


@dataclass
class BrokerState:
    balance: float
    btc_held: float = 0.0
    entry_price: Optional[float] = None
    entry_timestamp: Optional[str] = None
    entry_atr: Optional[float] = None
    # Bars elapsed since entry, persisted because the minimum hold belongs to the
    # position - a restart must not reset it and free an early exit.
    bars_held: int = 0
    peak_equity: float = 0.0
    halted: bool = False
    initial_balance: float = 0.0
    total_round_trips: int = 0
    winning_round_trips: int = 0
    last_bar: Optional[str] = None
    started_at: Optional[str] = None
    fills: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


class PaperBroker:
    """A simulated long/flat account driven by live bars."""

    def __init__(self, config: Dict[str, Any], state_path: str):
        self.config = config
        trading = config['trading']
        risk = config.get('risk', {})

        self.initial_balance = float(trading['initial_balance'])
        self.transaction_cost = float(trading['transaction_cost'])
        self.slippage = float(config.get('backtesting', {}).get('slippage', 0.0))
        self.max_position_size = float(trading.get('max_position_size', 0.95))
        self.min_position_size = float(trading.get('min_position_size', 0.02))

        self.max_drawdown_threshold = risk.get('max_drawdown_threshold')

        # Same execution model the backtest uses, so a live fill matches what
        # the backtest claimed it would be.
        self.execution = execution_config(config)
        self.maker = is_maker(config)
        self.transaction_cost = float(
            self.execution['maker_fee'] if self.maker else self.execution['taker_fee'])
        self.slippage = 0.0 if self.maker else float(self.execution['slippage'])

        self.state_path = state_path
        self.state = self._load_state()
        self.round_trips: List[Dict[str, Any]] = []
        self.new_events: List[Dict[str, Any]] = []

    # ------------------------------------------------------------- persistence

    def _load_state(self) -> BrokerState:
        if os.path.exists(self.state_path):
            with open(self.state_path) as file:
                raw = json.load(file)
            return BrokerState(**raw)

        return BrokerState(
            balance=self.initial_balance,
            initial_balance=self.initial_balance,
            peak_equity=self.initial_balance,
        )

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or '.', exist_ok=True)
        with open(self.state_path, 'w') as file:
            json.dump(asdict(self.state), file, indent=2)

    # ------------------------------------------------------------- accounting

    def equity(self, price: float) -> float:
        return self.state.balance + self.state.btc_held * price

    def unrealized_pct(self, price: float) -> float:
        if self.state.btc_held <= 0 or not self.state.entry_price:
            return 0.0
        return price / self.state.entry_price - 1.0

    def drawdown(self, price: float) -> float:
        if self.state.peak_equity <= 0:
            return 0.0
        return (self.equity(price) - self.state.peak_equity) / self.state.peak_equity

    def _fill_price(self, price: float, side: int) -> float:
        """Fill price under the configured execution mode."""
        return taker_fill(side, price, self.config).price if not self.maker else (
            price * (1 - float(self.execution['limit_offset'])) if side == BUY
            else price * (1 + float(self.execution['limit_offset'])))

    # ----------------------------------------------------------------- trading

    def buy(self, price: float, timestamp: str,
            atr: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if self.state.halted:
            return None

        equity = self.equity(price)
        target_notional = equity * self.max_position_size
        gap = target_notional - self.state.btc_held * price
        affordable = self.state.balance / (1 + self.transaction_cost + self.slippage)
        notional = min(gap, affordable)

        if notional <= equity * self.min_position_size or notional <= 0:
            return None

        fill = self._fill_price(price, BUY)
        btc = notional / fill
        fee = notional * self.transaction_cost

        previous_btc = self.state.btc_held
        self.state.btc_held += btc
        self.state.balance -= (notional + fee)
        self.state.fills += 1

        if self.state.entry_price is None or previous_btc <= 0:
            self.state.entry_price = fill
            self.state.entry_timestamp = timestamp
            self.state.entry_atr = float(atr) if atr and atr > 0 else None
            self.state.bars_held = 0
        else:
            self.state.entry_price = (
                (self.state.entry_price * previous_btc) + (fill * btc)
            ) / self.state.btc_held

        event = {
            'timestamp': timestamp, 'action': 'BUY', 'price': fill,
            'amount': btc, 'fee': fee, 'equity': self.equity(price),
        }
        self.new_events.append(event)
        return event

    def sell(self, price: float, timestamp: str, reason: str = 'signal') -> Optional[Dict[str, Any]]:
        if self.state.btc_held <= 0:
            return None

        fill = self._fill_price(price, SELL)
        btc = self.state.btc_held
        proceeds = btc * fill
        fee = proceeds * self.transaction_cost

        entry_price = self.state.entry_price or fill
        entry_fee = entry_price * btc * self.transaction_cost
        # Net of both legs: a trade that only moved sideways is a loss.
        pnl = (fill - entry_price) * btc - fee - entry_fee

        self.state.balance += (proceeds - fee)
        self.state.btc_held = 0.0
        self.state.fills += 1
        self.state.total_round_trips += 1
        if pnl > 0:
            self.state.winning_round_trips += 1

        round_trip = {
            'entry_timestamp': self.state.entry_timestamp,
            'exit_timestamp': timestamp,
            'entry_price': entry_price,
            'exit_price': fill,
            'amount': btc,
            'pnl': pnl,
            'return_pct': (fill / entry_price - 1.0) if entry_price else 0.0,
            'exit_reason': reason,
        }
        self.round_trips.append(round_trip)

        event = {
            'timestamp': timestamp, 'action': 'SELL', 'price': fill,
            'amount': btc, 'fee': fee, 'pnl': pnl, 'reason': reason,
            'equity': self.equity(price),
        }
        self.new_events.append(event)

        self.state.entry_price = None
        self.state.entry_timestamp = None
        self.state.entry_atr = None
        self.state.bars_held = 0
        return event

    def check_risk_exits(self, high: float, low: float, timestamp: str) -> Optional[str]:
        """
        Stop-loss / take-profit against a bar's range.

        Levels come from environment/risk.py - the same module the backtest
        environment uses - so a live exit fires at the price the backtest said it
        would.
        """
        if self.state.btc_held <= 0 or not self.state.entry_price:
            return None

        levels = stop_and_target(self.state.entry_price, self.state.entry_atr, self.config)
        hit = check_exit(high, low, levels)
        if hit is None:
            return None

        reason, fill_price = hit
        self.sell(fill_price, timestamp, reason=reason)
        return reason

    def advance_bar(self) -> None:
        """Count a closed bar against the open position's minimum hold."""
        if self.state.btc_held > 0:
            self.state.bars_held += 1

    def mark(self, price: float, timestamp: str) -> float:
        """Mark to market, update the peak, and trip the kill switch if breached."""
        equity = self.equity(price)
        self.state.peak_equity = max(self.state.peak_equity, equity)

        if (self.max_drawdown_threshold and not self.state.halted
                and self.drawdown(price) <= -float(self.max_drawdown_threshold)):
            self.sell(price, timestamp, reason='drawdown_halt')
            self.state.halted = True
            equity = self.equity(price)

        self.state.last_bar = timestamp
        return equity

    def act(self, action: int, price: float, timestamp: str,
            atr: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if action == BUY:
            return self.buy(price, timestamp, atr=atr)
        if action == SELL and self.state.btc_held > 0:
            # Discretionary exits respect the minimum hold; stops do not, and
            # run in check_risk_exits before the agent is asked to act.
            if can_exit_on_signal(self.state.bars_held, self.config):
                return self.sell(price, timestamp, reason='signal')
        return None

    def summary(self, price: float) -> Dict[str, Any]:
        equity = self.equity(price)
        return {
            'equity': equity,
            'balance': self.state.balance,
            'btc_held': self.state.btc_held,
            'position_value': self.state.btc_held * price,
            'entry_price': self.state.entry_price,
            'unrealized_pct': self.unrealized_pct(price),
            'total_return': equity / self.state.initial_balance - 1.0
            if self.state.initial_balance else 0.0,
            'drawdown': self.drawdown(price),
            'round_trips': self.state.total_round_trips,
            'win_rate': (self.state.winning_round_trips / self.state.total_round_trips)
            if self.state.total_round_trips else 0.0,
            'fills': self.state.fills,
            'bars_held': self.state.bars_held,
            'halted': self.state.halted,
        }
