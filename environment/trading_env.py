"""
Bitcoin Trading Environment for Reinforcement Learning.

A plain Python environment (no gym dependency) with a gymnasium-style API:
``reset(seed=None) -> (obs, info)`` and ``step(action) -> (obs, reward,
terminated, truncated, info)``.

Execution model, stated explicitly because it is what makes the reported numbers
mean anything:

* The agent observes bar ``t`` after it has closed and acts on that information.
* The order fills at bar ``t``'s close, adjusted by slippage against the agent.
* Stop-loss and take-profit are checked on bar ``t+1``'s high/low before the
  agent acts again, so an exit cannot use information the bar had not produced.
* Breaching the max-drawdown threshold flattens the book and blocks further
  trading, but the episode keeps running to the end of the data. Halting the
  clock as well would leave the equity curve covering fewer bars than the
  benchmarks it is compared against.
* Nothing in the observation, the reward, or the risk checks reads a row with an
  index greater than the current step.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from agents.rewards import build_reward
from environment.execution import (
    execution_config,
    expired,
    financing_cost,
    is_cfd,
    is_maker,
    place_limit,
    taker_fill,
    try_fill_limit,
)
from environment.risk import can_exit_on_signal, check_exit, stop_and_target
from environment.sizing import PositionSizer, ewma_volatility
from features.observation import (
    PORTFOLIO_FEATURES,
    build_market_features,
    build_portfolio_features,
)
from strategies.regime import RegimeConfig, classify, position_scale

HOLD, BUY, SELL = 0, 1, 2
ACTION_NAMES = {HOLD: 'HOLD', BUY: 'BUY', SELL: 'SELL'}


class BitcoinTradingEnv:
    """
    Long/flat Bitcoin trading environment.

    Actions:
        0: Hold      - leave the position as it is
        1: Buy       - enter (or top up to) the target long position
        2: Sell      - close the position entirely

    Observation: scale-free market features for the current bar plus the agent's
    own portfolio state. Every market feature is a ratio or a bounded oscillator,
    so the same policy stays meaningful whether BTC trades at 6k or 118k.
    """

    def __init__(self, data: pd.DataFrame, config: Optional[Dict[str, Any]] = None,
                 config_path: str = "config.yaml"):
        if config is None:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
        self.config = config

        trading = self.config['trading']
        risk = self.config.get('risk', {})

        self.initial_balance = float(trading['initial_balance'])
        self.transaction_cost = float(trading['transaction_cost'])
        self.slippage = float(self.config.get('backtesting', {}).get('slippage', 0.0))
        self.max_position_size = float(trading.get('max_position_size', 0.95))
        self.min_position_size = float(trading.get('min_position_size', 0.01))

        self.max_drawdown_threshold = risk.get('max_drawdown_threshold')

        self.data = data.copy()
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise TypeError("Environment data must be indexed by timestamp")
        if not self.data.index.is_monotonic_increasing:
            raise ValueError("Environment data must be sorted oldest-first")

        self.execution = execution_config(self.config)
        self.maker = is_maker(self.config)
        self.cfd = is_cfd(self.config)
        self.sizer = PositionSizer(self.config)
        self.reward_fn = build_reward(self.config)

        # Regime adaptation: exposure follows measured volatility and market
        # structure. Direction is unforecastable; volatility is not.
        self.regime_config = RegimeConfig.from_dict(self.config)
        self.regime_enabled = bool(self.config.get('regime', {}).get('enabled', False))

        self.feature_names, self._features = self._build_features(self.data)
        self._close = self.data['close'].to_numpy(dtype=np.float64)
        self._high = self.data['high'].to_numpy(dtype=np.float64)
        self._low = self.data['low'].to_numpy(dtype=np.float64)
        # ATR drives the stop distance; absent (or during warm-up) the risk
        # module falls back to the fixed percentage settings.
        self._atr = (self.data['ATR'].to_numpy(dtype=np.float64)
                     if 'ATR' in self.data.columns else None)

        # Ex-ante volatility for position sizing, shifted so bar t is sized using
        # information available before bar t.
        from utils.data_utils import infer_periods_per_year
        self.periods_per_year = infer_periods_per_year(self.data.index)
        self._bar_minutes = (365 * 24 * 60) / max(self.periods_per_year, 1e-9)
        returns = self.data['close'].pct_change()
        self._vol = ewma_volatility(
            returns,
            span=int(self.config.get('sizing', {}).get('vol_span', 60)),
            periods_per_year=self.periods_per_year,
        ).to_numpy(dtype=np.float64)
        self._returns = returns.to_numpy(dtype=np.float64)

        if self.regime_enabled:
            regimes = classify(self.data['close'], self.regime_config,
                               self.periods_per_year)
            self._regime_scale = regimes.apply(
                lambda row: position_scale(row, self.regime_config), axis=1
            ).to_numpy(dtype=np.float64)
            self._regime_labels = regimes['volatility_regime'].to_numpy()
        else:
            self._regime_scale = None
            self._regime_labels = None

        # Bars where a scheduled high-impact event makes the spread unreliable.
        self._blackout = np.zeros(len(self.data), dtype=bool)

        # First and last bar the agent may act on. Both are indices, not counts:
        # conflating the two is what previously left the final bars untraded.
        self.start_index = 0
        self.last_index = len(self.data) - 1
        if self.last_index <= self.start_index:
            raise ValueError(f"Not enough data to run an episode ({len(self.data)} rows)")

        self.n_features = len(self.feature_names) + len(self._portfolio_features())

        # Action table. The original three actions keep their meaning so existing
        # policies and tests stay valid; extra conviction levels are appended,
        # letting the agent ask for a half position instead of only all-or-nothing.
        #
        #   0 = HOLD          leave the position alone
        #   1 = BUY           target the largest configured level
        #   2 = SELL          go flat
        #   3+ = intermediate conviction levels, largest first
        positive_levels = sorted((lv for lv in self.sizer.levels if lv > 0), reverse=True)
        if not positive_levels:
            positive_levels = [1.0]
        self.action_levels = positive_levels
        self.n_actions = 3 + max(0, len(positive_levels) - 1)

        self._rng = np.random.default_rng()
        self.reset()

    # ------------------------------------------------------------------ setup

    def _build_features(self, data: pd.DataFrame) -> Tuple[List[str], np.ndarray]:
        """
        Build the market feature matrix once, up front.

        Delegated to features.observation so the live trader constructs its
        observations with the exact same code the agent trained against.
        Recomputing per step would also be slow and easy to get wrong; a
        materialized matrix means the step loop can only read row
        ``current_step``.
        """
        return build_market_features(data)

    @staticmethod
    def _portfolio_features() -> List[str]:
        return list(PORTFOLIO_FEATURES)

    # ------------------------------------------------------------- env API

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset to the first tradable bar."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.current_step = self.start_index
        self.balance = self.initial_balance
        self.btc_held = 0.0
        self.entry_price: Optional[float] = None
        self.entry_step: Optional[int] = None
        self.entry_atr: Optional[float] = None
        self.halted = False
        self.pending_order = None
        self.pending_target: Optional[float] = None
        self.missed_orders = 0
        self.maker_fills = 0
        self.financing_paid = 0.0

        self.total_trades = 0          # round trips, not order count
        self.winning_trades = 0
        self.open_orders = 0           # individual fills, for turnover reporting

        self.peak_value = self.initial_balance
        self.portfolio_values = [self.initial_balance]
        self.timestamps = [self.data.index[self.current_step]]
        self.trades: List[Dict[str, Any]] = []
        self.round_trips: List[Dict[str, Any]] = []
        self.rewards_history: List[float] = []

        for component in ('dsr', 'cost_aware', 'log_return'):
            if component in self.reward_fn:
                self.reward_fn[component].reset()

        return self._get_observation(), self._info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance one bar."""
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Invalid action: {action} (expected 0..{self.n_actions - 1})")

        prev_value = self._portfolio_value(self._close[self.current_step])

        if not self.halted:
            self._execute_action(action)

        # Advance, then resolve any resting limit order against the new bar,
        # then apply the risk stops. Order matters: a limit posted last bar can
        # only fill on a bar that has since printed.
        self.current_step += 1
        self._resolve_pending_order()
        exit_reason = self._check_risk_exits()

        current_price = self._close[self.current_step]

        # Overnight financing on a held CFD position, pro-rated per bar. A
        # strategy that drifts into multi-day holds pays this every night, and
        # omitting it would flatter every such result.
        if self.cfd and self.btc_held > 0:
            charge = financing_cost(self.btc_held * current_price,
                                    self._bar_minutes, True, self.config)
            self.balance += charge
            self.financing_paid += charge

        current_value = self._portfolio_value(current_price)
        self.peak_value = max(self.peak_value, current_value)

        if self._breached_drawdown(current_value) and not self.halted:
            self._close_position(current_price, reason='drawdown_halt')
            self.halted = True
            current_value = self._portfolio_value(current_price)

        reward = self._calculate_reward(prev_value, current_value, action)

        self.portfolio_values.append(current_value)
        self.timestamps.append(self.data.index[self.current_step])
        self.rewards_history.append(reward)

        # A halted agent sits in cash for the remaining bars rather than ending
        # the episode: the flat stretch is part of the strategy's record, and
        # dropping it would silently shorten the evaluation window.
        terminated = False
        truncated = self.current_step >= self.last_index

        if (terminated or truncated) and self.btc_held > 0:
            # Mark-to-market is not a result. Close out so the reported P&L is
            # what the strategy would actually have realized.
            self._close_position(current_price, reason='episode_end')
            self.portfolio_values[-1] = self._portfolio_value(current_price)

        info = self._info()
        if exit_reason:
            info['exit_reason'] = exit_reason

        return self._get_observation(), reward, terminated, truncated, info

    # --------------------------------------------------------------- trading

    def _fill_price(self, price: float, side: int) -> float:
        """Taker fill price: slippage always works against the agent."""
        return taker_fill(side, price, self.config).price

    def _resolve_pending_order(self) -> None:
        """
        Try to fill a resting limit against the current bar.

        A limit that never gets hit is cancelled and the decision is lost - that
        missed trade is the real cost of maker execution, and it is counted.
        """
        if self.pending_order is None:
            return

        fill = try_fill_limit(self.pending_order,
                              self._high[self.current_step],
                              self._low[self.current_step],
                              self.current_step)

        if fill is not None:
            self.maker_fills += 1
            if self.pending_order.side == BUY:
                self._settle_buy(fill.price, fill.fee_rate, self.pending_target)
            else:
                self._settle_sell(fill.price, fill.fee_rate, reason='signal')
            self.pending_order = None
            self.pending_target = None
            return

        if expired(self.pending_order, self.current_step, self.config):
            self.missed_orders += 1
            if self.execution.get('fallback_to_taker'):
                price = self._close[self.current_step]
                fill = taker_fill(self.pending_order.side, price, self.config)
                if self.pending_order.side == BUY:
                    self._settle_buy(fill.price, fill.fee_rate, self.pending_target)
                else:
                    self._settle_sell(fill.price, fill.fee_rate, reason='signal')
            self.pending_order = None
            self.pending_target = None

    def target_exposure(self, action: int) -> Optional[float]:
        """
        Exposure this action asks for, as a fraction of equity.

        Returns None for HOLD, which is "change nothing" rather than a target of
        zero. The level is scaled by the volatility target, so the same
        conviction buys a smaller position when the market is violent.
        """
        if action == HOLD:
            return None
        if action == SELL:
            return 0.0

        # A scheduled event window: the spread is unreliable, so take nothing new.
        if self._blackout[self.current_step]:
            return 0.0 if self.btc_held > 0 else None

        level = self.action_levels[0] if action == BUY else \
            self.action_levels[min(action - 2, len(self.action_levels) - 1)]

        vol = None
        if self._vol is not None:
            candidate = self._vol[self.current_step]
            vol = float(candidate) if np.isfinite(candidate) else None

        exposure = self.sizer.target_exposure_for_level(level, vol)

        if self._regime_scale is not None:
            scale = self._regime_scale[self.current_step]
            exposure *= float(scale) if np.isfinite(scale) else 0.0

        return exposure

    def _execute_action(self, action: int) -> None:
        price = self._close[self.current_step]
        target = self.target_exposure(action)
        if target is None:                      # HOLD
            return

        equity = self._portfolio_value(price)
        current = (self.btc_held * price / equity) if equity > 0 else 0.0

        if target <= 0:
            if self.btc_held > 0 and can_exit_on_signal(self._bars_held(), self.config):
                # A discretionary exit inside the minimum hold is ignored. Stops
                # are unaffected - they run before the agent acts.
                self._close_position(price, reason='signal')
            return

        # Rebalance deadband. Volatility-scaled exposure moves continuously, and
        # without a deadband every small drift in the target triggers a trade.
        # Measured cost of omitting it: trade count went 160 -> 624 and profit
        # factor fell from 1.83 to 0.90 - the rebalancing churn was larger than
        # the risk reduction it bought.
        deadband = float(self.config.get('sizing', {}).get('rebalance_deadband', 0.0))
        if self.btc_held > 0 and deadband > 0 and abs(target - current) < deadband:
            return

        if target > current:
            self._open_position(price, target_fraction=target)
        elif self.btc_held > 0 and current - target > max(self.min_position_size, deadband):
            # Reducing exposure is still an exit decision; the hold applies.
            if can_exit_on_signal(self._bars_held(), self.config):
                self._reduce_position(price, target_fraction=target)

    def _open_position(self, price: float, target_fraction: Optional[float] = None) -> None:
        """
        Buy toward a target exposure.

        In maker mode this posts a limit and returns: the position only exists if
        a later bar trades through it. In taker mode it crosses the spread and
        settles immediately.
        """
        fraction = self.max_position_size if target_fraction is None else target_fraction

        if self.maker:
            if self.pending_order is not None:
                return                      # one resting order at a time
            self.pending_order = place_limit(BUY, price, self.current_step, self.config)
            self.pending_target = fraction
            return

        fill = taker_fill(BUY, price, self.config)
        self._settle_buy(fill.price, fill.fee_rate, fraction)

    def _settle_buy(self, fill: float, fee_rate: float,
                    target_fraction: Optional[float]) -> None:
        """Book a buy that actually filled, at ``fill``."""
        price = self._close[self.current_step]
        equity = self._portfolio_value(price)
        fraction = self.max_position_size if target_fraction is None else target_fraction
        target_notional = equity * fraction
        current_notional = self.btc_held * price
        gap = target_notional - current_notional

        affordable = self.balance / (1 + fee_rate)
        notional = min(gap, affordable)

        # Reject dust orders. Without this floor the agent re-buys every bar with
        # its leftover cash and reports thousands of phantom trades.
        if notional <= equity * self.min_position_size or notional <= 0:
            return

        btc = notional / fill
        fee = notional * fee_rate

        self.btc_held += btc
        self.balance -= (notional + fee)
        self.open_orders += 1

        if self.entry_price is None:
            self.entry_price = fill
            self.entry_step = self.current_step
            self.entry_atr = (float(self._atr[self.current_step])
                              if self._atr is not None
                              and np.isfinite(self._atr[self.current_step]) else None)
        else:
            # Weighted average entry across top-ups.
            prev_btc = self.btc_held - btc
            self.entry_price = ((self.entry_price * prev_btc) + (fill * btc)) / self.btc_held

        self.trades.append({
            'timestamp': self.data.index[self.current_step],
            'step': self.current_step,
            'action': 'BUY',
            'price': fill,
            'amount': btc,
            'fee': fee,
            'portfolio_value': self._portfolio_value(price),
        })

    def _reduce_position(self, price: float, target_fraction: float) -> None:
        """
        Sell down to a target exposure without closing the position.

        Partial exits keep the entry price and the ATR level intact: the
        remaining shares are the same trade, so their stop must not move.
        """
        equity = self._portfolio_value(price)
        target_notional = equity * target_fraction
        excess_notional = (self.btc_held * price) - target_notional

        if excess_notional <= equity * self.min_position_size:
            return

        fill = self._fill_price(price, SELL)
        btc = min(self.btc_held, excess_notional / fill)
        proceeds = btc * fill
        fee = proceeds * self.transaction_cost

        self.balance += (proceeds - fee)
        self.btc_held -= btc
        self.open_orders += 1

        # Book the realized portion as its own round trip. Without this the
        # trade ledger no longer explains the equity curve, and win rate and
        # profit factor quietly stop counting part of the P&L.
        entry_price = self.entry_price or fill
        entry_fee = entry_price * btc * self.transaction_cost
        pnl = (fill - entry_price) * btc - fee - entry_fee

        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1

        self.round_trips.append({
            'entry_timestamp': self.data.index[self.entry_step] if self.entry_step is not None else None,
            'exit_timestamp': self.data.index[self.current_step],
            'entry_price': entry_price,
            'exit_price': fill,
            'amount': btc,
            'pnl': pnl,
            'return_pct': (fill / entry_price - 1.0) if entry_price else 0.0,
            'bars_held': self._bars_held(),
            'exit_reason': 'reduce',
        })

        self.trades.append({
            'timestamp': self.data.index[self.current_step],
            'step': self.current_step,
            'action': 'REDUCE',
            'pnl': pnl,
            'price': fill,
            'amount': btc,
            'fee': fee,
            'portfolio_value': self._portfolio_value(price),
        })

        if self.btc_held * price < equity * self.min_position_size:
            self._close_position(price, reason='reduce_to_flat')

    def _close_position(self, price: float, reason: str = 'signal') -> None:
        """
        Exit the position.

        Discretionary exits rest as limits in maker mode. Risk exits - stops,
        targets, the drawdown halt and the episode close - always cross the
        spread: a stop that waits for a better price is not a stop.
        """
        if self.btc_held <= 0:
            return

        if self.maker and reason == 'signal':
            if self.pending_order is None:
                self.pending_order = place_limit(SELL, price, self.current_step,
                                                 self.config)
                self.pending_target = None
            return

        fill = taker_fill(SELL, price, self.config)
        self._settle_sell(fill.price, fill.fee_rate, reason=reason)

    def _settle_sell(self, fill: float, fee_rate: float, reason: str = 'signal') -> None:
        """Book a sale that actually filled, at ``fill``."""
        if self.btc_held <= 0:
            return

        price = self._close[self.current_step]
        btc = self.btc_held
        proceeds = btc * fill
        fee = proceeds * fee_rate

        self.balance += (proceeds - fee)
        self.btc_held = 0.0

        entry_price = self.entry_price if self.entry_price is not None else fill
        entry_fee = entry_price * btc * fee_rate
        # Net of both legs' fees: a "win" has to actually clear its costs.
        pnl = (fill - entry_price) * btc - fee - entry_fee

        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1

        self.round_trips.append({
            'entry_timestamp': self.data.index[self.entry_step] if self.entry_step is not None else None,
            'exit_timestamp': self.data.index[self.current_step],
            'entry_price': entry_price,
            'exit_price': fill,
            'amount': btc,
            'pnl': pnl,
            'return_pct': (fill / entry_price - 1.0) if entry_price else 0.0,
            'bars_held': (self.current_step - self.entry_step) if self.entry_step is not None else 0,
            'exit_reason': reason,
        })

        self.trades.append({
            'timestamp': self.data.index[self.current_step],
            'step': self.current_step,
            'action': 'SELL',
            'price': fill,
            'amount': btc,
            'fee': fee,
            'pnl': pnl,
            'exit_reason': reason,
            'portfolio_value': self._portfolio_value(price),
        })

        self.entry_price = None
        self.entry_step = None
        self.entry_atr = None

    def _bars_held(self) -> int:
        if self.entry_step is None:
            return 0
        return self.current_step - self.entry_step

    def _check_risk_exits(self) -> Optional[str]:
        """
        Apply stop-loss / take-profit against the current bar's range.

        Levels come from environment/risk.py, which the live broker also uses, so
        the two cannot drift apart. They are ATR multiples frozen at entry, which
        keeps them meaningful whether a bar is 15 minutes or 4 hours.
        """
        if self.btc_held <= 0 or self.entry_price is None:
            return None

        levels = stop_and_target(self.entry_price, self.entry_atr, self.config)
        hit = check_exit(self._high[self.current_step],
                         self._low[self.current_step], levels)
        if hit is None:
            return None

        reason, fill_price = hit
        self._close_position(fill_price, reason=reason)
        return reason

    def _breached_drawdown(self, current_value: float) -> bool:
        if not self.max_drawdown_threshold or self.peak_value <= 0:
            return False
        drawdown = (current_value - self.peak_value) / self.peak_value
        return drawdown <= -float(self.max_drawdown_threshold)

    # ---------------------------------------------------------------- reward

    def _calculate_reward(self, prev_value: float, current_value: float, action: int) -> float:
        """
        Reward for the step just taken, per ``config['reward']['mode']``.

        * ``log_return`` - baseline: scaled log return of the portfolio.
        * ``differential_sharpe`` - Moody & Saffell (1998): an online estimate of
          how this bar's return moved the Sharpe ratio, so risk taken to earn a
          gain is priced in rather than ignored.
        * ``cost_aware`` - Zhang, Zohren & Roberts (2020): position-scaled return
          minus an explicit turnover charge, giving the agent a gradient against
          churn instead of only a consequence.
        """
        if prev_value <= 0 or current_value <= 0:
            return -1.0

        mode = self.reward_fn['mode']
        clip = self.reward_fn['clip']

        if mode == 'differential_sharpe':
            period_return = (current_value - prev_value) / prev_value
            reward = self.reward_fn['dsr'].update(period_return)

        elif mode == 'cost_aware':
            price = self._close[self.current_step]
            equity = self._portfolio_value(price)
            exposure = (self.btc_held * price / equity) if equity > 0 else 0.0
            asset_return = self._returns[self.current_step]
            if not np.isfinite(asset_return):
                asset_return = 0.0
            reward = self.reward_fn['cost_aware'].update(exposure, float(asset_return))

        else:
            reward = self.reward_fn['log_return'].update(prev_value, current_value)

        return float(np.clip(reward, -clip, clip))

    # ---------------------------------------------------------- observation

    def _get_observation(self) -> np.ndarray:
        market = self._features[self.current_step]
        price = self._close[self.current_step]
        equity = self._portfolio_value(price)

        portfolio = build_portfolio_features(
            balance=self.balance,
            position_value=self.btc_held * price,
            equity=equity,
            entry_price=self.entry_price,
            price=price,
            peak_equity=self.peak_value,
        )

        return np.concatenate([market, portfolio]).astype(np.float32)

    # ------------------------------------------------------------- reporting

    def _portfolio_value(self, price: Optional[float] = None) -> float:
        if price is None:
            price = self._close[self.current_step]
        return self.balance + (self.btc_held * price)

    def _info(self) -> Dict[str, Any]:
        price = self._close[self.current_step]
        return {
            'timestamp': self.data.index[self.current_step],
            'portfolio_value': self._portfolio_value(price),
            'balance': self.balance,
            'btc_held': self.btc_held,
            'total_trades': self.total_trades,
            'win_rate': self.winning_trades / self.total_trades if self.total_trades else 0.0,
            'current_price': price,
            'halted': self.halted,
            'maker_fills': self.maker_fills,
            'missed_orders': self.missed_orders,
            'financing_paid': self.financing_paid,
        }

    def apply_blackouts(self, windows) -> int:
        """
        Suppress new positions inside scheduled event windows.

        Returns the number of bars suppressed, so a run can report how much of
        the sample it stood aside for.
        """
        from live.calendar import blackout_mask

        if not windows:
            return 0
        self._blackout = blackout_mask(self.data.index, windows).to_numpy(dtype=bool)
        return int(self._blackout.sum())

    def equity_curve(self) -> pd.DataFrame:
        return pd.DataFrame(
            {'portfolio_value': self.portfolio_values},
            index=pd.DatetimeIndex(self.timestamps, name='timestamp'),
        )

    def get_performance_metrics(self) -> Dict[str, float]:
        """Delegate to utils.metrics so every report uses one implementation."""
        from utils.metrics import calculate_comprehensive_metrics
        from utils.data_utils import infer_periods_per_year

        return calculate_comprehensive_metrics(
            self.portfolio_values,
            trades=self.round_trips,
            periods_per_year=infer_periods_per_year(self.data.index),
        )
