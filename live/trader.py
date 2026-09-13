"""
Live paper-trading loop.

Each cycle:

1. Refresh the bar cache from a public exchange (closed bars only).
2. Attach the real Fear & Greed Index, compute causal indicators, build the
   feature matrix with the exact code the agent trained against.
3. For every bar not yet processed, in chronological order:
   a. apply stop-loss / take-profit against that bar's high/low,
   b. mark the account to that bar's close,
   c. build the observation and let the agent choose,
   d. fill the order at that bar's close, with slippage and fee.
4. Hand the transition to the online learner, which periodically runs a
   champion/challenger cycle (see live/learner.py).
5. Persist account state, equity curve, trades and a status snapshot.

Bar ordering matters: the agent only ever sees bars that have closed, and a
position opened on bar *t* can only be stopped out on bar *t+1* or later. This
is the same ordering the backtest uses, which is what makes the two comparable.
"""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from agents.torch_dqn import DQNAgent
from features.observation import build_market_features, build_portfolio_features
from features.technical_indicators import add_technical_indicators
from live.feed import INTERVALS, attach_fear_greed, build_cache, update_cache
from live.learner import OnlineLearner, ShadowLearner
from live.paper_broker import BUY, HOLD, SELL, PaperBroker
from utils.data_utils import ensure_fear_greed

ACTION_NAMES = {HOLD: 'HOLD', BUY: 'BUY', SELL: 'SELL'}


def action_name(action: int) -> str:
    """
    Human name for an action.

    The action space grows with the configured sizing levels - actions 3 and
    above are intermediate conviction levels - so this must not assume the
    original three.
    """
    return ACTION_NAMES.get(action, f'BUY_L{action - 1}')


class LiveTrader:
    """Runs an agent against live bars in a paper account."""

    def __init__(self, config: Dict[str, Any], model_path: Optional[str] = None,
                 state_dir: str = 'results/live', symbol: str = 'BTCUSD',
                 interval: str = '4h', source: str = 'auto'):
        self.config = config
        self.state_dir = state_dir
        self.symbol = symbol
        self.interval = interval
        self.source = source

        os.makedirs(state_dir, exist_ok=True)

        self.broker = PaperBroker(config, os.path.join(state_dir, 'account.json'))
        if self.broker.state.started_at is None:
            self.broker.state.started_at = datetime.now().isoformat(timespec='seconds')

        self.learner = OnlineLearner(config, state_dir)
        self.shadow = ShadowLearner(config, state_dir)
        self.model_path = model_path
        self.agent: Optional[DQNAgent] = None
        self.checkpoint_path = os.path.join(state_dir, 'live_model.pt')

        self.equity_path = os.path.join(state_dir, 'equity.csv')
        self.trades_path = os.path.join(state_dir, 'trades.csv')
        self.status_path = os.path.join(state_dir, 'status.json')
        self.decisions_path = os.path.join(state_dir, 'decisions.csv')

    # ------------------------------------------------------------------- data

    def refresh_data(self, seed_history: bool = True) -> pd.DataFrame:
        """Update the cache and return an indicator-enriched frame."""
        if seed_history:
            # One source, no gaps. Splicing the bundled CSV onto recent bars
            # would leave months missing in the middle, and every rolling
            # indicator spanning that hole would be wrong.
            build_cache(symbol=self.symbol, interval=self.interval)

        bars = update_cache(symbol=self.symbol, interval=self.interval, source=self.source)
        bars = attach_fear_greed(bars)
        bars = ensure_fear_greed(bars)

        enriched = add_technical_indicators(bars)
        enriched = enriched.dropna(
            subset=[c for c in enriched.columns if c != 'Fear & Greed Classification'])

        if enriched.empty:
            raise RuntimeError('No usable bars after indicator warm-up')
        return enriched

    def entry_blocked(self, moment: "pd.Timestamp") -> Optional[Dict[str, Any]]:
        """
        Whether a new position may be opened at this moment.

        Overridden by instrument-specific traders; the crypto path has no
        scheduled events to avoid, so nothing is ever blocked here.
        """
        return None

    # ------------------------------------------------------------------ agent

    def load_agent(self, state_size: int, data: Optional[pd.DataFrame] = None) -> DQNAgent:
        """
        Load the live model.

        Preference order: the checkpoint this bot has been improving, then the
        model passed on the command line, then a bootstrap trained on the cached
        history. The status file records which case applied, so live results are
        never mistaken for a trained model's when they came from an untrained one.
        """
        for path, origin in ((self.checkpoint_path, 'live'), (self.model_path, 'seed')):
            if path and os.path.exists(path):
                agent = DQNAgent.from_checkpoint(path)
                if agent.state_size != state_size:
                    print(f"[live] checkpoint {path} expects {agent.state_size} features, "
                          f"data provides {state_size}; ignoring it")
                    continue
                self.model_origin = origin
                return agent

        bootstrap = self.config.get('online_learning', {}).get('bootstrap', True)
        if bootstrap and data is not None and len(data) > self.learner.eval_window * 2:
            return self._bootstrap_agent(data)

        print('[live] no usable checkpoint - starting from an untrained agent')
        self.model_origin = 'untrained'
        return DQNAgent(state_size, 3, self.config,
                        seed=int(self.config['training'].get('seed', 42)))

    def _bootstrap_agent(self, data: pd.DataFrame) -> DQNAgent:
        """
        Train an initial model on cached history before trading anything.

        The most recent bars are excluded so the first thing the bot trades is
        data the model has not seen - otherwise its opening stretch would be an
        in-sample replay dressed up as live performance.
        """
        import copy as _copy

        from train_agent import train

        # Hold out at least the gate window, and at least whatever the caller is
        # about to replay - otherwise the bot's opening stretch would be an
        # in-sample replay presented as live performance.
        holdout = max(self.learner.eval_window, getattr(self, '_catch_up', 0) or 0)
        window = data.iloc[:-holdout] if len(data) > holdout else data

        max_bars = int(self.config.get('online_learning', {}).get('bootstrap_window_bars', 0))
        if max_bars and len(window) > max_bars:
            window = window.iloc[-max_bars:]

        config = _copy.deepcopy(self.config)
        config['training']['episodes'] = int(
            self.config.get('online_learning', {}).get('bootstrap_episodes', 25))

        print(f"[live] bootstrapping a model on {len(window)} cached bars "
              f"({window.index[0]} -> {window.index[-1]}), "
              f"holding out the last {holdout} bars")

        agent, _ = train(window, config, log_path=os.path.join(self.state_dir,
                                                              'bootstrap_log.csv'),
                         verbose=True)
        # Checkpoint immediately: a crash later in the cycle should not throw
        # away the whole bootstrap fit and force it to be repeated.
        agent.save_model(self.checkpoint_path)
        self.model_origin = 'bootstrap'
        return agent

    # -------------------------------------------------------------- one cycle

    def process_new_bars(self, data: pd.DataFrame, max_bars: Optional[int] = None,
                         verbose: bool = True) -> List[Dict[str, Any]]:
        """Walk every bar that has closed since the last run."""
        names, market = build_market_features(data)
        state_size = len(names) + 5

        # Set before loading, because a bootstrap fit must exclude every bar the
        # caller is about to replay.
        self._catch_up = max_bars

        if self.agent is None:
            self.agent = self.load_agent(state_size, data=data)
            self.shadow.attach(self.agent)

        last_bar = self.broker.state.last_bar
        start = 0
        if last_bar:
            processed = data.index <= pd.Timestamp(last_bar)
            start = int(processed.sum())

        if max_bars is not None:
            start = max(start, len(data) - max_bars)

        decisions: List[Dict[str, Any]] = []
        equity_rows: List[Dict[str, Any]] = []
        previous_observation: Optional[np.ndarray] = None
        previous_action: Optional[int] = None
        previous_equity: Optional[float] = None

        for i in range(start, len(data)):
            timestamp = str(data.index[i])
            row = data.iloc[i]
            price = float(row['close'])

            # (a) count the bar against the minimum hold, then stops: a
            # position opened earlier can exit on this bar.
            self.broker.advance_bar()
            exit_reason = self.broker.check_risk_exits(
                float(row['high']), float(row['low']), timestamp)

            # (b) mark to market, which also trips the drawdown kill switch.
            equity = self.broker.mark(price, timestamp)

            # (c) observe.
            portfolio = build_portfolio_features(
                balance=self.broker.state.balance,
                position_value=self.broker.state.btc_held * price,
                equity=equity,
                entry_price=self.broker.state.entry_price,
                price=price,
                peak_equity=self.broker.state.peak_equity,
            )
            observation = np.concatenate([market[i], portfolio]).astype(np.float32)

            # Reward for the *previous* decision, known only now that this bar
            # has closed. Same log-return reward the agent was trained on.
            if previous_observation is not None and previous_equity and equity > 0:
                reward = float(np.clip(np.log(equity / previous_equity) * 100.0, -10.0, 10.0))
                self.agent.remember(previous_observation, previous_action,
                                    reward, observation, False)
                # The continuous half: one gradient step per closed bar, applied
                # to a shadow model rather than the one currently trading.
                self.shadow.observe(previous_observation, previous_action,
                                    reward, observation, False)

            # (d) decide and fill at this bar's close.
            action = self.agent.act(observation, training=False)

            # Instruments with scheduled events (gold) suppress new entries in
            # the window around them, where the spread is unreliable. Exits are
            # always allowed: being unable to leave is worse than not entering.
            blocked = self.entry_blocked(pd.Timestamp(timestamp))
            if blocked and action == BUY:
                action = HOLD

            atr = float(row['ATR']) if 'ATR' in row and np.isfinite(row['ATR']) else None
            fill = self.broker.act(action, price, timestamp, atr=atr)

            previous_observation = observation
            previous_action = action
            previous_equity = self.broker.equity(price)

            self.learner.note_bar()

            decisions.append({
                'timestamp': timestamp,
                'price': price,
                'action': action_name(action),
                'filled': bool(fill),
                'exit_reason': exit_reason or '',
                'equity': previous_equity,
                'position_btc': self.broker.state.btc_held,
                'halted': self.broker.state.halted,
            })
            equity_rows.append({'timestamp': timestamp, 'equity': previous_equity,
                                'price': price})

            if verbose and (fill or exit_reason):
                label = exit_reason or action_name(action)
                print(f"[live] {timestamp}  {label:<13} @ ${price:,.2f}  "
                      f"equity ${previous_equity:,.2f}")

        if equity_rows:
            self._append_csv(self.equity_path, equity_rows)
        if decisions:
            self._append_csv(self.decisions_path, decisions)
        if self.broker.round_trips:
            self._append_csv(self.trades_path, self.broker.round_trips)
            self.broker.round_trips = []

        return decisions

    def tick(self, max_bars: Optional[int] = None, verbose: bool = True) -> Dict[str, Any]:
        """One full cycle: refresh, trade any new bars, maybe retrain, persist."""
        data = self.refresh_data()
        decisions = self.process_new_bars(data, max_bars=max_bars, verbose=verbose)

        price = float(data['close'].iloc[-1])

        # Nothing closed since the last cycle: no state changed, so skip the
        # checkpoint write and the retrain check. Polling faster than the bar
        # interval is normal (15m bars, 60s poll), and without this guard an
        # idle bot rewrites its model every minute for nothing.
        if not decisions:
            return self._write_status(data, price, 0, None, idle=True)

        # Persist the account *before* retraining. Fine-tuning is the longest
        # and most failure-prone step in the cycle, and losing a live account's
        # state to it would mean silently restarting from the initial balance.
        self.broker.save()
        if self.agent is not None:
            self.agent.save_model(self.checkpoint_path)
        self._write_status(data, price, len(decisions), None)

        promotion = None
        if self.agent is not None and self.shadow.due():
            promotion = self.shadow.maybe_promote(self.agent, data, verbose=verbose)
            self.agent.save_model(self.checkpoint_path)

        if self.agent is not None and self.learner.due():
            retrained = self.learner.maybe_retrain(self.agent, data, verbose=verbose)
            promotion = retrained or promotion
            self.agent.save_model(self.checkpoint_path)
            # A from-scratch promotion replaces what the shadow was built on.
            if retrained and retrained.get('promoted'):
                self.shadow.attach(self.agent)

        self.broker.save()
        return self._write_status(data, price, len(decisions), promotion)

    def run(self, poll_seconds: int = 300, max_cycles: Optional[int] = None,
            verbose: bool = True) -> None:
        """Poll forever (or for ``max_cycles``), trading each newly closed bar."""
        cycle = 0
        while max_cycles is None or cycle < max_cycles:
            cycle += 1
            try:
                status = self.tick(verbose=verbose)
                if verbose and status['new_bars']:
                    print(f"[live] cycle {cycle}: {status['new_bars']} new bar(s), "
                          f"equity ${status['equity']:,.2f} "
                          f"({status['total_return'] * 100:+.2f}%), "
                          f"position {status['btc_held']:.6f} BTC")
                elif verbose and cycle % 10 == 1:
                    # One line per ten idle polls, and it says what it is waiting
                    # for - a wall of "0 new bars" reads like a hung process.
                    print(f"[live] idle (cycle {cycle}); last bar "
                          f"{status['last_bar']}, next due ~{status['next_bar_due']}")
            except Exception as exc:                      # noqa: BLE001
                # A transient network failure must pause the bot, not kill a run
                # that may have been going for days.
                print(f"[live] cycle {cycle} failed: {exc}")

            if max_cycles is not None and cycle >= max_cycles:
                break
            time.sleep(poll_seconds)

    # ------------------------------------------------------------- artifacts

    @staticmethod
    def _append_csv(path: str, rows: List[Dict[str, Any]]) -> None:
        frame = pd.DataFrame(rows)
        header = not os.path.exists(path)
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        frame.to_csv(path, mode='a', header=header, index=False)

    def _next_bar_due(self, data: pd.DataFrame) -> str:
        """When the next bar closes, so an idle log line is legible."""
        step = pd.Timedelta(INTERVALS[self.interval]['pandas'])
        return str(data.index[-1] + 2 * step)

    def _write_status(self, data: pd.DataFrame, price: float, new_bars: int,
                      promotion: Optional[Dict[str, Any]],
                      idle: bool = False) -> Dict[str, Any]:
        summary = self.broker.summary(price)

        # Buy & hold over the same live window, as the reference the paper
        # account has to beat to be worth running at all.
        benchmark = None
        if os.path.exists(self.equity_path):
            curve = pd.read_csv(self.equity_path)
            if len(curve) > 1:
                benchmark = float(curve['price'].iloc[-1] / curve['price'].iloc[0] - 1.0)

        status = {
            'updated_at': datetime.now().isoformat(timespec='seconds'),
            'started_at': self.broker.state.started_at,
            'symbol': self.symbol,
            'interval': self.interval,
            'source': data.attrs.get('source', self.source),
            'mode': 'paper',
            'model_origin': getattr(self, 'model_origin', 'unknown'),
            'last_bar': str(data.index[-1]),
            'last_price': price,
            'new_bars': new_bars,
            'idle': idle,
            'next_bar_due': self._next_bar_due(data),
            'buy_hold_return_since_start': benchmark,
            'online_learning': {
                'enabled': self.learner.enabled,
                'bars_since_retrain': self.learner.bars_since_retrain,
                'retrain_every_bars': self.learner.retrain_every,
                'promotions': sum(1 for r in self.learner.history() if r.get('promoted')),
                'cycles': len(self.learner.history()),
                'last_decision': promotion,
                'per_bar': self.shadow.stats(),
            },
            **summary,
        }

        with open(self.status_path, 'w') as file:
            json.dump(status, file, indent=2, default=str)
        return status
