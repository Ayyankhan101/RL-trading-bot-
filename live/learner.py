"""
Online learning with a promotion gate.

Two learners run side by side:

* ``ShadowLearner`` takes a gradient step on **every** closed bar, against a
  shadow copy of the live model. This is the continuous half - at 15m it updates
  96 times a day instead of once a week.
* ``OnlineLearner`` periodically trains a fresh challenger from scratch on a
  recent window. This is the slower half, and catches drift a shadow that has
  been incrementally nudged for weeks may have baked in.

Neither ever writes to the trading model directly. Both must pass the same
gate.

"Self-improving" is only meaningful if the bot can tell improvement from noise.
Continuously fine-tuning on the newest bars and always keeping the result is how
a live model quietly degrades: each update is fit to a short, noisy window, and
nothing ever checks whether it helped.

So this module runs champion/challenger instead:

1. Every ``retrain_every`` new bars, copy the live model - the *champion* - into
   a *challenger*.
2. Fine-tune the challenger on a recent training window that **excludes** the
   most recent bars.
3. Score both models greedily on those excluded bars, which neither has trained
   on.
4. Promote the challenger only if it beats the champion by a margin on that
   held-out slice. Otherwise discard it and keep the champion.

Every decision, promoted or not, is appended to a JSON log so the model's
history is auditable rather than a black box that "got better".
"""

import copy
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from agents.torch_dqn import DQNAgent
from environment.trading_env import BitcoinTradingEnv
from utils.data_utils import infer_periods_per_year
from utils.metrics import calculate_comprehensive_metrics


def evaluate_agent(agent: DQNAgent, data: pd.DataFrame,
                   config: Dict[str, Any]) -> Dict[str, Any]:
    """Score an agent greedily over a slice of bars."""
    env = BitcoinTradingEnv(data, config=config)
    state, _ = env.reset()

    while True:
        action = agent.act(state, training=False)
        state, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    return calculate_comprehensive_metrics(
        env.portfolio_values,
        trades=env.round_trips,
        periods_per_year=infer_periods_per_year(data.index),
    )


class OnlineLearner:
    """Periodic fine-tuning, gated on held-out performance."""

    def __init__(self, config: Dict[str, Any], state_dir: str):
        online = config.get('online_learning', {})

        self.enabled = bool(online.get('enabled', True))
        self.retrain_every = int(online.get('retrain_every_bars', 42))
        self.train_window = int(online.get('train_window_bars', 1500))
        self.eval_window = int(online.get('eval_window_bars', 300))
        self.episodes = int(online.get('episodes', 15))
        self.min_improvement = float(online.get('min_improvement', 0.05))
        self.metric = str(online.get('metric', 'sharpe_ratio'))

        self.config = config
        self.state_dir = state_dir
        self.log_path = os.path.join(state_dir, 'promotions.json')
        self.bars_since_retrain = 0

    # ---------------------------------------------------------------- logging

    def _log(self, record: Dict[str, Any]) -> None:
        os.makedirs(self.state_dir, exist_ok=True)
        history = self.history()
        history.append(record)
        with open(self.log_path, 'w') as file:
            json.dump(history, file, indent=2, default=str)

    def history(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path) as file:
            return json.load(file)

    # --------------------------------------------------------------- training

    def note_bar(self) -> None:
        self.bars_since_retrain += 1

    def due(self) -> bool:
        return self.enabled and self.bars_since_retrain >= self.retrain_every

    def _train_config(self) -> Dict[str, Any]:
        config = copy.deepcopy(self.config)
        config['training']['episodes'] = self.episodes
        return config

    def maybe_retrain(self, champion: DQNAgent, data: pd.DataFrame,
                      verbose: bool = True) -> Optional[Dict[str, Any]]:
        """
        Run one champion/challenger cycle if enough new bars have arrived.

        Returns the decision record, or None if it was not due or there was too
        little data. When the challenger wins, ``champion`` is updated in place
        so the caller keeps trading with one object.
        """
        if not self.due():
            return None

        needed = self.train_window + self.eval_window
        if len(data) < needed:
            if verbose:
                print(f"[learn] {len(data)} bars available, need {needed}; skipping")
            self.bars_since_retrain = 0
            return None

        from train_agent import train

        window = data.iloc[-needed:]
        train_slice = window.iloc[:self.train_window]
        # Held out from fine-tuning, and the only bars the gate scores on.
        eval_slice = window.iloc[self.train_window:]

        if verbose:
            print(f"[learn] fine-tuning on {train_slice.index[0]} -> {train_slice.index[-1]}, "
                  f"gate on {eval_slice.index[0]} -> {eval_slice.index[-1]}")

        challenger, _ = train(train_slice, self._train_config(), verbose=False)

        champion_metrics = evaluate_agent(champion, eval_slice, self.config)
        challenger_metrics = evaluate_agent(challenger, eval_slice, self.config)

        champion_score = float(champion_metrics.get(self.metric, 0.0) or 0.0)
        challenger_score = float(challenger_metrics.get(self.metric, 0.0) or 0.0)
        promoted = challenger_score > champion_score + self.min_improvement

        record = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'source': 'retrain',
            'last_bar': str(data.index[-1]),
            'metric': self.metric,
            'champion_score': champion_score,
            'challenger_score': challenger_score,
            'min_improvement': self.min_improvement,
            'promoted': promoted,
            'train_window': [str(train_slice.index[0]), str(train_slice.index[-1])],
            'eval_window': [str(eval_slice.index[0]), str(eval_slice.index[-1])],
            'champion_return': champion_metrics.get('total_return'),
            'challenger_return': challenger_metrics.get('total_return'),
        }

        if promoted:
            champion.q_network.load_state_dict(challenger.q_network.state_dict())
            champion.target_network.load_state_dict(challenger.target_network.state_dict())
            champion.epsilon = challenger.epsilon
            if verbose:
                print(f"[learn] PROMOTED challenger "
                      f"({self.metric} {challenger_score:.3f} > {champion_score:.3f})")
        elif verbose:
            print(f"[learn] kept champion "
                  f"({self.metric} {champion_score:.3f} >= {challenger_score:.3f})")

        self._log(record)
        self.bars_since_retrain = 0
        return record


class ShadowLearner:
    """
    Per-bar online learning behind the same promotion gate.

    Every closed bar feeds a transition to a **shadow** copy of the live model
    and takes a gradient step on it. The live trading model is untouched until
    the shadow wins a scored comparison on bars it has not been updated on.

    This is what makes "constantly learning" safe rather than merely fast. An
    ungated per-bar update is fit to a single noisy observation; applied
    directly to the trader, thousands of them in sequence walk the policy
    somewhere nobody chose. Here they accumulate in a candidate, and the
    candidate has to prove itself once a day.
    """

    def __init__(self, config: Dict[str, Any], state_dir: str):
        online = config.get('online_learning', {})

        self.enabled = bool(online.get('per_bar_updates', True))
        self.gate_every = int(online.get('gate_every_bars', 96))
        self.eval_window = int(online.get('eval_window_bars', 1000))
        self.min_improvement = float(online.get('min_improvement', 0.05))
        self.metric = str(online.get('metric', 'sharpe_ratio'))
        self.warmup_bars = int(online.get('shadow_warmup_bars', 500))

        self.config = config
        self.state_dir = state_dir
        self.log_path = os.path.join(state_dir, 'promotions.json')

        self.shadow: Optional[DQNAgent] = None
        self.bars_seen = 0
        self.updates = 0
        self.bars_since_gate = 0

    # ----------------------------------------------------------------- shadow

    def attach(self, champion: DQNAgent) -> None:
        """Fork a shadow from the current live model."""
        self.shadow = copy.deepcopy(champion)
        # The shadow explores nothing; it learns from the trades the live model
        # actually took, so its epsilon is irrelevant and set to the floor.
        self.shadow.epsilon = self.shadow.epsilon_min

    def observe(self, state, action, reward, next_state, done) -> Optional[float]:
        """Record one bar's transition and take a gradient step."""
        if not self.enabled or self.shadow is None:
            return None

        self.shadow.remember(state, action, reward, next_state, done)
        self.bars_seen += 1
        self.bars_since_gate += 1

        loss = self.shadow.replay()
        if loss is not None:
            self.updates += 1
        return loss

    def due(self) -> bool:
        return (self.enabled and self.shadow is not None
                and self.bars_since_gate >= self.gate_every
                and self.bars_seen >= self.warmup_bars)

    def maybe_promote(self, champion: DQNAgent, data: pd.DataFrame,
                      verbose: bool = True) -> Optional[Dict[str, Any]]:
        """
        Score the shadow against the live model and promote it only if better.

        The scoring window is the most recent ``eval_window`` bars. The shadow
        has seen those bars as transitions, so this is not a clean hold-out in
        the way the from-scratch challenger's window is - it is a check that the
        accumulated updates did not make the policy worse on recent conditions.
        The stricter test is ``OnlineLearner``; this one is the fast guard.
        """
        if not self.due():
            return None

        self.bars_since_gate = 0

        if len(data) < self.eval_window + 10:
            return None

        window = data.iloc[-self.eval_window:]
        champion_metrics = evaluate_agent(champion, window, self.config)
        shadow_metrics = evaluate_agent(self.shadow, window, self.config)

        champion_score = float(champion_metrics.get(self.metric, 0.0) or 0.0)
        shadow_score = float(shadow_metrics.get(self.metric, 0.0) or 0.0)
        promoted = shadow_score > champion_score + self.min_improvement

        record = {
            'timestamp': datetime.now().isoformat(timespec='seconds'),
            'source': 'shadow',
            'last_bar': str(data.index[-1]),
            'metric': self.metric,
            'champion_score': champion_score,
            'challenger_score': shadow_score,
            'min_improvement': self.min_improvement,
            'promoted': promoted,
            'bars_observed': self.bars_seen,
            'gradient_updates': self.updates,
            'eval_window': [str(window.index[0]), str(window.index[-1])],
            'champion_return': champion_metrics.get('total_return'),
            'challenger_return': shadow_metrics.get('total_return'),
        }

        if promoted:
            champion.q_network.load_state_dict(self.shadow.q_network.state_dict())
            champion.target_network.load_state_dict(self.shadow.target_network.state_dict())
            if verbose:
                print(f"[shadow] PROMOTED after {self.updates} updates "
                      f"({self.metric} {shadow_score:.3f} > {champion_score:.3f})")
        else:
            # Re-fork from the champion so the next cycle starts from what is
            # actually trading, instead of compounding a rejected direction.
            self.attach(champion)
            if verbose:
                print(f"[shadow] rejected ({self.metric} {shadow_score:.3f} "
                      f"<= {champion_score:.3f}); re-forked from champion")

        self._log(record)
        return record

    def _log(self, record: Dict[str, Any]) -> None:
        os.makedirs(self.state_dir, exist_ok=True)
        history = self.history()
        history.append(record)
        with open(self.log_path, 'w') as file:
            json.dump(history, file, indent=2, default=str)

    def history(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path) as file:
            return json.load(file)

    def stats(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'bars_observed': self.bars_seen,
            'gradient_updates': self.updates,
            'bars_until_gate': max(0, self.gate_every - self.bars_since_gate),
            'gate_every_bars': self.gate_every,
        }
