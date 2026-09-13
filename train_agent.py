"""
Training script for the Bitcoin RL trading bot.

Trains a DQN (or Double/Dueling variant) on a chronological slice of the data and
evaluates it greedily on a held-out, purged validation slice.

    python train_agent.py --data data/BTC.csv --episodes 60
"""

import argparse
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

from agents.torch_dqn import DQNAgent
from environment.trading_env import BitcoinTradingEnv
from utils.data_utils import infer_periods_per_year, prepare_dataset, split_data
from utils.metrics import print_performance_report


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Read config.yaml and apply CLI overrides **in memory**.

    The previous version wrote overrides back to config.yaml, silently mutating
    a tracked file (and destroying its comments) on every run.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    for section, values in (overrides or {}).items():
        config.setdefault(section, {}).update(values)

    return config


def run_episode(agent: DQNAgent, env: BitcoinTradingEnv, training: bool = True,
                seed: Optional[int] = None) -> Dict[str, Any]:
    """Run one full pass over the environment's data."""
    state, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0

    while True:
        action = agent.act(state, training=training)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        if training:
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

        state = next_state
        total_reward += reward
        steps += 1

        if done:
            break

    if training:
        agent.decay_epsilon()

    return {
        'total_reward': total_reward,
        'steps': steps,
        'portfolio_value': info['portfolio_value'],
        'total_trades': info['total_trades'],
        'win_rate': info['win_rate'],
        'halted': info['halted'],
    }


def train(train_data: pd.DataFrame, config: Dict[str, Any],
          val_data: Optional[pd.DataFrame] = None,
          log_path: Optional[str] = None,
          verbose: bool = True) -> Tuple[DQNAgent, List[Dict[str, Any]]]:
    """Train an agent on ``train_data``; returns the agent and the per-episode log."""
    seed = int(config['training'].get('seed', 42))
    set_seed(seed)

    env = BitcoinTradingEnv(train_data, config=config)
    agent = DQNAgent(
        state_size=env.n_features,
        action_size=env.n_actions,
        config=config,
        seed=seed,
        feature_names=env.feature_names + env._portfolio_features(),
    )

    episodes = int(config['training']['episodes'])
    log_frequency = int(config['training'].get('log_frequency', 5))
    history: List[Dict[str, Any]] = []

    if verbose:
        print(f"\nTraining {agent.variant} agent: {episodes} episodes, "
              f"{len(train_data)} bars, {env.n_features} features")

    for episode in range(1, episodes + 1):
        result = run_episode(agent, env, training=True, seed=seed + episode)
        stats = agent.get_training_stats()
        record = {'episode': episode, **result, **stats}
        history.append(record)

        if verbose and (episode % log_frequency == 0 or episode == 1):
            print(f"Episode {episode:4d} | reward {result['total_reward']:9.2f} "
                  f"| equity ${result['portfolio_value']:10,.2f} "
                  f"| trades {result['total_trades']:4d} "
                  f"| win {result['win_rate']:5.1%} "
                  f"| eps {stats['epsilon']:.3f} "
                  f"| loss {stats['avg_loss']:.4f}")

    if log_path:
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
        pd.DataFrame(history).to_csv(log_path, index=False)
        if verbose:
            print(f"Training log written to {log_path}")

    if val_data is not None and len(val_data) > 1 and verbose:
        val_env = BitcoinTradingEnv(val_data, config=config)
        run_episode(agent, val_env, training=False)
        metrics = val_env.get_performance_metrics()
        print_performance_report(metrics, title="VALIDATION (greedy, out-of-sample)")

    return agent, history


def main() -> None:
    parser = argparse.ArgumentParser(description='Train the Bitcoin RL trading bot')
    parser.add_argument('--data', type=str, default='data/BTC.csv', help='Path to OHLC CSV')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--episodes', type=int, help='Override training.episodes')
    parser.add_argument('--variant', type=str, choices=['dqn', 'double', 'dueling'],
                        help='Override model.variant')
    parser.add_argument('--seed', type=int, help='Override training.seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Checkpoint path (default: models/trained_models/<variant>_<ts>.pt)')
    args = parser.parse_args()

    overrides: Dict[str, Dict[str, Any]] = {}
    if args.episodes:
        overrides.setdefault('training', {})['episodes'] = args.episodes
    if args.seed is not None:
        overrides.setdefault('training', {})['seed'] = args.seed
    if args.variant:
        overrides.setdefault('model', {})['variant'] = args.variant

    config = load_config(args.config, overrides)

    data = prepare_dataset(args.data)
    print(f"Bars: {len(data)} | {data.index[0]} -> {data.index[-1]} "
          f"| {infer_periods_per_year(data.index):.0f} periods/year")

    train_data, val_data = split_data(
        data,
        train_fraction=1 - float(config['training']['validation_split']),
        purge=int(config['training'].get('purge_bars', 0)),
    )
    print(f"Train: {len(train_data)} bars ({train_data.index[0]} -> {train_data.index[-1]})")
    print(f"Validation: {len(val_data)} bars ({val_data.index[0]} -> {val_data.index[-1]})")

    agent, history = train(train_data, config, val_data=val_data,
                           log_path='results/training_log.csv')

    output = args.output or (
        f"models/trained_models/{agent.variant}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    agent.save_model(output)

    os.makedirs('results', exist_ok=True)
    with open('results/training_summary.json', 'w') as file:
        json.dump({
            'checkpoint': output,
            'variant': agent.variant,
            'episodes': len(history),
            'train_range': [str(train_data.index[0]), str(train_data.index[-1])],
            'validation_range': [str(val_data.index[0]), str(val_data.index[-1])],
            'final_epsilon': agent.epsilon,
        }, file, indent=2)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
