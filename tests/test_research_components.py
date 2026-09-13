"""
Reward functions, position sizing, n-step returns and the Deflated Sharpe Ratio.

Each of these implements a published formula; the tests check the formula, not
just that the code runs.
"""

import copy

import numpy as np
import pandas as pd
import pytest

from agents.rewards import CostAwareReturn, DifferentialSharpe, LogReturnReward, build_reward
from environment.sizing import (
    PositionSizer,
    ewma_volatility,
    fractional_kelly,
    volatility_scale,
)
from utils.metrics import deflated_sharpe_ratio, expected_max_sharpe


# ------------------------------------------------- differential Sharpe ratio

def test_differential_sharpe_matches_the_published_recursion():
    """
    Moody & Saffell (1998):
        D_t = (B_{t-1} dA - 0.5 A_{t-1} dB) / (B_{t-1} - A_{t-1}^2)^{3/2}
    """
    eta = 0.1
    dsr = DifferentialSharpe(eta=eta, warmup=0)

    a = b = 0.0
    for r in (0.01, -0.005, 0.02, 0.001):
        delta_a, delta_b = r - a, r * r - b
        variance = b - a * a
        expected = 0.0 if variance <= 1e-12 else \
            (b * delta_a - 0.5 * a * delta_b) / variance ** 1.5

        assert dsr.update(r) == pytest.approx(expected)

        a += eta * delta_a
        b += eta * delta_b


def test_differential_sharpe_prefers_steady_gains_over_volatile_ones():
    """
    The point of the reward: two series with the same mean must not score the
    same. The log-return reward paid identically, which is why the agent had no
    reason to avoid risk.
    """
    rng = np.random.default_rng(0)
    steady = np.full(400, 0.001)
    volatile = rng.normal(0.001, 0.02, 400)

    steady_score = np.mean([DifferentialSharpe(eta=0.01, warmup=50).update(r)
                            for r in steady][50:])

    dsr = DifferentialSharpe(eta=0.01, warmup=50)
    volatile_score = np.mean([dsr.update(r) for r in volatile][50:])

    assert steady_score > volatile_score


def test_differential_sharpe_is_silent_during_warmup():
    dsr = DifferentialSharpe(eta=0.1, warmup=20)
    assert all(dsr.update(0.01) == 0.0 for _ in range(20))


def test_log_return_reward_is_additive_and_clipped():
    reward = LogReturnReward(scale=100.0, clip=10.0)
    total = reward.update(100, 110) + reward.update(110, 121)

    assert total == pytest.approx(np.log(1.21) * 100)
    assert reward.update(100, 1_000_000) == 10.0


# ----------------------------------------------------- cost-aware reward

def test_cost_aware_charges_only_position_changes():
    """Zhang, Zohren & Roberts (2020) eq. 5: the charge is on turnover."""
    reward = CostAwareReturn(cost_bps=10.0, scale=1.0)

    reward.update(1.0, 0.0)                       # open: pays 10bps on |1 - 0|
    held = reward.update(1.0, 0.01)               # hold: no turnover charge

    assert held == pytest.approx(0.01)


def test_cost_aware_penalises_churn():
    flipping = CostAwareReturn(cost_bps=10.0, scale=1.0)
    holding = CostAwareReturn(cost_bps=10.0, scale=1.0)

    flip_total = sum(flipping.update(float(i % 2), 0.0) for i in range(20))
    hold_total = sum(holding.update(1.0, 0.0) for _ in range(20))

    assert flip_total < hold_total


def test_build_reward_rejects_unknown_modes():
    with pytest.raises(ValueError):
        build_reward({'reward': {'mode': 'wishful_thinking'}})


# ------------------------------------------------------- volatility sizing

def test_volatility_scale_is_target_over_realized():
    assert volatility_scale(0.60, 0.30, max_leverage=10) == pytest.approx(0.5)
    assert volatility_scale(0.15, 0.30, max_leverage=10) == pytest.approx(2.0)


def test_volatility_scale_refuses_an_unusable_estimate():
    """
    A near-zero or missing vol estimate would demand infinite size - and that is
    exactly the regime that precedes a volatility spike. Size to nothing instead.
    """
    assert volatility_scale(1e-12, 0.30, max_leverage=1.0) == 0.0
    assert volatility_scale(float('nan'), 0.30) == 0.0


def test_sizing_shrinks_exposure_when_volatility_rises(config):
    sizer = PositionSizer(config)
    calm = sizer.target_exposure_for_level(1.0, 0.10)
    wild = sizer.target_exposure_for_level(1.0, 1.00)

    assert calm > wild


def test_ewma_volatility_is_lagged():
    """
    Sizing bar t with bar t's own return is lookahead. The estimate must be
    shifted.
    """
    returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300))
    vol = ewma_volatility(returns, span=30, periods_per_year=35040)

    unshifted = returns.ewm(span=30, min_periods=15).std(bias=False) * np.sqrt(35040)
    assert vol.iloc[100] == pytest.approx(unshifted.iloc[99])


def test_fractional_kelly_formula():
    """Kelly (1956): f* = mu / sigma^2, scaled by the chosen fraction."""
    assert fractional_kelly(0.001, 0.0004, fraction=0.5, cap=10) == pytest.approx(1.25)


def test_kelly_refuses_a_negative_edge():
    assert fractional_kelly(-0.001, 0.0004, fraction=0.5) == 0.0


def test_sizing_disabled_ignores_volatility(config):
    cfg = copy.deepcopy(config)
    cfg['sizing']['volatility_target'] = False
    sizer = PositionSizer(cfg)

    assert sizer.target_exposure_for_level(1.0, 0.10) == \
        sizer.target_exposure_for_level(1.0, 2.00)


# ------------------------------------------------------------ deflated Sharpe

def test_deflated_sharpe_rejects_a_lucky_winner():
    """
    Bailey & Lopez de Prado (2014). Pure noise searched over 50 configurations
    must not read as skill.
    """
    noise = np.random.default_rng(0).normal(0, 0.01, 5000)
    result = deflated_sharpe_ratio(noise, n_trials=50, periods_per_year=35040)

    assert result['deflated_sharpe'] < 0.5


def test_deflated_sharpe_ranks_a_real_edge_above_noise():
    rng = np.random.default_rng(0)
    edge = rng.normal(0.0004, 0.01, 5000)
    noise = rng.normal(0.0, 0.01, 5000)

    edge_result = deflated_sharpe_ratio(edge, n_trials=50, periods_per_year=35040)
    noise_result = deflated_sharpe_ratio(noise, n_trials=50, periods_per_year=35040)

    assert edge_result['deflated_sharpe'] > noise_result['deflated_sharpe']
    assert edge_result['deflated_sharpe'] > 0.5
    assert noise_result['deflated_sharpe'] < 0.5


def test_more_trials_raise_the_bar():
    returns = np.random.default_rng(1).normal(0.0002, 0.01, 4000)

    few = deflated_sharpe_ratio(returns, n_trials=2, periods_per_year=35040)
    many = deflated_sharpe_ratio(returns, n_trials=500, periods_per_year=35040)

    assert many['deflated_sharpe'] < few['deflated_sharpe']
    assert many['threshold_sharpe'] > few['threshold_sharpe']


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(100, 0.001) > expected_max_sharpe(10, 0.001)
    assert expected_max_sharpe(1) == 0.0


# ------------------------------------------------------------ n-step returns

def test_n_step_return_is_the_discounted_sum(config):
    """
    Sutton & Barto ch. 7: G_t = sum_k gamma^k r_{t+k} + gamma^n max_a Q(s_{t+n},a).
    With a constant reward of 1 the stored return is the geometric series.
    """
    from agents.torch_dqn import DQNAgent

    cfg = copy.deepcopy(config)
    cfg['model'].update({'n_step': 5, 'memory_size': 100, 'prioritized': False})
    agent = DQNAgent(8, 3, cfg, seed=0)

    state = np.zeros(8, dtype=np.float32)
    for _ in range(10):
        agent.remember(state, 1, 1.0, state, False)

    gamma = agent.gamma
    expected = sum(gamma ** k for k in range(5))
    stored = agent.memory.buffer[0]

    assert len(agent.memory) == 6           # 10 pushes, first 4 still buffering
    assert stored[2] == pytest.approx(expected)


def test_n_step_flushes_the_tail_on_episode_end(config):
    from agents.torch_dqn import DQNAgent

    cfg = copy.deepcopy(config)
    cfg['model'].update({'n_step': 4, 'memory_size': 100, 'prioritized': False})
    agent = DQNAgent(8, 3, cfg, seed=0)

    state = np.zeros(8, dtype=np.float32)
    for _ in range(2):
        agent.remember(state, 0, 1.0, state, False)
    agent.remember(state, 0, 1.0, state, True)      # episode ends early

    # Nothing may be stranded in the buffer when the episode ends.
    assert len(agent.memory) == 3
    assert len(agent._n_step_buffer) == 0


def test_n_step_one_is_the_plain_transition(config):
    from agents.torch_dqn import DQNAgent

    cfg = copy.deepcopy(config)
    cfg['model'].update({'n_step': 1, 'memory_size': 100, 'prioritized': False})
    agent = DQNAgent(8, 3, cfg, seed=0)

    state = np.zeros(8, dtype=np.float32)
    agent.remember(state, 2, 0.5, state, False)

    assert len(agent.memory) == 1
    assert agent.memory.buffer[0][2] == pytest.approx(0.5)
