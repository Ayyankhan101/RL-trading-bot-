"""
Reward functions for trading agents.

The plain log-return reward has a known failure mode, and this bot exhibited it:
every positive bar is worth the same regardless of the risk taken to get it, so
the policy learns to take profit at the first opportunity and has no incentive to
hold a winner. In the 15m run the median holding period was exactly the minimum
allowed - the agent learned to leave.

Two literature-backed alternatives are implemented:

**Differential Sharpe Ratio** - Moody & Saffell, "Performance Functions and
Reinforcement Learning for Trading Systems and Portfolios" (Journal of
Forecasting, 1998) and "Learning to Trade via Direct Reinforcement" (IEEE
Transactions on Neural Networks, 2001). An online, per-step approximation of the
*derivative* of the Sharpe ratio with respect to the newest return, so the agent
optimizes risk-adjusted performance incrementally rather than raw profit.

Given exponential moving estimates of the first and second moments,

    A_t = A_{t-1} + eta * (R_t - A_{t-1})
    B_t = B_{t-1} + eta * (R_t^2 - B_{t-1})

the differential Sharpe ratio is

    D_t = (B_{t-1} * dA - 0.5 * A_{t-1} * dB) / (B_{t-1} - A_{t-1}^2)^{3/2}

with dA = R_t - A_{t-1} and dB = R_t^2 - B_{t-1}.

**Cost-aware return** - Zhang, Zohren & Roberts, "Deep Reinforcement Learning for
Trading" (Journal of Financial Data Science, 2020), equation 5. The reward is the
position-scaled return minus an explicit charge for *changing* the position:

    R_t = mu * (sigma_tgt / sigma_{t-1}) * A_{t-1} * r_t
          - bp * p_t * | (sigma_tgt/sigma_{t-1}) A_t - (sigma_tgt/sigma_{t-2}) A_{t-1} |

Charging turnover inside the reward, rather than only inside the equity curve,
gives the agent a gradient against churn instead of merely a consequence.
"""

import math
from typing import Any, Dict, Optional


class DifferentialSharpe:
    """
    Online differential Sharpe ratio (Moody & Saffell 1998).

    ``eta`` is the adaptation rate: roughly 1/window, so 1/1000 tracks a
    thousand-bar Sharpe. Returns 0 during warm-up, when the variance estimate is
    not yet meaningful.
    """

    def __init__(self, eta: float = 1e-3, warmup: int = 50, scale: float = 1.0):
        self.eta = float(eta)
        self.warmup = int(warmup)
        self.scale = float(scale)
        self.reset()

    def reset(self) -> None:
        self.a = 0.0          # EMA of returns
        self.b = 0.0          # EMA of squared returns
        self.n = 0

    def update(self, portfolio_return: float) -> float:
        """Feed one period return, receive the reward for that step."""
        r = float(portfolio_return)
        delta_a = r - self.a
        delta_b = r * r - self.b

        variance = self.b - self.a * self.a
        if self.n < self.warmup or variance <= 1e-12:
            reward = 0.0
        else:
            numerator = self.b * delta_a - 0.5 * self.a * delta_b
            reward = numerator / (variance ** 1.5)

        # The moments update *after* the reward, so D_t uses A_{t-1}, B_{t-1}.
        self.a += self.eta * delta_a
        self.b += self.eta * delta_b
        self.n += 1

        return float(reward) * self.scale


class CostAwareReturn:
    """
    Position-scaled return minus a turnover charge (Zhang et al. 2020, eq. 5).

    ``cost_bps`` should be the full one-way cost - fee plus expected slippage -
    because the charge is applied to the size of the position *change*.
    """

    def __init__(self, cost_bps: float = 15.0, scale: float = 100.0,
                 turnover_multiplier: float = 1.0):
        self.cost = float(cost_bps) / 10_000.0
        self.scale = float(scale)
        self.turnover_multiplier = float(turnover_multiplier)
        self.reset()

    def reset(self) -> None:
        self.previous_position = 0.0

    def update(self, position: float, asset_return: float) -> float:
        """
        Args:
            position: exposure held *going into* this bar, in [0, 1].
            asset_return: the bar's return on the underlying.
        """
        pnl = self.previous_position * float(asset_return)
        turnover = abs(float(position) - self.previous_position)
        charge = self.turnover_multiplier * self.cost * turnover

        self.previous_position = float(position)
        return (pnl - charge) * self.scale


class LogReturnReward:
    """Baseline: scaled log return of the portfolio. Kept for comparison."""

    def __init__(self, scale: float = 100.0, clip: float = 10.0):
        self.scale = float(scale)
        self.clip = float(clip)

    def reset(self) -> None:
        return

    def update(self, previous_value: float, current_value: float) -> float:
        if previous_value <= 0 or current_value <= 0:
            return -1.0
        reward = math.log(current_value / previous_value) * self.scale
        return max(-self.clip, min(self.clip, reward))


def build_reward(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct the reward components named by ``config['reward']``.

    Returns a dict with the mode and whichever components that mode needs, so the
    environment can stay agnostic about which paper it is following.
    """
    reward_cfg = config.get('reward', {}) or {}
    mode = str(reward_cfg.get('mode', 'log_return')).lower()

    components: Dict[str, Any] = {'mode': mode}

    if mode == 'differential_sharpe':
        components['dsr'] = DifferentialSharpe(
            eta=float(reward_cfg.get('eta', 1e-3)),
            warmup=int(reward_cfg.get('warmup', 50)),
            scale=float(reward_cfg.get('scale', 1.0)),
        )
    elif mode == 'cost_aware':
        components['cost_aware'] = CostAwareReturn(
            cost_bps=float(reward_cfg.get('cost_bps', 15.0)),
            scale=float(reward_cfg.get('scale', 100.0)),
            turnover_multiplier=float(reward_cfg.get('turnover_multiplier', 1.0)),
        )
    elif mode != 'log_return':
        raise ValueError(f"Unknown reward.mode: {mode!r}")

    components['log_return'] = LogReturnReward(
        scale=float(reward_cfg.get('scale', 100.0)),
        clip=float(reward_cfg.get('clip', 10.0)),
    )
    components['clip'] = float(reward_cfg.get('clip', 10.0))
    return components
