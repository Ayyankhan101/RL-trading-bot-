"""
PyTorch DQN agent for Bitcoin trading.

Supports **n-step returns** (Sutton & Barto, *Reinforcement Learning: An
Introduction*, 2nd ed., ch. 7; used in Rainbow - Hessel et al., "Rainbow:
Combining Improvements in Deep Reinforcement Learning", AAAI 2018). With a
one-step target, a decision whose payoff arrives 24 bars later has to propagate
backwards through 24 bootstrapped updates before it influences the policy at
all. The n-step target

    G_t = sum_{k=0}^{n-1} gamma^k r_{t+k} + gamma^n max_a Q(s_{t+n}, a)

carries the reward back directly, which matters here because the minimum holding
period guarantees the consequence of an entry is always several bars away.

One agent class covers three target rules, selected by ``model.variant`` in
config.yaml:

* ``dqn``     - vanilla: r + gamma * max_a Q_target(s', a)
* ``double``  - action chosen by the online net, valued by the target net
* ``dueling`` - dueling network head, Double-DQN target rule

Replay is prioritized (proportional, sum-tree) with importance-sampling
correction and beta annealing; set ``model.prioritized: false`` for uniform
sampling.
"""

import random
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VARIANTS = ('dqn', 'double', 'dueling')


class SumTree:
    """Fixed-capacity binary tree over priorities, for O(log n) sampling."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: List[Any] = [None] * capacity
        self.write = 0
        self.size = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data: Any) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        if idx != 0:
            self._propagate(idx, change)

    def get(self, value: float) -> Tuple[int, float, Any]:
        idx = 0
        while True:
            left = 2 * idx + 1
            if left >= len(self.tree):
                break
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]


class PrioritizedReplayBuffer:
    """Proportional prioritized experience replay."""

    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4,
                 beta_frames: int = 100_000, epsilon: float = 1e-5):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = max(1, beta_frames)
        self.epsilon = epsilon
        self.max_priority = 1.0
        self.frame = 0

    def __len__(self) -> int:
        return self.tree.size

    def push(self, transition: Tuple) -> None:
        # New transitions enter at the highest seen priority so they are sampled
        # at least once before their true TD error is known.
        self.tree.add(self.max_priority ** self.alpha, transition)

    def beta(self) -> float:
        progress = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + progress * (1.0 - self.beta_start)

    def sample(self, batch_size: int) -> Tuple[List[Tuple], np.ndarray, np.ndarray]:
        self.frame += 1
        batch, indices, priorities = [], [], []
        segment = self.tree.total() / batch_size

        for i in range(batch_size):
            value = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(value)
            if data is None:            # tree not yet full at this leaf
                idx, priority, data = self.tree.get(random.uniform(0, self.tree.total()))
            batch.append(data)
            indices.append(idx)
            priorities.append(priority)

        probs = np.array(priorities, dtype=np.float64) / max(self.tree.total(), 1e-12)
        weights = (len(self) * np.maximum(probs, 1e-12)) ** (-self.beta())
        weights /= max(weights.max(), 1e-12)

        return batch, np.array(indices), weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, error in zip(indices, td_errors):
            priority = float(abs(error) + self.epsilon)
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(int(idx), priority ** self.alpha)


class UniformReplayBuffer:
    """Plain uniform replay, exposing the same interface as the prioritized one."""

    def __init__(self, capacity: int):
        self.buffer: List[Tuple] = []
        self.capacity = capacity
        self.position = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Tuple) -> None:
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Tuple], np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        return batch, np.zeros(batch_size, dtype=np.int64), np.ones(batch_size, dtype=np.float32)

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        return


class QNetwork(nn.Module):
    """MLP Q-network, with an optional dueling head."""

    def __init__(self, state_size: int, action_size: int, hidden_layers: List[int],
                 dropout: float = 0.0, dueling: bool = False):
        super().__init__()
        self.dueling = dueling

        layers: List[nn.Module] = []
        in_dim = state_size
        for units in hidden_layers:
            layers.append(nn.Linear(in_dim, units))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = units
        self.body = nn.Sequential(*layers)

        if dueling:
            self.value_head = nn.Linear(in_dim, 1)
            self.advantage_head = nn.Linear(in_dim, action_size)
        else:
            self.head = nn.Linear(in_dim, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.body(state)
        if not self.dueling:
            return self.head(features)

        value = self.value_head(features)
        advantage = self.advantage_head(features)
        # Mean-centred advantage keeps V and A identifiable.
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class DQNAgent:
    """Value-based agent over a discrete action space."""

    def __init__(self, state_size: int, action_size: int, config: Dict[str, Any],
                 device: Optional[str] = None, seed: Optional[int] = None,
                 feature_names: Optional[List[str]] = None):
        self.config = config
        model_cfg = config['model']
        network_cfg = config['network']

        self.state_size = state_size
        self.action_size = action_size
        self.feature_names = feature_names or []

        self.variant = str(model_cfg.get('variant', 'double')).lower()
        if self.variant not in VARIANTS:
            raise ValueError(f"model.variant must be one of {VARIANTS}, got {self.variant!r}")

        self.gamma = float(model_cfg['gamma'])
        self.batch_size = int(model_cfg['batch_size'])
        self.learning_rate = float(model_cfg['learning_rate'])
        self.epsilon = float(model_cfg['epsilon_start'])
        self.epsilon_min = float(model_cfg['epsilon_end'])
        self.epsilon_decay = float(model_cfg['epsilon_decay'])
        self.target_update_frequency = int(model_cfg.get('target_update_frequency', 500))
        self.train_frequency = int(model_cfg.get('train_frequency', 4))
        self.learning_starts = int(model_cfg.get('learning_starts', 1000))
        self.grad_clip = float(model_cfg.get('grad_clip', 10.0))

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))

        hidden = list(network_cfg['hidden_layers'])
        dropout = float(network_cfg.get('dropout_rate', 0.0))
        dueling = self.variant == 'dueling'

        self.q_network = QNetwork(state_size, action_size, hidden, dropout, dueling).to(self.device)
        self.target_network = QNetwork(state_size, action_size, hidden, dropout, dueling).to(self.device)
        self.update_target_network()
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

        capacity = int(model_cfg['memory_size'])
        if bool(model_cfg.get('prioritized', True)):
            self.memory: Any = PrioritizedReplayBuffer(
                capacity,
                alpha=float(model_cfg.get('per_alpha', 0.6)),
                beta_start=float(model_cfg.get('per_beta_start', 0.4)),
                beta_frames=int(model_cfg.get('per_beta_frames', 100_000)),
            )
            self.prioritized = True
        else:
            self.memory = UniformReplayBuffer(capacity)
            self.prioritized = False

        # n-step returns: transitions are buffered until n rewards have accrued.
        self.n_step = max(1, int(model_cfg.get('n_step', 1)))
        self._n_step_buffer: deque = deque(maxlen=self.n_step)

        self.train_steps = 0
        self.env_steps = 0
        self.loss_history: List[float] = []

    # ----------------------------------------------------------------- acting

    def act(self, state: np.ndarray, training: bool = True) -> int:
        """Epsilon-greedy over Q(s, .)."""
        self.env_steps += 1

        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        self.q_network.eval()
        with torch.no_grad():
            tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_network(tensor)
        if training:
            self.q_network.train()
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action, reward, next_state, done) -> None:
        """
        Store a transition, accumulating n-step returns when configured.

        With ``n_step > 1`` a transition is only pushed once ``n`` rewards have
        been observed, and it carries the discounted sum plus the state ``n``
        bars ahead. On episode end the buffer is flushed so the tail is not lost.
        """
        transition = (np.asarray(state, dtype=np.float32), int(action), float(reward),
                      np.asarray(next_state, dtype=np.float32), bool(done))

        if self.n_step == 1:
            self.memory.push(transition)
            return

        self._n_step_buffer.append(transition)

        pushed_full = len(self._n_step_buffer) == self.n_step
        if pushed_full:
            self.memory.push(self._build_n_step())

        if done:
            # Flush every remaining suffix so the tail of an episode is not
            # stranded in the buffer: shorter horizons, same bootstrap target.
            if pushed_full:
                self._n_step_buffer.popleft()
            while self._n_step_buffer:
                self.memory.push(self._build_n_step())
                self._n_step_buffer.popleft()

    def _build_n_step(self) -> Tuple:
        """Collapse the buffer into one transition with a discounted n-step return."""
        state, action = self._n_step_buffer[0][0], self._n_step_buffer[0][1]

        cumulative = 0.0
        next_state, done = self._n_step_buffer[-1][3], self._n_step_buffer[-1][4]
        for index, (_, _, reward, step_next, step_done) in enumerate(self._n_step_buffer):
            cumulative += (self.gamma ** index) * reward
            if step_done:
                next_state, done = step_next, True
                break

        return (state, action, cumulative, next_state, done)

    # --------------------------------------------------------------- learning

    def replay(self) -> Optional[float]:
        """One gradient step. Returns the batch loss, or None if it was skipped."""
        if len(self.memory) < max(self.batch_size, self.learning_starts):
            return None
        if self.env_steps % self.train_frequency != 0:
            return None

        batch, indices, weights = self.memory.sample(self.batch_size)

        states = torch.as_tensor(np.stack([b[0] for b in batch]), device=self.device)
        actions = torch.as_tensor([b[1] for b in batch], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([b[3] for b in batch]), device=self.device)
        dones = torch.as_tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)
        is_weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.variant == 'dqn':
                next_q = self.target_network(next_states).max(dim=1).values
            else:
                # Double-DQN decoupling, shared by the 'double' and 'dueling' variants.
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)

            # gamma^n, because each stored reward is already an n-step return.
            target_q = rewards + (self.gamma ** self.n_step) * next_q * (1.0 - dones)

        td_errors = target_q - current_q
        loss = (is_weights * F.smooth_l1_loss(current_q, target_q, reduction='none')).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.prioritized:
            self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        self.train_steps += 1
        if self.train_steps % self.target_update_frequency == 0:
            self.update_target_network()

        loss_value = float(loss.item())
        self.loss_history.append(loss_value)
        return loss_value

    def decay_epsilon(self) -> float:
        """
        Decay exploration **once per episode**.

        The TensorFlow version decayed on every replay call, i.e. every few env
        steps, which drove epsilon to its floor inside the first episode and
        meant the agent never explored.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def update_target_network(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())

    # ------------------------------------------------------------ persistence

    def save_model(self, filepath: str) -> None:
        """Checkpoint weights *and* the state needed to resume or replay a run."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'variant': self.variant,
            'n_step': self.n_step,
            'feature_names': self.feature_names,
            'config': self.config,
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        if checkpoint['state_size'] != self.state_size:
            raise ValueError(
                f"Checkpoint expects state_size={checkpoint['state_size']}, "
                f"environment provides {self.state_size}. The feature set changed."
            )

        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)
        self.train_steps = checkpoint.get('train_steps', 0)
        self.feature_names = checkpoint.get('feature_names', self.feature_names)
        print(f"Model loaded from {filepath} (variant={checkpoint.get('variant')})")

    @classmethod
    def from_checkpoint(cls, filepath: str, device: Optional[str] = None) -> "DQNAgent":
        """Rebuild an agent straight from a checkpoint's own config."""
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
        agent = cls(
            state_size=checkpoint['state_size'],
            action_size=checkpoint['action_size'],
            config=checkpoint['config'],
            device=device,
            feature_names=checkpoint.get('feature_names'),
        )
        agent.load_model(filepath)
        return agent

    def get_training_stats(self) -> Dict[str, float]:
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'train_steps': self.train_steps,
            'avg_loss': float(np.mean(self.loss_history[-100:])) if self.loss_history else 0.0,
        }
