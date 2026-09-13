"""Agent behaviour: variants, PER, epsilon schedule, checkpoint round-trip."""

import copy

import numpy as np
import pytest
import torch

from agents.torch_dqn import DQNAgent, PrioritizedReplayBuffer, QNetwork, SumTree

STATE_SIZE, ACTION_SIZE = 12, 3


@pytest.fixture
def fast_config(config):
    cfg = copy.deepcopy(config)
    cfg['model'].update({
        'memory_size': 500,
        'batch_size': 16,
        'learning_starts': 32,
        'train_frequency': 1,
        'target_update_frequency': 5,
    })
    cfg['network']['hidden_layers'] = [16, 16]
    return cfg


def make_agent(cfg, variant='double'):
    cfg = copy.deepcopy(cfg)
    cfg['model']['variant'] = variant
    return DQNAgent(STATE_SIZE, ACTION_SIZE, cfg, seed=0)


def fill(agent, n=100, rng=None):
    rng = rng or np.random.default_rng(0)
    for _ in range(n):
        state = rng.normal(size=STATE_SIZE).astype(np.float32)
        next_state = rng.normal(size=STATE_SIZE).astype(np.float32)
        agent.remember(state, int(rng.integers(0, ACTION_SIZE)),
                       float(rng.normal()), next_state, False)


def test_sum_tree_total_and_sampling():
    tree = SumTree(4)
    for i, priority in enumerate([1.0, 2.0, 3.0, 4.0]):
        tree.add(priority, f'item{i}')

    assert tree.total() == pytest.approx(10.0)
    _, _, data = tree.get(9.5)
    assert data == 'item3'          # highest-priority leaf


def test_per_weights_are_normalized():
    buffer = PrioritizedReplayBuffer(capacity=64)
    for i in range(64):
        buffer.push((np.zeros(3, dtype=np.float32), 0, float(i), np.zeros(3, dtype=np.float32), False))

    _, indices, weights = buffer.sample(16)
    assert weights.max() == pytest.approx(1.0)
    assert (weights > 0).all()
    assert len(indices) == 16


def test_per_priorities_update():
    buffer = PrioritizedReplayBuffer(capacity=16)
    for _ in range(16):
        buffer.push((np.zeros(2, dtype=np.float32), 0, 0.0, np.zeros(2, dtype=np.float32), False))

    before = buffer.tree.total()
    _, indices, _ = buffer.sample(4)
    buffer.update_priorities(indices, np.full(len(indices), 5.0))
    assert buffer.tree.total() != pytest.approx(before)


@pytest.mark.parametrize('variant', ['dqn', 'double', 'dueling'])
def test_each_variant_trains(fast_config, variant):
    agent = make_agent(fast_config, variant)
    fill(agent, 200)

    losses = [agent.replay() for _ in range(20)]
    assert any(loss is not None for loss in losses)
    assert all(np.isfinite(loss) for loss in losses if loss is not None)


def test_invalid_variant_rejected(fast_config):
    with pytest.raises(ValueError):
        make_agent(fast_config, 'triple')


def test_dueling_head_is_used(fast_config):
    agent = make_agent(fast_config, 'dueling')
    assert agent.q_network.dueling
    assert hasattr(agent.q_network, 'value_head')


def test_dueling_advantage_is_mean_centred():
    net = QNetwork(STATE_SIZE, ACTION_SIZE, [8], dueling=True)
    with torch.no_grad():
        q = net(torch.zeros(1, STATE_SIZE))
        features = net.body(torch.zeros(1, STATE_SIZE))
        value = net.value_head(features)
    assert q.mean(dim=1).item() == pytest.approx(value.item(), abs=1e-6)


def test_epsilon_decays_per_episode_not_per_step(fast_config):
    """
    The TensorFlow agent decayed epsilon on every replay call, driving it to the
    floor inside the first episode. Decay must be an explicit per-episode call.
    """
    agent = make_agent(fast_config)
    fill(agent, 200)

    start = agent.epsilon
    for _ in range(50):
        agent.replay()
    assert agent.epsilon == start

    agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(start * agent.epsilon_decay)


def test_epsilon_never_falls_below_floor(fast_config):
    agent = make_agent(fast_config)
    for _ in range(500):
        agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(agent.epsilon_min)


def test_greedy_action_is_deterministic(fast_config):
    agent = make_agent(fast_config)
    state = np.ones(STATE_SIZE, dtype=np.float32)
    actions = {agent.act(state, training=False) for _ in range(10)}
    assert len(actions) == 1


def test_target_network_updates_on_schedule(fast_config):
    agent = make_agent(fast_config)
    fill(agent, 200)

    # Perturb the online net so the two differ before the scheduled sync.
    with torch.no_grad():
        for param in agent.q_network.parameters():
            param.add_(1.0)

    stale = [param.detach().clone() for param in agent.target_network.parameters()]

    for _ in range(fast_config['model']['target_update_frequency'] + 1):
        agent.replay()

    # The online net keeps training after the sync, so the two are close but not
    # identical; what matters is that the target picked up the online weights.
    for before, online, target in zip(stale, agent.q_network.parameters(),
                                      agent.target_network.parameters()):
        assert not torch.allclose(before, target)
        assert torch.allclose(online, target, atol=1e-2)


def test_checkpoint_round_trip_preserves_policy(fast_config, tmp_path):
    agent = make_agent(fast_config)
    fill(agent, 200)
    for _ in range(20):
        agent.replay()
    agent.decay_epsilon()

    path = tmp_path / 'agent.pt'
    agent.save_model(str(path))
    restored = DQNAgent.from_checkpoint(str(path))

    assert restored.epsilon == pytest.approx(agent.epsilon)
    assert restored.train_steps == agent.train_steps
    assert restored.variant == agent.variant

    rng = np.random.default_rng(1)
    for _ in range(20):
        state = rng.normal(size=STATE_SIZE).astype(np.float32)
        assert restored.act(state, training=False) == agent.act(state, training=False)


def test_checkpoint_rejects_mismatched_state_size(fast_config, tmp_path):
    """A changed feature set must fail loudly, not silently mis-index the input."""
    agent = make_agent(fast_config)
    path = tmp_path / 'agent.pt'
    agent.save_model(str(path))

    other = DQNAgent(STATE_SIZE + 3, ACTION_SIZE, copy.deepcopy(fast_config), seed=0)
    with pytest.raises(ValueError):
        other.load_model(str(path))


def test_uniform_buffer_path(fast_config):
    cfg = copy.deepcopy(fast_config)
    cfg['model']['prioritized'] = False
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE, cfg, seed=0)
    fill(agent, 200)

    assert not agent.prioritized
    assert agent.replay() is not None
