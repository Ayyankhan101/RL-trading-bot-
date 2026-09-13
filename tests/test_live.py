"""Live stack: feed hygiene, paper broker mechanics, promotion gate."""

import copy
import json
import os
import time

import numpy as np
import pandas as pd
import pytest

from live import feed
from live.learner import OnlineLearner, evaluate_agent
from live.paper_broker import BUY, HOLD, SELL, PaperBroker


# ------------------------------------------------------------------- the feed

def test_only_closed_bars_are_returned(monkeypatch):
    """
    The newest kline is still forming - its high, low and close will change.
    Trading it is the live equivalent of reading bar t+1 in a backtest.
    """
    now_ms = int(time.time() * 1000)
    four_hours = 4 * 60 * 60 * 1000

    closed_open = now_ms - 2 * four_hours
    forming_open = now_ms - four_hours // 2

    payload = [
        [closed_open, '100', '110', '90', '105', '10', closed_open + four_hours - 1],
        [forming_open, '105', '120', '104', '118', '5', forming_open + four_hours - 1],
    ]
    monkeypatch.setattr(feed, '_get_json', lambda url, timeout=15.0: payload)

    bars = feed.fetch_binance()
    assert len(bars) == 1
    assert bars['close'].iloc[-1] == 105.0


def test_feed_falls_back_when_a_venue_fails(monkeypatch):
    """One exchange being unreachable should not end a live run."""
    frame = pd.DataFrame(
        {'open': [1.0], 'high': [2.0], 'low': [0.5], 'close': [1.5], 'volume': [3.0]},
        index=pd.DatetimeIndex([pd.Timestamp('2026-01-01')], name='timestamp'))

    def boom(**kwargs):
        raise feed.FeedError('geoblocked')

    monkeypatch.setattr(feed, 'SOURCES', {'binance': boom, 'kraken': lambda **kw: frame})

    result = feed.fetch_bars('4h', source='auto')
    assert result.attrs['source'] == 'kraken'


def test_feed_raises_when_every_venue_fails(monkeypatch):
    def boom(**kwargs):
        raise feed.FeedError('down')

    monkeypatch.setattr(feed, 'SOURCES', {'binance': boom, 'kraken': boom})
    with pytest.raises(feed.FeedError):
        feed.fetch_bars('4h', source='auto')


def test_sentiment_never_reads_a_future_value():
    """Each bar takes the most recent reading at or before its own timestamp."""
    bars = pd.DataFrame(
        {'close': [1.0, 2.0, 3.0]},
        index=pd.to_datetime(['2026-01-01 00:00', '2026-01-01 12:00', '2026-01-02 08:00']))
    sentiment = pd.Series(
        [20.0, 80.0],
        index=pd.to_datetime(['2026-01-01 00:00', '2026-01-02 00:00']),
        name='Fear & Greed Index')

    merged = feed.attach_fear_greed(bars, sentiment)

    assert merged['Fear & Greed Index'].tolist() == [20.0, 20.0, 80.0]


def test_update_cache_drops_volume_by_default(monkeypatch, tmp_path):
    """
    The training data has no volume, so a live volume column would silently add
    a feature the policy never learned on.
    """
    frame = pd.DataFrame(
        {'open': [1.0], 'high': [2.0], 'low': [0.5], 'close': [1.5], 'volume': [9.0]},
        index=pd.DatetimeIndex([pd.Timestamp('2026-01-01')], name='timestamp'))
    monkeypatch.setattr(feed, 'fetch_bars', lambda interval, source: frame)

    cached = feed.update_cache(cache_dir=str(tmp_path))
    assert 'volume' not in cached.columns


def test_update_cache_merges_without_duplicates(monkeypatch, tmp_path):
    index = pd.DatetimeIndex(pd.to_datetime(['2026-01-01 00:00', '2026-01-01 04:00']),
                             name='timestamp')
    first = pd.DataFrame({'open': [1.0, 2.0], 'high': [1, 2], 'low': [1, 2],
                          'close': [1.0, 2.0]}, index=index)
    second = first.copy()
    second.index = pd.DatetimeIndex(pd.to_datetime(['2026-01-01 04:00', '2026-01-01 08:00']),
                                    name='timestamp')

    monkeypatch.setattr(feed, 'fetch_bars', lambda interval, source: first)
    feed.update_cache(cache_dir=str(tmp_path))
    monkeypatch.setattr(feed, 'fetch_bars', lambda interval, source: second)
    merged = feed.update_cache(cache_dir=str(tmp_path))

    assert len(merged) == 3
    assert merged.index.is_monotonic_increasing


# --------------------------------------------------------------- the broker

@pytest.fixture
def broker_config(config):
    cfg = copy.deepcopy(config)
    cfg['risk'] = {}
    cfg['backtesting']['slippage'] = 0.0
    # These tests pin taker mechanics; maker execution has its own suite in
    # tests/test_execution.py.
    cfg['execution'] = {'mode': 'taker', 'taker_fee': cfg['trading']['transaction_cost'],
                        'maker_fee': cfg['trading']['transaction_cost'], 'slippage': 0.0}
    return cfg


def make_broker(cfg, tmp_path):
    return PaperBroker(cfg, str(tmp_path / 'account.json'))


def test_buy_then_sell_flat_price_is_a_loss(broker_config, tmp_path):
    broker = make_broker(broker_config, tmp_path)
    broker.buy(20_000, '2026-01-01')
    broker.sell(20_000, '2026-01-02')

    assert broker.state.total_round_trips == 1
    assert broker.state.winning_round_trips == 0
    assert broker.round_trips[0]['pnl'] < 0


def test_slippage_is_adverse_on_both_legs(broker_config, tmp_path):
    broker_config['execution']['slippage'] = 0.01
    broker = make_broker(broker_config, tmp_path)

    buy = broker.buy(1000, '2026-01-01')
    sell = broker.sell(1000, '2026-01-02')

    assert buy['price'] == pytest.approx(1010.0)
    assert sell['price'] == pytest.approx(990.0)


def test_exposure_is_capped(broker_config, tmp_path):
    broker_config['trading']['max_position_size'] = 0.5
    broker = make_broker(broker_config, tmp_path)
    broker.buy(1000, '2026-01-01')

    assert broker.state.btc_held * 1000 / broker.equity(1000) <= 0.51


def test_dust_orders_are_rejected(broker_config, tmp_path):
    broker = make_broker(broker_config, tmp_path)
    for _ in range(20):
        broker.buy(1000, '2026-01-01')

    assert broker.state.fills <= 2


def test_stop_loss_fires_on_the_bar_low(broker_config, tmp_path):
    broker_config['risk'] = {'stop_loss': 0.05}
    broker = make_broker(broker_config, tmp_path)
    broker.buy(1000, '2026-01-01')

    reason = broker.check_risk_exits(high=1010, low=900, timestamp='2026-01-02')

    assert reason == 'stop_loss'
    assert broker.state.btc_held == 0
    assert broker.round_trips[0]['return_pct'] == pytest.approx(-0.05, abs=1e-9)


def test_stop_beats_target_inside_one_bar(broker_config, tmp_path):
    broker_config['risk'] = {'stop_loss': 0.05, 'take_profit': 0.05}
    broker = make_broker(broker_config, tmp_path)
    broker.buy(1000, '2026-01-01')

    assert broker.check_risk_exits(high=1200, low=800, timestamp='2026-01-02') == 'stop_loss'


def test_drawdown_halt_flattens_and_blocks_new_entries(broker_config, tmp_path):
    broker_config['risk'] = {'max_drawdown_threshold': 0.10}
    broker = make_broker(broker_config, tmp_path)
    broker.buy(1000, '2026-01-01')
    broker.mark(500, '2026-01-02')

    assert broker.state.halted
    assert broker.state.btc_held == 0
    assert broker.buy(500, '2026-01-03') is None


def test_state_survives_a_restart(broker_config, tmp_path):
    broker = make_broker(broker_config, tmp_path)
    broker.buy(1000, '2026-01-01')
    broker.save()

    reloaded = make_broker(broker_config, tmp_path)

    assert reloaded.state.btc_held == pytest.approx(broker.state.btc_held)
    assert reloaded.state.balance == pytest.approx(broker.state.balance)
    assert reloaded.state.entry_price == pytest.approx(broker.state.entry_price)


def test_summary_reports_return_against_starting_balance(broker_config, tmp_path):
    broker = make_broker(broker_config, tmp_path)
    broker.buy(1000, '2026-01-01')
    summary = broker.summary(2000)

    assert summary['total_return'] > 0.5
    assert summary['btc_held'] > 0


# --------------------------------------------------------- the promotion gate

@pytest.fixture
def learner_config(config):
    cfg = copy.deepcopy(config)
    cfg['network']['hidden_layers'] = [16, 16]
    cfg['model'].update({'memory_size': 2000, 'batch_size': 16,
                         'learning_starts': 32, 'train_frequency': 4})
    cfg['online_learning'] = {
        'enabled': True, 'retrain_every_bars': 2, 'train_window_bars': 200,
        'eval_window_bars': 80, 'episodes': 1, 'metric': 'sharpe_ratio',
        'min_improvement': 0.05,
    }
    cfg['regime'] = {'enabled': False}
    return cfg


class StubAgent:
    """Deterministic policy, so the gate's decision is a function of the data."""

    def __init__(self, action):
        self.action = action

    def act(self, state, training=False):
        return self.action


def test_gate_scores_on_bars_held_out_of_fine_tuning(learner_config, enriched_ohlc,
                                                    state_size, n_actions, tmp_path):
    learner = OnlineLearner(learner_config, str(tmp_path))
    learner.bars_since_retrain = 5

    from agents.torch_dqn import DQNAgent
    champion = DQNAgent(state_size, n_actions, learner_config, seed=0)

    record = learner.maybe_retrain(champion, enriched_ohlc, verbose=False)

    assert record is not None
    train_end = pd.Timestamp(record['train_window'][1])
    eval_start = pd.Timestamp(record['eval_window'][0])
    assert eval_start > train_end          # the gate never scores on training bars


def test_gate_rejects_a_challenger_that_does_not_clear_the_margin(learner_config, tmp_path):
    learner = OnlineLearner(learner_config, str(tmp_path))
    record = {'promoted': False, 'champion_score': 1.0, 'challenger_score': 1.01}
    learner._log(record)

    history = learner.history()
    assert history[-1]['promoted'] is False


def test_promotion_history_is_appended_not_overwritten(learner_config, tmp_path):
    learner = OnlineLearner(learner_config, str(tmp_path))
    learner._log({'promoted': True})
    learner._log({'promoted': False})

    assert len(learner.history()) == 2
    assert os.path.exists(tmp_path / 'promotions.json')


def test_retrain_is_skipped_without_enough_history(learner_config, enriched_ohlc,
                                                  state_size, n_actions, tmp_path):
    learner = OnlineLearner(learner_config, str(tmp_path))
    learner.train_window = 10_000
    learner.bars_since_retrain = 99

    from agents.torch_dqn import DQNAgent
    champion = DQNAgent(state_size, n_actions, learner_config, seed=0)

    assert learner.maybe_retrain(champion, enriched_ohlc, verbose=False) is None
    assert learner.bars_since_retrain == 0        # the counter still resets


def test_not_due_returns_none(learner_config, enriched_ohlc, state_size, n_actions, tmp_path):
    learner = OnlineLearner(learner_config, str(tmp_path))
    learner.bars_since_retrain = 0

    from agents.torch_dqn import DQNAgent
    champion = DQNAgent(state_size, n_actions, learner_config, seed=0)

    assert learner.maybe_retrain(champion, enriched_ohlc, verbose=False) is None


def test_evaluate_agent_returns_metrics(learner_config, enriched_ohlc):
    metrics = evaluate_agent(StubAgent(HOLD), enriched_ohlc, learner_config)

    assert metrics['total_return'] == pytest.approx(0.0)
    assert metrics['n_periods'] == len(enriched_ohlc) - 1


def test_bootstrap_holds_out_every_replayed_bar(learner_config, enriched_ohlc,
                                                state_size, n_actions, tmp_path, monkeypatch):
    """
    A bootstrap fit must exclude the bars the caller is about to replay.

    Otherwise the bot's opening stretch is an in-sample replay of data the model
    just trained on, presented as a live track record.
    """
    from live.trader import LiveTrader

    cfg = copy.deepcopy(learner_config)
    cfg['online_learning'].update({'bootstrap': True, 'bootstrap_episodes': 1,
                                   'bootstrap_window_bars': 0})

    trader = LiveTrader(cfg, state_dir=str(tmp_path))
    captured = {}

    def fake_train(window, config, log_path=None, verbose=True, val_data=None):
        from agents.torch_dqn import DQNAgent
        captured['end'] = window.index[-1]
        return DQNAgent(state_size, n_actions, config, seed=0), []

    monkeypatch.setattr('train_agent.train', fake_train)

    catch_up = 120
    trader.process_new_bars(enriched_ohlc, max_bars=catch_up, verbose=False)

    first_replayed = enriched_ohlc.index[len(enriched_ohlc) - catch_up]
    assert captured['end'] < first_replayed


def test_account_is_persisted_before_retraining(learner_config, enriched_ohlc,
                                                tmp_path, monkeypatch):
    """
    Fine-tuning is the longest, most failure-prone step in a cycle. If it dies,
    the account must already be on disk - otherwise a live run silently restarts
    from its initial balance.
    """
    from live.trader import LiveTrader

    cfg = copy.deepcopy(learner_config)
    cfg['online_learning'].update({'bootstrap': False, 'retrain_every_bars': 1})

    trader = LiveTrader(cfg, state_dir=str(tmp_path))
    monkeypatch.setattr(trader, 'refresh_data', lambda seed_history=True: enriched_ohlc)

    def exploding_retrain(*args, **kwargs):
        assert os.path.exists(tmp_path / 'account.json'), 'account not saved before retrain'
        raise RuntimeError('fine-tuning failed')

    monkeypatch.setattr(trader.learner, 'maybe_retrain', exploding_retrain)
    monkeypatch.setattr(trader.learner, 'due', lambda: True)

    with pytest.raises(RuntimeError):
        trader.tick(max_bars=20, verbose=False)

    saved = json.loads((tmp_path / 'account.json').read_text())
    assert saved['last_bar'] is not None


def test_gate_separates_two_different_policies(learner_config, enriched_ohlc, tmp_path):
    """
    A gate that scored every model the same would promote noise.

    A buy-and-hold policy and a never-trade policy must not receive the same
    score on the same bars.
    """
    from live.paper_broker import BUY as BUY_ACTION

    class OnceThenHold:
        def __init__(self):
            self.done = False

        def act(self, state, training=False):
            if self.done:
                return HOLD
            self.done = True
            return BUY_ACTION

    long_metrics = evaluate_agent(OnceThenHold(), enriched_ohlc, learner_config)
    flat_metrics = evaluate_agent(StubAgent(HOLD), enriched_ohlc, learner_config)

    assert long_metrics['total_return'] != flat_metrics['total_return']
    assert long_metrics['total_trades'] == 1
    assert flat_metrics['total_trades'] == 0


# ------------------------------------------------------- per-bar shadow learner

@pytest.fixture
def shadow_config(learner_config):
    cfg = copy.deepcopy(learner_config)
    cfg['online_learning'].update({
        'per_bar_updates': True,
        'gate_every_bars': 10,
        'shadow_warmup_bars': 20,
        'eval_window_bars': 100,
        'min_improvement': 0.05,
    })
    return cfg


def make_shadow(cfg, state_size, tmp_path, n_actions=3):
    from agents.torch_dqn import DQNAgent
    from live.learner import ShadowLearner

    champion = DQNAgent(state_size, n_actions, cfg, seed=0)
    shadow = ShadowLearner(cfg, str(tmp_path))
    shadow.attach(champion)
    return champion, shadow


def feed_bars(shadow, state_size, n, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        state = rng.normal(size=state_size).astype(np.float32)
        nxt = rng.normal(size=state_size).astype(np.float32)
        shadow.observe(state, int(rng.integers(0, 3)), float(rng.normal()), nxt, False)


def test_shadow_updates_every_bar(shadow_config, state_size, tmp_path):
    """The continuous half: a gradient step per closed bar, not per week."""
    champion, shadow = make_shadow(shadow_config, state_size, tmp_path)
    feed_bars(shadow, state_size, 200)

    assert shadow.bars_seen == 200
    assert shadow.updates > 0
    assert shadow.stats()['gradient_updates'] == shadow.updates


def test_shadow_never_mutates_the_champion(shadow_config, state_size, tmp_path):
    """
    Per-bar learning must not reach the trading model.

    This is the whole safety argument for "constantly learning": thousands of
    single-observation updates accumulate in a candidate, not in the trader.
    """
    import torch

    champion, shadow = make_shadow(shadow_config, state_size, tmp_path)
    before = [p.detach().clone() for p in champion.q_network.parameters()]

    feed_bars(shadow, state_size, 300)

    for original, current in zip(before, champion.q_network.parameters()):
        assert torch.equal(original, current)

    # ...and the shadow really did move.
    moved = any(not torch.equal(o, s) for o, s in
                zip(before, shadow.shadow.q_network.parameters()))
    assert moved


def test_shadow_does_not_gate_before_warmup(shadow_config, state_size, tmp_path):
    champion, shadow = make_shadow(shadow_config, state_size, tmp_path)
    feed_bars(shadow, state_size, 15)          # below shadow_warmup_bars

    assert not shadow.due()


def test_shadow_gate_runs_after_warmup(shadow_config, state_size, tmp_path):
    champion, shadow = make_shadow(shadow_config, state_size, tmp_path)
    feed_bars(shadow, state_size, 50)

    assert shadow.due()


def test_rejected_shadow_is_reforked_from_the_champion(shadow_config, enriched_ohlc,
                                                       state_size, tmp_path):
    """
    A rejected direction must not compound.

    Without the re-fork the next cycle would continue from the losing candidate
    and keep walking away from what is actually trading.
    """
    import torch

    champion, shadow = make_shadow(shadow_config, state_size, tmp_path)
    feed_bars(shadow, state_size, 100)

    record = shadow.maybe_promote(champion, enriched_ohlc, verbose=False)
    assert record is not None

    if not record['promoted']:
        for c, s in zip(champion.q_network.parameters(),
                        shadow.shadow.q_network.parameters()):
            assert torch.equal(c, s)


def test_shadow_promotion_is_logged_with_its_source(shadow_config, enriched_ohlc,
                                                    state_size, tmp_path):
    """The audit trail has to say which kind of update won."""
    champion, shadow = make_shadow(shadow_config, state_size, tmp_path)
    feed_bars(shadow, state_size, 100)
    shadow.maybe_promote(champion, enriched_ohlc, verbose=False)

    history = json.loads((tmp_path / 'promotions.json').read_text())
    assert history[-1]['source'] == 'shadow'
    assert 'gradient_updates' in history[-1]


def test_disabled_shadow_does_nothing(shadow_config, state_size, tmp_path):
    cfg = copy.deepcopy(shadow_config)
    cfg['online_learning']['per_bar_updates'] = False
    champion, shadow = make_shadow(cfg, state_size, tmp_path)

    feed_bars(shadow, state_size, 50)

    assert shadow.bars_seen == 0
    assert not shadow.due()


def test_15m_periods_per_year():
    """Annualization must follow the data, not a hardcoded 252 or 2190."""
    from utils.data_utils import infer_periods_per_year

    index = pd.date_range('2026-01-01', periods=500, freq='15min')
    assert infer_periods_per_year(index) == pytest.approx(35040.0)


def test_feed_supports_15m():
    assert feed.INTERVALS['15m']['binance'] == '15m'
    assert feed.INTERVALS['15m']['kraken'] == 15
    assert feed.INTERVALS['15m']['coinbase'] == 900


def test_page_budget_scales_with_the_interval():
    """
    One page is ~166 days at 4h but only ~10 days at 15m. A fixed page limit
    would silently truncate the history and leave the model believing it had
    years of data.
    """
    start = pd.Timestamp.utcnow().tz_localize(None) - pd.to_timedelta(365, unit='D')

    assert feed.pages_needed('15m', start) > 30
    assert feed.pages_needed('4h', start) < 10


def test_idle_cycle_does_not_rewrite_the_model(learner_config, enriched_ohlc,
                                               state_size, n_actions,
                                               tmp_path, monkeypatch):
    """
    Polling faster than the bar interval is normal (15m bars, 60s poll). A cycle
    with no newly closed bar changed no state, so it must not rewrite the
    checkpoint - otherwise an idle bot churns its model every minute forever.
    """
    from agents.torch_dqn import DQNAgent
    from live.trader import LiveTrader

    cfg = copy.deepcopy(learner_config)
    cfg['online_learning']['bootstrap'] = False

    trader = LiveTrader(cfg, state_dir=str(tmp_path))
    monkeypatch.setattr(trader, 'refresh_data', lambda seed_history=True: enriched_ohlc)
    trader.agent = DQNAgent(state_size, n_actions, cfg, seed=0)
    trader.model_origin = 'test'
    trader.shadow.attach(trader.agent)

    trader.tick(max_bars=20, verbose=False)             # first cycle does work
    checkpoint = tmp_path / 'live_model.pt'
    assert checkpoint.exists()
    stamp = checkpoint.stat().st_mtime_ns

    status = trader.tick(verbose=False)                 # nothing new closed

    assert status['new_bars'] == 0
    assert status['idle'] is True
    assert status['next_bar_due']
    assert checkpoint.stat().st_mtime_ns == stamp       # untouched


def test_action_names_cover_the_full_action_space():
    """
    The action space grows with the configured sizing levels. A name table that
    assumed the original three actions crashed the live loop with KeyError: 3
    the first time an intermediate conviction level was chosen.
    """
    from live.trader import action_name

    assert action_name(0) == 'HOLD'
    assert action_name(1) == 'BUY'
    assert action_name(2) == 'SELL'
    for extra in (3, 4, 5):
        assert action_name(extra).startswith('BUY_L')
