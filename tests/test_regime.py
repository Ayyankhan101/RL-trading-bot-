"""Regime detection, adaptive sizing, and the event blackout."""

import numpy as np
import pandas as pd
import pytest

from live.calendar import blackout_mask, blackout_windows, in_blackout, relevant_events
from strategies.regime import (
    CALM,
    RANGE,
    STRESSED,
    TREND,
    RegimeConfig,
    classify,
    efficiency_ratio,
    position_scale,
    realized_volatility,
    spread_is_acceptable,
    volatility_scale,
)


def series(values, freq='1h'):
    return pd.Series(values, index=pd.date_range('2026-01-01', periods=len(values), freq=freq))


# ------------------------------------------------------------- volatility

def test_volatility_estimate_is_lagged():
    """
    Sizing bar t with bar t's own return is lookahead: the position has to be
    set before the move it is sized for.
    """
    rng = np.random.default_rng(0)
    close = series(100 * np.cumprod(1 + rng.normal(0, 0.01, 300)))

    vol = realized_volatility(close, span=20, periods_per_year=8760)
    unshifted = (close.pct_change().ewm(span=20, min_periods=10).std(bias=False)
                 * np.sqrt(8760))

    assert vol.iloc[100] == pytest.approx(unshifted.iloc[99])


def test_volatility_scale_halves_when_volatility_doubles():
    """The entire adaptation mechanism: risk per trade stays put as scale moves."""
    config = RegimeConfig(target_volatility=0.20, max_scale=10.0)

    assert volatility_scale(0.20, config) == pytest.approx(1.0)
    assert volatility_scale(0.40, config) == pytest.approx(0.5)
    assert volatility_scale(0.10, config) == pytest.approx(2.0)


def test_volatility_scale_refuses_an_unusable_estimate():
    config = RegimeConfig()
    assert volatility_scale(float('nan'), config) == config.min_scale
    assert volatility_scale(0.0, config) == config.min_scale


# --------------------------------------------------------------- structure

def test_efficiency_ratio_separates_trend_from_chop():
    """Kaufman's ratio: net move over path travelled."""
    trending = series(np.arange(100, 200, dtype=float))
    choppy = series(np.tile([100.0, 101.0], 50))

    assert efficiency_ratio(trending, 24).dropna().mean() > 0.9
    assert efficiency_ratio(choppy, 24).dropna().mean() < 0.2


def test_classify_labels_both_dimensions():
    rng = np.random.default_rng(1)
    close = series(100 * np.cumprod(1 + rng.normal(0, 0.005, 1200)))

    regimes = classify(close, RegimeConfig(vol_lookback=200), periods_per_year=8760)

    assert set(regimes['volatility_regime'].dropna().unique()) <= {CALM, 'normal', STRESSED}
    assert set(regimes['structure'].dropna().unique()) <= {TREND, RANGE}


def test_stressed_regime_gets_less_exposure_than_calm():
    """A shock shrinks the position without anyone predicting the shock."""
    config = RegimeConfig(target_volatility=0.15)

    calm = pd.Series({'volatility': 0.10, 'volatility_regime': CALM, 'structure': RANGE})
    stressed = pd.Series({'volatility': 0.60, 'volatility_regime': STRESSED,
                          'structure': RANGE})

    assert position_scale(calm, config) > position_scale(stressed, config) * 4


def test_trending_market_halves_a_mean_reversion_position():
    """A mean-reversion edge inverts in a trend, so it is sized down there."""
    config = RegimeConfig(target_volatility=0.15)
    ranging = pd.Series({'volatility': 0.15, 'volatility_regime': 'normal',
                         'structure': RANGE})
    trending = pd.Series({'volatility': 0.15, 'volatility_regime': 'normal',
                          'structure': TREND})

    assert position_scale(trending, config) == pytest.approx(
        position_scale(ranging, config) * 0.5)


# ------------------------------------------------------------ spread guard

def test_spread_guard_rejects_a_news_blowout():
    """
    Gold spreads widen 10-50x through an FOMC print. The cost is observable
    before the trade, which makes this the cheapest protection available.
    """
    config = RegimeConfig(max_spread_multiple=3.0)

    assert spread_is_acceptable(0.30, 0.15, config)
    assert not spread_is_acceptable(1.50, 0.15, config)


def test_spread_guard_passes_when_normal_is_unknown():
    assert spread_is_acceptable(5.0, 0.0, RegimeConfig())


# --------------------------------------------------------------- blackout

def event_frame():
    return pd.DataFrame({
        'timestamp': pd.to_datetime(['2026-09-16T14:00Z', '2026-09-14T08:30Z'], utc=True),
        'country': ['USD', 'CAD'],
        'impact': ['High', 'Low'],
        'title': ['FOMC Statement', 'CPI m/m'],
    })


def test_only_relevant_high_impact_events_create_blackouts():
    relevant = relevant_events(event_frame(), ('USD', 'EUR'), impacts=('High',))

    assert len(relevant) == 1
    assert relevant.iloc[0]['title'] == 'FOMC Statement'


def test_blackout_window_brackets_the_event():
    windows = blackout_windows(relevant_events(event_frame(), ('USD',)), 30, 30)
    event = pd.Timestamp('2026-09-16T14:00Z')

    assert windows[0]['start'] == event - pd.Timedelta(minutes=30)
    assert windows[0]['end'] == event + pd.Timedelta(minutes=30)


def test_in_blackout_identifies_the_event():
    windows = blackout_windows(relevant_events(event_frame(), ('USD',)), 30, 30)

    assert in_blackout(pd.Timestamp('2026-09-16 14:05'), windows)['title'] == 'FOMC Statement'
    assert in_blackout(pd.Timestamp('2026-09-16 16:00'), windows) is None


def test_blackout_mask_suppresses_only_the_window():
    windows = blackout_windows(relevant_events(event_frame(), ('USD',)), 30, 30)
    index = pd.date_range('2026-09-16 12:00', periods=24, freq='15min')

    mask = blackout_mask(index, windows)

    assert mask.sum() == 5                       # 13:30 through 14:30 inclusive
    assert not mask.iloc[0]


def test_env_suppresses_entries_during_a_blackout(enriched_ohlc, config):
    """The one case where the timing of a shock genuinely is known in advance."""
    import copy

    from environment.trading_env import BitcoinTradingEnv

    cfg = copy.deepcopy(config)
    cfg['regime'] = {'enabled': False}
    cfg['risk'] = {'min_hold_bars': 0}

    env = BitcoinTradingEnv(enriched_ohlc, config=cfg)
    covered = [{'start': enriched_ohlc.index[0].tz_localize('UTC'),
                'end': enriched_ohlc.index[-1].tz_localize('UTC'),
                'title': 'blanket', 'country': 'USD', 'impact': 'High'}]

    assert env.apply_blackouts(covered) == len(enriched_ohlc)

    env.reset(seed=1)
    for _ in range(50):
        env.step(1)                              # every entry attempt suppressed

    assert env.btc_held == 0
    assert env.total_trades == 0


def test_rebalance_deadband_prevents_churn(enriched_ohlc, config):
    """
    Volatility-scaled exposure moves continuously. Without a deadband every
    small drift in the target triggers a trade: measured on hourly gold, trade
    count went 160 -> 624 and profit factor fell from 1.83 to 0.90. The churn
    cost more than the risk reduction bought.
    """
    import copy

    from environment.trading_env import BitcoinTradingEnv

    def trade_count(deadband):
        cfg = copy.deepcopy(config)
        cfg['regime'] = {'enabled': True, 'target_volatility': 0.15}
        cfg['risk'] = {'min_hold_bars': 0}
        cfg['sizing'] = {**cfg.get('sizing', {}), 'rebalance_deadband': deadband}

        env = BitcoinTradingEnv(enriched_ohlc, config=cfg)
        env.reset(seed=1)
        rng = np.random.default_rng(2)
        while True:
            _, _, terminated, truncated, _ = env.step(int(rng.integers(0, env.n_actions)))
            if terminated or truncated:
                break
        return env.open_orders

    assert trade_count(0.25) < trade_count(0.0)


# ---------------------------------------------------------------- gold feed

def test_forming_bar_is_dropped():
    """
    Yahoo stamps a bar with its opening time, so the last row is still open
    until a full interval has elapsed. Trading it is the live equivalent of
    reading bar t+1 in a backtest - and the bot would look brilliant.
    """
    from live.gold_feed import drop_forming_bar

    now = pd.Timestamp.utcnow().tz_localize(None)
    index = pd.DatetimeIndex([now - pd.Timedelta(hours=3),
                              now - pd.Timedelta(hours=2),
                              now - pd.Timedelta(minutes=20)])   # still forming
    frame = pd.DataFrame({'close': [1.0, 2.0, 3.0]}, index=index)

    closed = drop_forming_bar(frame, '1h')

    assert len(closed) == 2
    assert closed.index[-1] == index[1]


def test_a_fully_elapsed_bar_is_kept():
    from live.gold_feed import drop_forming_bar

    now = pd.Timestamp.utcnow().tz_localize(None)
    index = pd.DatetimeIndex([now - pd.Timedelta(hours=3), now - pd.Timedelta(hours=2)])
    frame = pd.DataFrame({'close': [1.0, 2.0]}, index=index)

    assert len(drop_forming_bar(frame, '1h')) == 2


def test_session_gaps_are_counted_not_filled():
    """Gold stops trading at weekends; filling those smears indicators."""
    from live.gold_feed import session_gaps

    index = pd.DatetimeIndex(['2026-09-11 20:00', '2026-09-11 21:00',
                              '2026-09-14 01:00'])              # weekend gap
    frame = pd.DataFrame({'close': [1.0, 2.0, 3.0]}, index=pd.to_datetime(index))

    assert session_gaps(frame, '1h') == 1
    assert len(frame) == 3                                       # nothing added


def test_gold_trader_blocks_entries_during_an_event(config):
    """Exits stay allowed: being unable to leave is worse than not entering."""
    import copy

    from live.gold_trader import GoldLiveTrader

    cfg = copy.deepcopy(config)
    cfg['calendar'] = {'enabled': False}

    trader = GoldLiveTrader(cfg, state_dir='/tmp/gold_test_state')
    trader.blackouts = [{
        'start': pd.Timestamp('2026-09-16 13:30', tz='UTC'),
        'end': pd.Timestamp('2026-09-16 14:30', tz='UTC'),
        'title': 'FOMC Statement', 'country': 'USD', 'impact': 'High',
    }]

    assert trader.entry_blocked(pd.Timestamp('2026-09-16 14:00'))['title'] == 'FOMC Statement'
    assert trader.entry_blocked(pd.Timestamp('2026-09-16 16:00')) is None
    assert trader.blocked_entries == 1


def test_next_bar_message_does_not_point_into_the_past():
    """
    Gold closes at weekends. Projecting the next bar from the last one then runs
    into the past, and an idle log line saying 'next due' at a time already gone
    reads like a broken clock rather than a shut market.
    """
    import copy

    from live.trader import LiveTrader

    frame = pd.DataFrame(
        {'close': [1.0, 2.0]},
        index=pd.to_datetime(['2020-01-01 00:00', '2020-01-01 01:00']))

    trader = LiveTrader.__new__(LiveTrader)
    trader.interval = '1h'

    assert trader._next_bar_due(frame) == 'market closed - awaiting next session'


def test_next_bar_message_is_a_timestamp_when_the_market_is_open():
    from live.trader import LiveTrader

    now = pd.Timestamp.now()
    frame = pd.DataFrame({'close': [1.0]}, index=pd.DatetimeIndex([now]))

    trader = LiveTrader.__new__(LiveTrader)
    trader.interval = '1h'

    assert 'market closed' not in trader._next_bar_due(frame)


def test_goldbot_stop_unloads_rather_than_signalling():
    """
    `launchctl stop` only sends SIGTERM, and KeepAlive then restarts the job -
    so stop reported success while the bot kept running. Only unloading removes
    it from launchd, which is the only way a stop stays stopped.
    """
    import pathlib

    script = pathlib.Path('scripts/goldbot.sh').read_text()

    stop_block = script.split('  stop)')[1].split(';;')[0]
    assert 'launchctl unload' in stop_block
    assert 'launchctl stop' not in stop_block

    start_block = script.split('  start)')[1].split(';;')[0]
    assert 'launchctl load' in start_block
