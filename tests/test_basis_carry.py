"""
Funding-rate carry: P&L mechanics, risk model, and the assumptions that decide
whether the result is real.
"""

import numpy as np
import pandas as pd
import pytest

from strategies.basis_carry import (
    BasisConfig,
    liquidation_move,
    run_basis_carry,
    trailing_funding,
    weekly_statistics,
)


def make_data(n=300, funding=0.0001, basis=0.0, price=100_000.0, seed=0):
    """Synthetic funding series with a controllable rate and basis."""
    index = pd.date_range('2026-01-01', periods=n, freq='8h')
    rng = np.random.default_rng(seed)
    spot = price * np.cumprod(1 + rng.normal(0, 0.01, n))

    basis_series = np.full(n, basis) if np.isscalar(basis) else np.asarray(basis)
    return pd.DataFrame({
        'funding_rate': np.full(n, funding) if np.isscalar(funding) else np.asarray(funding),
        'spot_close': spot,
        'perp_close': spot * (1 + basis_series),
        'basis': basis_series,
    }, index=index)


# ------------------------------------------------------------------- signal

def test_trailing_funding_is_lagged():
    """
    The decision for interval t may only use funding that has already settled.
    Including the current rate is reading the payment before deciding to be
    there to receive it.
    """
    rates = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05])
    trailing = trailing_funding(rates, lookback=2)

    assert np.isnan(trailing.iloc[0])
    assert trailing.iloc[2] == pytest.approx(np.mean([0.01, 0.02]))


# --------------------------------------------------------------------- P&L

def test_funding_is_the_return_source():
    """Flat basis, positive funding: equity grows by funding alone."""
    data = make_data(n=100, funding=0.0001, basis=0.0)
    result = run_basis_carry(data, BasisConfig(leverage=1.0, fee_per_leg=0.0))

    assert result.funding_collected > 0
    assert result.basis_pnl == pytest.approx(0.0, abs=1e-6)
    assert result.equity.iloc[-1] > result.equity.iloc[0]


def test_price_direction_does_not_matter():
    """
    The whole point: the position is delta neutral. A crashing market and a
    rallying one must produce the same P&L when funding and basis are identical.
    """
    up = make_data(n=200, funding=0.0001, basis=0.0, seed=1)
    down = up.copy()
    down['spot_close'] = up['spot_close'].iloc[0] * 0.5      # halve the price
    down['perp_close'] = down['spot_close']

    config = BasisConfig(leverage=3.0, fee_per_leg=0.0)
    assert run_basis_carry(up, config).equity.iloc[-1] == pytest.approx(
        run_basis_carry(down, config).equity.iloc[-1])


def test_negative_funding_loses_money():
    data = make_data(n=100, funding=-0.0001, basis=0.0)
    # Threshold disabled so the position is held through the negative regime.
    config = BasisConfig(leverage=1.0, fee_per_leg=0.0,
                         entry_threshold=-1.0, exit_threshold=-1.0)
    result = run_basis_carry(data, config)

    assert result.funding_collected < 0
    assert result.equity.iloc[-1] < result.equity.iloc[0]


def test_widening_basis_is_a_loss_for_the_short_leg():
    """A short perp gains as the spread narrows and loses as it widens."""
    widening = np.linspace(0.0, 0.004, 200)
    data = make_data(n=200, funding=0.0, basis=widening)
    result = run_basis_carry(data, BasisConfig(leverage=1.0, fee_per_leg=0.0,
                                               entry_threshold=-1.0,
                                               exit_threshold=-1.0))

    assert result.basis_pnl < 0


def test_narrowing_basis_is_a_gain():
    narrowing = np.linspace(0.004, 0.0, 200)
    data = make_data(n=200, funding=0.0, basis=narrowing)
    result = run_basis_carry(data, BasisConfig(leverage=1.0, fee_per_leg=0.0,
                                               entry_threshold=-1.0,
                                               exit_threshold=-1.0))

    assert result.basis_pnl > 0


def test_basis_is_marked_every_interval_not_at_exit():
    """
    Booking basis P&L only on exit hides the drawdown a widening basis causes
    while the position is open, and dumps months of movement onto one bar.
    """
    widening = np.concatenate([np.zeros(50), np.linspace(0, 0.01, 150)])
    data = make_data(n=200, funding=0.0, basis=widening)
    result = run_basis_carry(data, BasisConfig(leverage=1.0, fee_per_leg=0.0,
                                               entry_threshold=-1.0,
                                               exit_threshold=-1.0))

    equity = result.equity
    # The decline must be spread across the widening, not a single cliff.
    steps = equity.diff().dropna()
    assert (steps < 0).sum() > 50
    assert abs(steps.min()) < abs(equity.iloc[0] - equity.iloc[-1]) * 0.5


def test_leverage_scales_pnl():
    data = make_data(n=200, funding=0.0001, basis=0.0)
    one = run_basis_carry(data, BasisConfig(leverage=1.0, fee_per_leg=0.0))
    three = run_basis_carry(data, BasisConfig(leverage=3.0, fee_per_leg=0.0))

    assert three.funding_collected > 2.5 * one.funding_collected


def test_fees_are_charged_on_all_four_fills():
    """Two legs in and two legs out."""
    data = make_data(n=50, funding=0.0, basis=0.0)
    result = run_basis_carry(data, BasisConfig(leverage=1.0, fee_per_leg=0.001,
                                              entry_threshold=-1.0))

    assert result.costs_paid == pytest.approx(10_000 * 0.001 * 4, rel=0.02)


# -------------------------------------------------------------------- risk

def test_liquidation_distance_shrinks_with_leverage():
    assert liquidation_move(1, 0.005) > liquidation_move(10, 0.005)
    assert liquidation_move(107, 0.005) < 0.005


def test_107x_is_liquidated_by_a_sub_1pct_move():
    """
    The leverage that would turn 0.09%/week into 10%/week. BTC covers this
    distance in an hour, which is why the target is not reachable this way.
    """
    assert liquidation_move(107, 0.005) < 0.01


def test_liquidation_wipes_the_account():
    """A perp rally past the margin closes the short leg at the worst moment."""
    n = 60
    index = pd.date_range('2026-01-01', periods=n, freq='8h')
    perp = np.full(n, 100_000.0)
    perp[30:] = 100_000.0 * 1.5          # +50% gap, past a 10x liquidation

    data = pd.DataFrame({
        'funding_rate': np.full(n, 0.0001),
        'spot_close': np.full(n, 100_000.0),
        'perp_close': perp,
        'basis': perp / 100_000.0 - 1.0,
    }, index=index)

    result = run_basis_carry(data, BasisConfig(leverage=10.0, fee_per_leg=0.0))

    assert result.liquidations == 1
    assert result.equity.iloc[-1] == 0.0


def test_no_position_no_pnl():
    """Below the entry threshold the strategy stays flat rather than guessing."""
    data = make_data(n=100, funding=-0.001, basis=0.0)
    result = run_basis_carry(data, BasisConfig(leverage=3.0, entry_threshold=0.0))

    assert result.intervals_in_position < 5
    assert result.equity.iloc[-1] == pytest.approx(10_000.0, rel=0.01)


# ------------------------------------------------------------------ weekly

def test_weekly_statistics_reports_the_hit_rate():
    """The property this strategy is selected for is consistency, not size."""
    data = make_data(n=400, funding=0.0001, basis=0.0)
    result = run_basis_carry(data, BasisConfig(leverage=3.0, fee_per_leg=0.0))
    weekly = weekly_statistics(result.equity)

    assert weekly['positive_week_rate'] > 0.9
    assert weekly['weeks'] > 5
    assert weekly['worst_week'] <= weekly['best_week']


def test_missing_columns_raise():
    data = make_data(n=50).drop(columns=['basis'])
    with pytest.raises(KeyError):
        run_basis_carry(data, BasisConfig())


# ------------------------------------------------------- live paper trading

@pytest.fixture
def carry_config(config):
    import copy
    cfg = copy.deepcopy(config)
    cfg['basis'] = {'initial_capital': 10_000.0, 'leverage': 3.0, 'fee_per_leg': 0.0002,
                    'entry_threshold': 0.0, 'exit_threshold': -0.00005, 'lookback': 5,
                    'maintenance_margin': 0.005}
    return cfg


def test_live_trader_matches_the_backtest(carry_config, tmp_path, monkeypatch):
    """
    The live path must reproduce the backtest on identical data.

    Two implementations of the same strategy drift, and then a live week stops
    being comparable to a backtested one.
    """
    from live.basis_trader import LiveBasisTrader

    data = make_data(n=200, funding=0.0001, basis=0.0)
    trader = LiveBasisTrader(carry_config, state_dir=str(tmp_path))
    monkeypatch.setattr(trader, 'refresh_data', lambda pages=2: data)
    status = trader.tick(verbose=False)

    expected = run_basis_carry(data, BasisConfig.from_dict(carry_config))

    assert status['equity'] == pytest.approx(expected.equity.iloc[-1], rel=0.02)


def test_live_trader_resumes_after_restart(carry_config, tmp_path, monkeypatch):
    """A restart continues the same account, not a fresh flattering one."""
    from live.basis_trader import LiveBasisTrader

    data = make_data(n=200, funding=0.0001, basis=0.0)

    first = LiveBasisTrader(carry_config, state_dir=str(tmp_path))
    monkeypatch.setattr(first, 'refresh_data', lambda pages=2: data.iloc[:100])
    first.tick(verbose=False)
    midpoint = first.state.equity

    second = LiveBasisTrader(carry_config, state_dir=str(tmp_path))
    monkeypatch.setattr(second, 'refresh_data', lambda pages=2: data)
    second.tick(verbose=False)

    assert second.state.equity > midpoint
    assert second.state.intervals_seen == 200


def test_live_trader_does_not_reprocess_settled_intervals(carry_config, tmp_path,
                                                          monkeypatch):
    """Funding is paid once. Replaying an interval would double-count it."""
    from live.basis_trader import LiveBasisTrader

    data = make_data(n=120, funding=0.0001, basis=0.0)
    trader = LiveBasisTrader(carry_config, state_dir=str(tmp_path))
    monkeypatch.setattr(trader, 'refresh_data', lambda pages=2: data)

    trader.tick(verbose=False)
    equity_after_first = trader.state.equity
    status = trader.tick(verbose=False)

    assert status['new_intervals'] == 0
    assert trader.state.equity == pytest.approx(equity_after_first)
