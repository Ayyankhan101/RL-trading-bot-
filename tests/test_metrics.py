"""Metric correctness against closed-form fixtures."""

import numpy as np
import pytest

from utils.metrics import (
    calculate_annualized_return,
    calculate_calmar_ratio,
    calculate_comprehensive_metrics,
    calculate_cvar,
    calculate_max_drawdown,
    calculate_profit_factor,
    calculate_returns,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_var,
    calculate_volatility,
    calculate_win_rate,
)

PPY_4H = 2190.0


def test_returns_from_equity_curve():
    returns = calculate_returns([100, 110, 99])
    np.testing.assert_allclose(returns, [0.1, -0.1])


def test_annualized_return_compounds():
    """10% over exactly half a year compounds to 21%, not 20%."""
    result = calculate_annualized_return(0.10, n_periods=int(PPY_4H / 2), periods_per_year=PPY_4H)
    assert result == pytest.approx(1.1 ** 2 - 1)


def test_annualized_return_handles_total_loss():
    assert calculate_annualized_return(-1.0, 100, PPY_4H) == -1.0


def test_sharpe_uses_the_supplied_annualization():
    """
    The same series must not report the same Sharpe at daily and 4-hourly
    frequency. Hardcoding 252 for 4h bars overstated Sharpe by ~2.9x.
    """
    rng = np.random.default_rng(0)
    returns = rng.normal(0.0002, 0.01, 5000)

    daily = calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=365)
    four_hourly = calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=PPY_4H)

    assert four_hourly == pytest.approx(daily * np.sqrt(PPY_4H / 365), rel=1e-9)


def test_sharpe_closed_form():
    returns = np.array([0.01, -0.01] * 50)
    expected = (np.mean(returns) - 0.0) / np.std(returns, ddof=1) * np.sqrt(PPY_4H)
    assert calculate_sharpe_ratio(returns, risk_free_rate=0.0,
                                  periods_per_year=PPY_4H) == pytest.approx(expected)


def test_sortino_downside_divides_by_all_periods():
    """Downside deviation averages squared shortfalls over N, not over N_negative."""
    returns = np.array([0.02, 0.02, 0.02, -0.01])
    expected_downside = np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2))
    expected = np.mean(returns) / expected_downside * np.sqrt(PPY_4H)

    assert calculate_sortino_ratio(returns, risk_free_rate=0.0,
                                   periods_per_year=PPY_4H) == pytest.approx(expected)


def test_sortino_flat_curve_is_zero():
    assert calculate_sortino_ratio([0.0] * 50, periods_per_year=PPY_4H) == 0.0


def test_max_drawdown_known_curve():
    curve = [100, 120, 60, 80, 200]
    result = calculate_max_drawdown(curve)
    assert result['max_drawdown'] == pytest.approx(-0.5)
    assert result['peak_index'] == 1
    assert result['trough_index'] == 2
    assert result['recovery_index'] == 4


def test_max_drawdown_monotonic_curve_is_zero():
    assert calculate_max_drawdown([1, 2, 3, 4])['max_drawdown'] == 0.0


def test_calmar_uses_annualized_not_total_return():
    """Feeding total return inflates Calmar by the number of years."""
    assert calculate_calmar_ratio(0.20, -0.10) == pytest.approx(2.0)


def test_var_and_cvar_are_positive_loss_magnitudes():
    returns = np.concatenate([np.full(95, 0.01), np.full(5, -0.20)])
    var = calculate_var(returns, 0.05)
    cvar = calculate_cvar(returns, 0.05)

    assert var > 0
    assert cvar >= var


def test_volatility_annualization():
    returns = np.array([0.01, -0.01] * 100)
    raw = calculate_volatility(returns, annualize=False)
    annual = calculate_volatility(returns, annualize=True, periods_per_year=PPY_4H)
    assert annual == pytest.approx(raw * np.sqrt(PPY_4H))


def test_win_rate_counts_round_trips_only():
    """
    Entry orders must not dilute the denominator.

    The old implementation counted every order, and tagged every buy with
    pnl = 0, which structurally capped win rate near 50%.
    """
    round_trips = [{'pnl': 5}, {'pnl': 3}, {'pnl': -2}, {'pnl': 8}]
    assert calculate_win_rate(round_trips) == pytest.approx(0.75)


def test_win_rate_ignores_entries_without_pnl():
    trades = [{'pnl': 5}, {'action': 'BUY'}, {'pnl': -1}]
    assert calculate_win_rate(trades) == pytest.approx(0.5)


def test_profit_factor_known_values():
    trades = [{'pnl': 10}, {'pnl': 5}, {'pnl': -5}]
    assert calculate_profit_factor(trades) == pytest.approx(3.0)


def test_profit_factor_undefined_without_losses():
    """None, not inf: inf only propagates into report formatting as nonsense."""
    assert calculate_profit_factor([{'pnl': 10}]) is None


def test_comprehensive_metrics_shape():
    curve = list(np.linspace(10_000, 12_000, 500))
    trades = [{'pnl': 100}, {'pnl': -50}, {'pnl': 20}]
    metrics = calculate_comprehensive_metrics(curve, trades, periods_per_year=PPY_4H)

    assert metrics['total_return'] == pytest.approx(0.2)
    assert metrics['total_trades'] == 3
    assert metrics['winning_trades'] == 2
    assert metrics['periods_per_year'] == PPY_4H
    assert metrics['max_drawdown'] == 0.0


def test_comprehensive_metrics_empty_curve():
    assert calculate_comprehensive_metrics([100.0]) == {}


def test_a_fold_with_no_trades_reports_zero_sharpe():
    """
    A flat curve has no risk to adjust for. Testing std against exact zero is
    not enough - a fold where the agent never traded leaves float noise in the
    standard deviation, which once produced a Sharpe of -5e17.
    """
    flat = [10_000.0] * 500

    assert calculate_sharpe_ratio(calculate_returns(flat), periods_per_year=8760) == 0.0
    assert calculate_sortino_ratio(calculate_returns(flat), periods_per_year=8760) == 0.0

    metrics = calculate_comprehensive_metrics(flat, periods_per_year=8760)
    assert abs(metrics['sharpe_ratio']) < 1.0
