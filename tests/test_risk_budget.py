"""Ruin-based leverage sizing."""

import numpy as np
import pytest

from strategies.risk_budget import (
    INTERVALS_PER_WEEK,
    INTERVALS_PER_YEAR,
    leverage_for_target,
    liquidation_distance,
    max_leverage,
    ruin_probability,
    ruin_table,
    survival_intervals,
)

# Real 8h BTC move distribution: sigma ~1.3%, fat right tail.
BTC_MOVES = np.abs(np.random.default_rng(0).standard_t(3, 2000) * 0.009)


def test_liquidation_distance_shrinks_with_leverage():
    assert liquidation_distance(1) > liquidation_distance(10) > liquidation_distance(65)


def test_65x_is_liquidated_inside_one_sigma():
    """The leverage 4.5%/week requires, against a 1.30% 8h standard deviation."""
    assert liquidation_distance(65, 0.005) < 0.013


def test_65x_is_near_certain_ruin_within_a_week():
    """
    The headline claim, as a test rather than prose: 4-5%/week is not a
    high-risk strategy, it is a near-certain loss on a one-week horizon.
    """
    weekly_edge = 0.00069                      # best measured carry, unlevered
    leverage = leverage_for_target(0.045, weekly_edge)
    ruin = ruin_probability(BTC_MOVES, leverage, 0.005, INTERVALS_PER_WEEK)

    assert leverage > 50
    assert ruin > 0.8


def test_low_leverage_survives():
    assert ruin_probability(BTC_MOVES, 5, 0.005, INTERVALS_PER_YEAR) < 0.05


def test_max_leverage_respects_the_ruin_budget():
    solved = max_leverage(BTC_MOVES, 0.005, budget=0.01,
                          horizon_intervals=INTERVALS_PER_YEAR, cap=50)
    assert ruin_probability(BTC_MOVES, solved, 0.005, INTERVALS_PER_YEAR) <= 0.01


def test_tighter_budget_gives_less_leverage():
    loose = max_leverage(BTC_MOVES, 0.005, budget=0.20, cap=50)
    tight = max_leverage(BTC_MOVES, 0.005, budget=0.001, cap=50)
    assert tight <= loose


def test_ruin_uses_empirical_tails_not_a_gaussian_fit():
    """
    A normal approximation understates ruin exactly where it matters. Carry
    strategies die in the tail, which is the part a Gaussian fit smooths away.
    """
    rng = np.random.default_rng(1)
    sigma = 0.013
    gaussian = np.abs(rng.normal(0, sigma, 20_000))
    fat = np.abs(rng.standard_t(2.5, 20_000) * sigma * 0.6)

    leverage = 20.0
    assert ruin_probability(fat, leverage, 0.005, INTERVALS_PER_YEAR) > \
        ruin_probability(gaussian, leverage, 0.005, INTERVALS_PER_YEAR)


def test_ruin_rises_with_horizon():
    short = ruin_probability(BTC_MOVES, 20, 0.005, INTERVALS_PER_WEEK)
    long = ruin_probability(BTC_MOVES, 20, 0.005, INTERVALS_PER_YEAR)
    assert long >= short


def test_survival_is_infinite_when_never_observed():
    assert np.isinf(survival_intervals(BTC_MOVES, 1.0))


def test_ruin_table_covers_every_target():
    rows = ruin_table(BTC_MOVES, weekly_edge=0.00069)
    assert len(rows) == 5
    assert rows[-1]['weekly_target'] == pytest.approx(0.045)
    # More return demands more leverage demands more ruin - monotone throughout.
    assert all(rows[i]['leverage'] < rows[i + 1]['leverage'] for i in range(len(rows) - 1))
    assert rows[-1]['ruin_per_week'] >= rows[0]['ruin_per_week']


def test_zero_budget_permits_no_leverage():
    assert max_leverage(BTC_MOVES, 0.005, budget=0.0) == 0.0
