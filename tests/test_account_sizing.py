"""Lot sizing against a real account: the $20 gold problem."""

import numpy as np
import pytest

from strategies.account_sizing import (
    CENT_LOT_DIVISOR,
    STANDARD_LOT_OZ,
    account_table,
    effective_leverage,
    lot_for_risk,
    minimum_viable_equity,
    notional,
    wipeout_move,
)

PRICE = 4408.90
CENT_CONTRACT = STANDARD_LOT_OZ / CENT_LOT_DIVISOR


def test_standard_lot_on_20_dollars_is_ruinous():
    """
    The finding that decides the whole question: the broker's *smallest*
    position is 220x leverage on a $20 account.
    """
    leverage = effective_leverage(20.0, 0.01, PRICE, STANDARD_LOT_OZ)
    wipe = wipeout_move(20.0, 0.01, PRICE, STANDARD_LOT_OZ)

    assert leverage > 200
    assert wipe < 0.005                       # half a percent ends the account


def test_cent_lot_keeps_the_same_trade_survivable():
    """Identical signal, identical broker - only the contract size differs."""
    leverage = effective_leverage(20.0, 0.01, PRICE, CENT_CONTRACT)
    wipe = wipeout_move(20.0, 0.01, PRICE, CENT_CONTRACT)

    assert leverage == pytest.approx(2.2, abs=0.2)
    assert wipe > 0.40


def test_one_bar_of_gold_noise_against_a_standard_lot():
    """A median 15m gold bar is over a third of a $20 standard-lot account."""
    bar_range = 0.0017
    rows = account_table(20.0, PRICE, bar_range, lots=(0.01,))
    standard = next(r for r in rows if 'standard' in r['account'])

    assert standard['bar_pct_of_equity'] > 0.30
    assert standard['bars_to_zero'] < 5


def test_account_table_flags_the_reckless_configuration():
    rows = account_table(20.0, PRICE, 0.0017, lots=(0.01,))
    reckless = [r for r in rows if r['leverage'] > 50]

    assert len(reckless) == 1
    assert 'standard' in reckless[0]['account']


def test_minimum_viable_equity_scales_with_target_leverage():
    at_five = minimum_viable_equity(PRICE, 0.01, 5.0, STANDARD_LOT_OZ)
    at_two = minimum_viable_equity(PRICE, 0.01, 2.0, STANDARD_LOT_OZ)

    assert at_two > at_five
    assert at_five == pytest.approx(881.78, rel=0.01)


def test_cent_account_needs_a_hundredth_of_the_equity():
    standard = minimum_viable_equity(PRICE, 0.01, 5.0, STANDARD_LOT_OZ)
    cent = minimum_viable_equity(PRICE, 0.01, 5.0, CENT_CONTRACT)

    assert standard / cent == pytest.approx(CENT_LOT_DIVISOR)


def test_risk_first_sizing_follows_the_stop():
    """Size follows the stop distance, never the other way round."""
    tight = lot_for_risk(20.0, PRICE, stop_pct=0.002, risk_per_trade=0.02)
    wide = lot_for_risk(20.0, PRICE, stop_pct=0.010, risk_per_trade=0.02)

    assert tight > wide                       # a tighter stop permits more size


def test_risk_first_sizing_respects_the_risk_budget():
    stop, risk, equity = 0.005, 0.02, 20.0
    lots = lot_for_risk(equity, PRICE, stop, risk)
    loss_at_stop = notional(lots, PRICE, STANDARD_LOT_OZ) * stop

    assert loss_at_stop == pytest.approx(equity * risk)


def test_zero_stop_gives_no_position():
    assert lot_for_risk(20.0, PRICE, stop_pct=0.0) == 0.0
