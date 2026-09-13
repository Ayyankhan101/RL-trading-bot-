"""Multi-sleeve carry book."""

import numpy as np
import pandas as pd
import pytest

from strategies.carry_book import (
    LONG_PERP,
    SHORT_PERP,
    CarryBookConfig,
    run_carry_book,
    solve_per_symbol_leverage,
    to_panels,
)


def make_book(symbols, n=200, funding=None, price_paths=None, basis=0.0):
    """Long-format dataset with controllable funding per symbol."""
    index = pd.date_range('2026-01-01', periods=n, freq='8h')
    frames = []
    for j, symbol in enumerate(symbols):
        rate = funding[symbol] if funding else 0.0001
        price = (price_paths[symbol] if price_paths
                 else np.full(n, 100.0 * (j + 1)))
        frames.append(pd.DataFrame({
            'timestamp': index,
            'symbol': symbol,
            'funding_rate': np.full(n, rate) if np.isscalar(rate) else rate,
            'spot': price,
            'perp': price * (1 + basis),
            'basis': np.full(n, basis),
        }))
    return pd.concat(frames, ignore_index=True)


def test_positive_funding_is_collected_by_shorting_the_perp():
    data = make_book(['AAA'], funding={'AAA': 0.0002})
    result = run_carry_book(data, CarryBookConfig(max_sleeves=1, leverage=1.0,
                                                  fee_per_leg=0.0))
    assert result.funding_collected > 0
    assert result.equity.iloc[-1] > result.equity.iloc[0]


def test_negative_funding_is_collected_by_the_mirror_sleeve():
    """
    The finding that tripled the book's return: assets paying negative funding
    are an opportunity, not a reason to sit out.
    """
    data = make_book(['AAA'], funding={'AAA': -0.0002})

    allowed = run_carry_book(data, CarryBookConfig(
        max_sleeves=1, leverage=1.0, fee_per_leg=0.0, allow_negative_funding=True))
    refused = run_carry_book(data, CarryBookConfig(
        max_sleeves=1, leverage=1.0, fee_per_leg=0.0, allow_negative_funding=False))

    assert allowed.funding_collected > 0
    assert allowed.equity.iloc[-1] > refused.equity.iloc[-1]


def test_selection_prefers_the_richest_funding():
    data = make_book(['LOW', 'HIGH'],
                     funding={'LOW': 0.00002, 'HIGH': 0.0005})
    result = run_carry_book(data, CarryBookConfig(max_sleeves=1, leverage=1.0,
                                                  fee_per_leg=0.0))
    opened = [e['symbol'] for e in result.events if e['action'] == 'ENTER']
    assert opened and set(opened) == {'HIGH'}


def test_selection_uses_only_settled_funding():
    """
    Ranking on the rate about to be paid is reading the payment before deciding
    to be there for it. A spike must not be captured on the interval it occurs.
    """
    n = 120
    rates = np.full(n, 0.00001)
    rates[60] = 0.01                      # one enormous interval
    data = make_book(['AAA'], n=n, funding={'AAA': rates})

    result = run_carry_book(data, CarryBookConfig(
        max_sleeves=1, leverage=1.0, fee_per_leg=0.0, entry_threshold=0.0005))
    entries = [e for e in result.events if e['action'] == 'ENTER']

    assert all(pd.Timestamp(e['timestamp']) > data.timestamp.iloc[60] for e in entries)


def test_one_sleeve_liquidation_does_not_wipe_the_book():
    """
    The structural reason to run a book. A single leveraged position that gets
    liquidated goes to zero; a sleeve takes 1/k of capital with it.
    """
    n = 80
    calm = np.full(n, 100.0)
    blowup = np.full(n, 100.0)
    blowup[40:] = 200.0                   # +100% gap against a short sleeve

    data = make_book(['CALM', 'BOOM'], n=n,
                     funding={'CALM': 0.0002, 'BOOM': 0.0002},
                     price_paths={'CALM': calm, 'BOOM': blowup})

    result = run_carry_book(data, CarryBookConfig(
        max_sleeves=2, leverage=10.0, fee_per_leg=0.0))

    assert result.liquidations == 1
    assert 'BOOM' in result.liquidated_symbols
    # Half the book survived, so equity is well above zero.
    assert result.equity.iloc[-1] > 4_000


def test_equal_weight_allocation_does_not_create_extra_leverage():
    data = make_book(['A', 'B', 'C', 'D'], funding={s: 0.0002 for s in 'ABCD'})
    result = run_carry_book(data, CarryBookConfig(max_sleeves=4, leverage=1.0,
                                                  fee_per_leg=0.0))
    # Four sleeves at 1/4 of capital each: the book cannot exceed its own equity.
    assert result.equity.iloc[0] <= 10_000.0


def test_per_symbol_leverage_is_sized_by_each_tail():
    """
    A book-wide leverage is sized by its worst member. Measured on real data,
    one altcoin's 33.67% move forced every sleeve, BTC included, down to 2.9x.
    """
    n = 400
    rng = np.random.default_rng(0)
    calm = 100.0 * np.cumprod(1 + rng.normal(0, 0.002, n))
    wild = 100.0 * np.cumprod(1 + rng.normal(0, 0.05, n))

    data = make_book(['CALM', 'WILD'], n=n,
                     price_paths={'CALM': calm, 'WILD': wild})
    leverages = solve_per_symbol_leverage(data, 0.005, budget=0.01, cap=12.0)

    assert leverages['CALM'] > leverages['WILD']


def test_book_with_one_sleeve_matches_a_single_position():
    """Generalizing to N sleeves must not change the k=1 behaviour."""
    data = make_book(['AAA'], funding={'AAA': 0.0001})
    result = run_carry_book(data, CarryBookConfig(max_sleeves=1, leverage=2.0,
                                                  fee_per_leg=0.0))

    intervals = result.sleeve_intervals
    expected = 10_000 * ((1 + 0.0001 * 2) ** intervals)
    assert result.equity.iloc[-1] == pytest.approx(expected, rel=0.02)


def test_fees_scale_with_sleeve_leverage():
    data = make_book(['AAA'], funding={'AAA': 0.0002})
    cheap = run_carry_book(data, CarryBookConfig(max_sleeves=1, leverage=1.0,
                                                 fee_per_leg=0.001))
    dear = run_carry_book(data, CarryBookConfig(max_sleeves=1, leverage=5.0,
                                                fee_per_leg=0.001))
    assert dear.costs_paid > cheap.costs_paid


def test_missing_columns_raise():
    data = make_book(['AAA']).drop(columns=['basis'])
    with pytest.raises(KeyError):
        to_panels(data)
