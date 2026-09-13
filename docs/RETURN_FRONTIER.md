# What return is actually achievable

Written for anyone who asks this system for 4-5% per week. Every number is
measured, and reproducible:

```bash
python tools/fetch_multi_funding.py --out data/perp_funding.csv
python carry_backtest.py --data data/perp_funding.csv
pytest tests/test_risk_budget.py -q
```

---

## 1. The funding market is capped

Across 16 perpetuals and ~16,000 settled 8-hour intervals:

| Percentile | Funding / 8h | Annualized |
|---|---|---|
| 50th | +0.0023% | +2.5% |
| 90th | +0.0100% | +11.0% |
| 99th | +0.0100% | +11.0% |
| 99.9th | +0.0139% | +15.2% |

The 90th and 99th percentiles are identical because Binance caps the basic
funding rate at 0.01%/8h and the cap binds. **The extreme tail of the entire
funding market pays 15% a year.** No selection skill extracts 4%/week from that.

## 2. What 4-5%/week costs

Best measured carry is 0.069%/week unlevered, so the target has to come from
leverage. Against 1,997 real 8-hour BTC moves (σ = 1.30%):

| Target/week | Leverage | Liquidation move | P(ruin)/week | Expected survival |
|---|---|---|---|---|
| 0.5% | 7x | 13.30% | 0.00% | >10 years |
| 1.0% | 14x | 6.40% | 3.11% | 222 days |
| 2.0% | 29x | 2.95% | 29.5% | 20 days |
| 4.0% | 58x | 1.22% | 93.7% | 2.7 days |
| **4.5%** | **65x** | **1.03%** | **96.8%** | **2.2 days** |

A 1.03% adverse move is **0.79 standard deviations**. 4-5%/week is not an
aggressive strategy; it is a 97% chance of zero inside seven days. This is
asserted as a test, not as prose: `test_65x_is_near_certain_ruin_within_a_week`.

## 3. Why Sharpe-based sizing kills carry strategies

The funding book has a weekly Sharpe above 11 in places. Kelly, or any
volatility-based rule, reads that as licence for enormous leverage. It is not:
**the standard deviation of a carry return series does not contain the jump that
ends it.** Sizing therefore comes from an explicit ruin budget against the
*empirical* tail (`strategies/risk_budget.py`), never from σ.

## 4. And even ruin-budgeted leverage is not enough

The sharpest lesson in this project. Sizing each sleeve at the maximum its own
historical tail permitted (12x) produced **+93.5% annualized in-sample** and
**-10.5% out-of-sample**:

| Leverage cap | Annualized (walk-forward) | Max DD | Liquidations |
|---|---|---|---|
| 3x | +15.43% | -1.13% | 0 |
| **5x** | **+26.93%** | **-1.89%** | **0** |
| 8x | -18.19% | -33.47% | 1 |
| 12x (ruin-budgeted) | -14.21% | -33.56% | 1 |

At 8x and above, fold 3 takes a single liquidation and gives back every gain from
the other three folds. The ruin budget said 0.00% probability — computed on
training data that did not contain the move that then happened.

**Leverage fitted to historical tails does not protect you from the tail you have
not seen.** The shipped cap is 5x, deliberately below what the risk model alone
would allow.

## 5. The measured frontier

| Strategy | Annualized | Sharpe | Max DD | Positive weeks | Deflated Sharpe |
|---|---|---|---|---|---|
| **16-symbol carry book, 5x** | **+26.93%** | 3.66 | -1.89% | 71.1% | 0.84 |
| BTC single-sleeve carry | +19.31% | 6.99 | -2.12% | 81.6% | 1.00 |
| RSI rule (15m price) | -6.40% | -0.13 | -22.99% | — | — |
| Buy & hold | -24.73% | -0.25 | -51.60% | — | — |
| RL agent (15m price) | -43.03% | -2.19 | -50.04% | — | — |

**The frontier is roughly 20-27% a year at 0.35-0.46% a week**, with drawdowns
around 2% and no liquidations. That is a genuinely good risk-adjusted return. It
is 10x below the request.

## 6. Where higher returns genuinely exist

They exist. They are not available to a single-venue retail setup, and the reason
is always capacity:

- **Latency-sensitive market making** — real returns, requires colocation, and
  the edge is measured in microseconds against firms that have spent years on it.
- **Cross-venue arbitrage** — requires capital sitting on several exchanges
  simultaneously, so the capital requirement scales with the number of venues
  while the opportunity does not.
- **New-listing and illiquid-token inefficiencies** — the highest returns in
  crypto, and they saturate at a few hundred thousand dollars. A strategy that
  cannot absorb capital is never sold to you; it is run quietly on the owner's
  own money.

That last point is the general rule: **anything genuinely returning 4-5% a week
is capacity-constrained, which is exactly why it is not being offered to you.**

## 7. Account size

Binance minimums, checked live:

| Constraint | Value |
|---|---|
| Spot minimum notional | $5.00 |
| Futures minimum lot | 0.001 BTC (~$77) |
| Futures minimum notional | $50.00 |
| **Delta-neutral carry, both legs** | **~$82 minimum** |

Below roughly $200 the strategy cannot be opened at all, and below ~$500 fees
consume the return. At $20, 4.5%/week is $0.90 — and one taker round trip on $20
of notional costs $0.06, so fifteen round trips consume the entire weekly target.

| Capital | What runs |
|---|---|
| $20 | Nothing — below exchange minimums |
| ~$500 | One sleeve with a safe buffer, ~$95/year |
| ~$2,000 | Five-sleeve book |
| ~$10,000 | The strategy as designed, ~$2,700/year at 26.9% |
