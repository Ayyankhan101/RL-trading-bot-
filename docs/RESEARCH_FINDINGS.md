# Research findings: what this strategy can and cannot do

This document records what was measured, how, and what follows from it. Every
number here is reproducible from a committed script; none were typed by hand.

```bash
python tools/signal_analysis.py --data data/BTC_15m.csv   # -> results/signal_analysis.json
python tools/sweep.py --data data/BTC_15m.csv             # -> results/sweep.json
python backtest.py --data data/BTC_15m.csv                # -> results/metrics.json
```

---

## 1. Is there a signal?

**Yes, and it is mean reversion.** Information coefficient — the Spearman
correlation between a feature and the forward return — measured over 69,727
15-minute bars (2024-09 to 2026-09):

| Horizon | mean \|IC\| | Strongest features |
|---|---|---|
| 4 bars (1h) | 0.0309 | stoch_k −0.045, close_over_EMA_12 −0.044, rsi −0.042 |
| 24 bars (6h) | 0.0254 | price_momentum_96 −0.036, close_over_SMA_200 −0.033 |
| 96 bars (24h) | 0.0088 | price_momentum_384 −0.023 |
| 192 bars (48h) | 0.0093 | bb_width +0.037, price_momentum_384 −0.037 |

In quantitative equity practice (Grinold & Kahn, *Active Portfolio Management*),
\|IC\| ≈ 0.03 is a usable signal and below 0.01 is noise. These sit at the
usable-but-weak end.

**The signs are consistently negative**, which means strength predicts weakness:
when price sits above its moving average, or RSI is elevated, forward returns are
lower. That is mean reversion, not momentum.

**It is stable.** Splitting the data in half and measuring each feature's IC
separately, **10 of 23 features keep their sign with \|IC\| > 0.02 in both
halves**. A feature whose IC flips between halves is not a signal however large
it looks overall; these did not flip.

## 2. Is the signal bigger than the cost of trading it?

This is the question that decides everything, and for most configurations the
answer is no.

Method: enter on decile extremes of a signal, exit after a fixed horizon, and
compare the average gross return per trade against the round trip.

Costs on Binance BTCUSDT:
- **Taker** (market order): 0.10% fee + 0.05% slippage, both legs = **0.30%**
- **Maker** (limit order): 0.02% fee, no slippage, both legs = **0.04%**

| Signal | Hold | Gross/trade | Net taker | Net maker | Annualized (maker) |
|---|---|---|---|---|---|
| price_momentum_96 | 24 bars | +0.0612% | −0.2388% | **+0.0212%** | **+30.9%** |
| close_over_EMA_26 | 96 bars | +0.0988% | −0.2012% | **+0.0588%** | **+21.4%** |
| close_over_SMA_50 | 96 bars | +0.0772% | −0.2228% | **+0.0372%** | **+13.6%** |
| close_over_SMA_200 | 24 bars | +0.0448% | −0.2552% | +0.0048% | +7.0% |
| close_over_SMA_50 | 4 bars | +0.0089% | −0.2911% | −0.0311% | −272.5% |
| price_momentum_384 | 192 bars | −0.1023% | −0.4023% | −0.1423% | −26.0% |

**0 of 20 signal/horizon pairs clear taker costs. 7 of 20 clear maker costs.**

Two conclusions follow directly:

1. **Maker execution is mandatory, not an optimization.** At 0.30% a round trip,
   the gross edge is 4x to 30x too small. No model fixes that; it is arithmetic.
2. **Short horizons are hopeless.** At 4 bars the edge is +0.009% against a
   0.04% maker cost. The horizon has to be hours, not minutes, for the signal to
   outgrow its own cost.

## 3. Why the previous version lost 65%

Re-pricing the 832 realized round trips from the taker run at different cost
levels:

| Cost level | Total P&L | Per trade | Win rate |
|---|---|---|---|
| Taker (as run) | −$8,930 | −$10.73 | 29.0% |
| Maker | −$3,259 | −$3.92 | 45.2% |
| Zero | −$1,841 | −$2.21 | 52.5% |

Costs were **53% of the loss**. But note the last row: at *zero* cost the policy
still lost. Execution was the larger problem and the policy was the smaller one,
and both had to be addressed.

Trade-level diagnosis of that run:

| | Before fixes | After reward/stop fixes |
|---|---|---|
| Payoff (avg win / avg loss) | 1.40 | 2.31 |
| Hit rate needed to break even | 41.7% | 30.2% |
| Hit rate achieved | 30.7% | 29.0% |
| **Gap to break-even** | **11.0 pts** | **1.2 pts** |
| Stop-loss exits | 245 | 34 |
| Median holding period | 24 bars (the floor) | 228 bars |

The 3×ATR stop sat *inside* the range 15m noise covers routinely — median 6-hour
move 0.39%, single-bar range 0.19%, stop distance 0.57%. Positions were killed by
noise before any move developed. Widening it cut stop-outs from 245 to 34.

## 4. What was ruled out by measurement

- **Shorting.** Short-side gross return was **−0.018%/trade** against long-side
  **+0.066%** on the same signal. Tested, rejected, not built.
- **Taker execution at any horizon.** 0 of 20 pairs clear it.
- **Sub-hour holding.** Every 4-bar configuration is deeply negative after costs.
- **Cost-aware reward** (Zhang et al. 2020). Swept across 18 configurations and
  it was the worst of the three reward functions — it suppresses churn and the
  profitable trades together.
- **A C rewrite.** Profiling shows **99.1%** of runtime is already in PyTorch's
  C++/BLAS kernels; the Python env loop is 0.9%. A perfect C rewrite buys ~1%.

## 4b. The signal is tradeable; the RL policy does not capture it

This is the most important result in this document.

The idealized decile rule clears maker costs on 7 of 20 signal/horizon pairs,
the best at +30.9% annualized. But when the RL agent is given the same data,
the same maker execution and the same holding constraints, a 12-configuration
sweep produced:

| Execution | Mean return | Best | Positive configs |
|---|---|---|---|
| Maker | −5.09% | +2.05% | 1 of 6 |
| Taker | −5.30% | −0.69% | 0 of 6 |

Maker execution barely moves the *mean* (−5.09% vs −5.30%), even though the
arithmetic says it should be decisive. The conclusion is uncomfortable and
worth stating plainly: **the learned policy is not expressing the edge that is
demonstrably present in the features.** The bottleneck has moved from the signal
and from execution to the model.

That is why `backtest.py` now runs a **mean-reversion rule** as a baseline
alongside buy & hold and RSI — the same decile logic, with expanding-window
quantiles so it stays causal, and no model in between. If a learned policy
cannot beat the rule it was supposed to learn, the rule is the honest
deliverable.

## 4c. The idealized edge did not survive honest implementation

The decile test in section 2 said +30.9% annualized. Implemented properly it
lost 8.15% over the same out-of-sample window.

The difference is the three things the offline test quietly assumed:

1. **Whole-series quantiles.** The decile boundary was computed over all data,
   so every entry knew the full distribution of a signal it had not yet seen.
   The backtest uses expanding-window quantiles, and `test_mean_reversion_
   quantiles_are_causal` enforces it.
2. **Fixed-horizon exits with no interference.** The real strategy has stops, a
   minimum hold, and a drawdown halt, all of which cut trades short.
3. **Guaranteed fills.** The decile test assumed every signal became a position.
   Maker execution misses ~15% of them.

Out-of-sample results on 2025-01-18 to 2026-09-13 (57,906 bars):

| Strategy | Return | Sharpe | Max DD | Trades | Win rate |
|---|---|---|---|---|---|
| RSI rule | **−6.40%** | −0.13 | −22.99% | 298 | 26.5% |
| Mean-reversion rule | −8.15% | −0.24 | −29.02% | 320 | 25.9% |
| Buy & hold | −24.73% | −0.25 | −51.60% | 1 | — |
| RL agent | −43.03% | −2.19 | −50.04% | 1,316 | 42.7% |

Everything loses. The simple rules lose least, the RL agent loses most, and
**the RL agent is beaten by the rule it was meant to learn** as well as by doing
nothing.

## 4d. Results by market regime

The window above is a bear market, which flatters defensive strategies and
punishes long-only ones. Splitting by regime:

| Window | Regime | Buy & hold | Mean reversion | RSI |
|---|---|---|---|---|
| 2024-09 → 2025-01 | up | **+74.36%** | +21.29% | +32.75% |
| 2025-01 → 2025-07 | flat | +2.42% | **+4.13%** | −3.55% |
| 2025-07 → 2026-01 | down | −17.52% | −12.17% | **−7.03%** |
| 2026-01 → 2026-09 | flat | 0.00% | −3.47% | **+5.34%** |

This is the clearest statement of what these strategies are:

- **In a rising market they make money but badly underperform simply holding**
  (+21% against +74%). Being flat half the time costs most of the upside.
- **In a flat market they can add value** — mean reversion returned +4.13%
  against buy & hold's +2.42%, the one regime where the edge shows up.
- **In a falling market they lose less than holding**, but still lose. A
  long/flat strategy cannot profit from a decline; its best case is 0%.

There is no regime in which any tested configuration produces consistent
positive weekly returns.

## 5. Why 10-12% per week is not attainable

- 10%/week compounds to **1.10^52 = 142x per year (+14,100%)**.
- The best long-run record in the industry (Renaissance Medallion) is roughly
  **66%/year** gross.
- The measured per-trade edge here is **+0.02% to +0.10% gross**, and the best
  annualized net figure from the idealized decile test is **+30.9%** — before
  slippage variance, before walk-forward, before correcting for the fact that it
  was selected from 20 candidates.

The gap between those numbers is not a tuning problem and no reward function
closes it. A configuration that *displayed* 10%/week in a backtest would be a
bug or a curve fit, and the Deflated Sharpe machinery in this repo exists to
catch exactly that.

## 6. Guardrails against fooling ourselves

The easiest way to produce a profitable-looking backtest is to search until one
appears. These are in place to prevent it:

- **Walk-forward validation** with purge gaps between train and test windows, so
  no indicator lookback straddles the boundary.
- **A validation split that ends before the test region**, used for all tuning.
  The reported number is never the number that was optimized.
- **Deflated Sharpe Ratio** (Bailey & López de Prado 2014): the published Sharpe
  is corrected for how many configurations were searched. With 12-18 trials the
  luck threshold is an annualized Sharpe above 5 — verified on synthetic data to
  reject pure noise at DSR 0.005 while passing a real edge at 0.96.
- **No-lookahead tests**: replacing every bar after step *k* with garbage must
  leave all observations and rewards up to *k* bit-identical.
- **Shared execution and risk modules** between backtest and live, so a live fill
  matches what the backtest claimed it would be.
- **A fill model that misses.** Maker execution is modeled with real rejections
  (~15% of orders), because modeling the cheaper fee without the missed fills
  would manufacture the entire improvement.

## 7. Realistic target

Based on the idealized decile test, the plausible ceiling for this signal with
maker execution at a 6-24 hour horizon is **roughly +10% to +30% annualized
gross**, which after implementation slippage, walk-forward decay and
selection-bias correction should be expected to land **materially lower — quite
possibly near zero**.

That is the honest range. It is not 10% per week, and no amount of further
tuning on this feature set will make it so.

## 8. What would actually raise the ceiling

In descending order of expected value:

1. **Data nobody else is modeling** — order-book imbalance, perpetual funding
   rates, cross-exchange basis, on-chain flows. The IC ceiling on price-derived
   features here is ~0.04; new information is the only way past it.
2. **Funding-rate harvesting** — long spot, short perpetual, collect funding.
   Mechanically harvestable, historically 5-15%/year on BTC, and it does not
   require predicting price at all.
3. **Meta-labeling** (López de Prado, *Advances in Financial Machine Learning*,
   ch. 3) — a second model that sizes or vetoes the primary model's trades,
   which attacks precision rather than direction.
4. **Multi-asset training** so the policy cannot memorize one price path.
