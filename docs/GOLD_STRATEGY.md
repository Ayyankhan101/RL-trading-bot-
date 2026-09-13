# Gold: why it works where BTC could not, and what it can actually pay

All figures reproducible:

```bash
python tools/fetch_gold.py
python tools/signal_analysis.py --data data/GOLD_1h.csv
python backtest.py --data data/GOLD_1h.csv  --output-dir results/gold_1h
python backtest.py --data data/GOLD_15m.csv --output-dir results/gold_15m --folds 3
python -m strategies.account_sizing --equity 20
```

---

## 1. The cost structure is the entire difference

| Series | Median 15m move | Round trip | Edge / cost |
|---|---|---|---|
| BTC (Binance taker) | 0.069% | 0.300% | 0.23x |
| Gold PAXG (Binance) | 0.032% | 0.300% | 0.11x |
| **Gold (CFD broker)** | **0.074%** | **0.005%** | **14.8x** |

A 0.01 gold lot round trip is $0.15 spread + $0.07 commission = **$0.22**, which
is 0.005% of notional. Crypto is 0.30%. Gold moves 0.84x as much as BTC for 1/60
of the cost. This is asserted as a test
(`test_cfd_round_trip_matches_broker_arithmetic`), not as a claim.

Gold's signal is also stronger: information coefficient **-0.115** at a 4-day
horizon against BTC's -0.036.

## 2. Walk-forward results

**Hourly, 13,365 bars, 2024-04 to 2026-09:**

| Strategy | Return | Sharpe | Max DD | Calmar | Win rate |
|---|---|---|---|---|---|
| RL agent | +27.26% | 1.43 | -10.65% | 2.00 | 50.0% |
| **Buy & hold** | **+58.30%** | 1.35 | -29.00% | 1.53 | — |
| RSI rule | +30.80% | 1.13 | -18.97% | 1.26 | 48.1% |
| Mean reversion | +28.89% | 1.08 | -22.79% | 0.99 | 42.9% |

Deflated Sharpe **0.54** against a luck threshold of 1.49 — below it.

The agent made money, which no price strategy in this project had done before.
It also lost to buy & hold, because gold rallied 58% over the window. On
risk-adjusted terms it is competitive: **a third of the drawdown for half the
return**, and a better Sharpe and Calmar than holding.

**15-minute, 4,487 bars, 2 months:**

| Strategy | Return | Sharpe | Max DD | Win rate |
|---|---|---|---|---|
| RL agent | +7.56% | 5.28 | -3.08% | 91.7% |
| Buy & hold | +7.18% | 3.26 | -8.41% | — |

Deflated Sharpe **0.45** against a threshold of 5.81 — below it, and a 91.7% win
rate over 48 trades in 2 months is exactly the shape overfitting takes. **The 15m
result is not established.** It is promising and it is not evidence.

## 3. Why 4-5% per week is not a leverage problem

The decisive point, and the reason no amount of clever design reaches it:

**Leverage cannot improve the Calmar ratio.** Multiplying returns by L multiplies
drawdown by L too. Calmar is leverage-invariant — it is a property of the edge,
not of the position size.

So a weekly return target is really a Calmar requirement:

| Weekly | Annual | Calmar needed at -20% DD | at -30% | at -50% |
|---|---|---|---|---|
| 0.50% | 29.6% | 1.5 | 1.0 | 0.6 |
| 0.67% | 41.5% | 2.1 | 1.4 | 0.8 |
| 1.00% | 67.8% | 3.4 | 2.3 | 1.4 |
| 2.00% | 180.0% | 9.0 | 6.0 | 3.6 |
| **4.50%** | **886.4%** | **44.3** | **29.5** | **17.7** |

Against what is actually achieved:

| | Calmar |
|---|---|
| Typical good hedge fund | 1.0 - 2.0 |
| Exceptional / top decile | 2.0 - 3.0 |
| **This gold strategy (measured)** | **2.00** |
| Renaissance Medallion (estimated) | ~3 - 4 |

4.5%/week needs a Calmar around 30. The best sustained record in the industry is
roughly 4. That is not a gap a smarter design closes; it is an order of
magnitude beyond the frontier.

## 4. What leverage does to this actual strategy

Applying leverage to the real walk-forward equity curve, not a model of it:

| Leverage | Weekly | Annual | Max DD | Liquidation distance | Outcome | $20 after 99 weeks |
|---|---|---|---|---|---|---|
| 1x | 0.243% | 13.5% | -10.65% | 99.50% | survives | $25.45 |
| 2x | 0.466% | 27.4% | -20.35% | 49.50% | survives | $31.74 |
| **3x** | **0.670%** | **41.5%** | **-29.14%** | **32.83%** | **survives** | **$38.79** |
| 5x | — | — | -44.30% | 19.50% | **WIPED OUT** | $0.00 |
| 12x | — | — | -77.66% | 7.83% | **WIPED OUT** | $0.00 |

4.5%/week requires **18.5x**. The strategy's own historical drawdown of -10.65%
becomes -197% at that leverage. It is liquidated by its own past.

**Maximum survivable leverage on this edge is 3x.**

## 5. Sizing a $20 cent account

```
account type               lots    notional  leverage   wipeout  bar % eq  bars to 0
standard (100 oz/lot)     0.010$   4,408.90    220.4x     0.45%    37.48%       2.7  <-- RECKLESS
cent (1 oz/lot)           0.010$      44.09      2.2x    45.36%     0.37%     266.8
```

The cent account is what makes $20 workable at all — on a standard account the
broker's *smallest* position is 220x leverage and 2.7 adverse bars from zero.

Minimum equity for a standard 0.01 lot to be sane: **$882 at 5x, $2,204 at 2x.**
On a cent account: $8.82 and $22.04.

**Swap-free helps and is worth having.** Gold's measurable edge sits at a
multi-day horizon, and an Islamic account carries that for free.

## 6. The honest number

On a $20 swap-free cent account, at the maximum leverage this edge survives
(3x): **roughly 0.67% per week, 41.5% annualized.** Twenty dollars becomes about
$39 over two years.

That is a genuinely strong annualized figure. It is 1/7th of the target, and the
gap is in the edge, not the execution.

---

## 7. Adaptive regime scaling: tested, and it made things worse

The question was whether the strategy can adapt to market structure, volatility
and news. Measured answer, in three parts.

### What is genuinely forecastable

On 2.4 years of hourly gold:

| | 1h lag | 24h lag | 96h lag |
|---|---|---|---|
| Autocorrelation of **returns** | -0.029 | -0.007 | +0.006 |
| Autocorrelation of **volatility** | **+0.989** | **+0.636** | **+0.437** |

Direction is unforecastable. Volatility is highly forecastable - Engle's
volatility clustering, the most reliable regularity in financial time series.

So no system can anticipate a central-bank decision or a geopolitical shock. It
can measure the volatility already in progress and react to that.

### What was built

`strategies/regime.py` and `live/calendar.py`:

- **Volatility-scaled exposure** - `target_vol / realized_vol`, lagged one bar.
  Measured effect on gold: calm regimes ran 0.750 of maximum exposure, normal
  0.599, stressed **0.216**. A shock shrinks the position without predicting it.
- **Scheduled-event blackout** - high-impact events from a live economic
  calendar, ±30 minutes, cached to disk with a fallback source.
- **Spread guard** - refuses a quote more than 3x normal. Gold spreads widen
  10-50x through an FOMC print, and the cost is observable *before* trading.
- **Trend/range classification** - Kaufman efficiency ratio; a mean-reversion
  edge is halved in a trending regime, where it inverts.

### What it did to returns

| | No regime | Regime, no deadband | Regime + deadband |
|---|---|---|---|
| Total return | **27.26%** | 9.46% | 9.59% |
| Annualized | **21.31%** | 7.51% | 7.61% |
| Sharpe | **1.43** | 0.53 | 0.59 |
| Max drawdown | **-10.65%** | -14.07% | -12.68% |
| Calmar | **2.00** | 0.53 | 0.60 |
| Round trips | 160 | 624 | 64 |
| Profit factor | **1.83** | 0.90 | 1.41 |
| Deflated Sharpe | **0.54** | 0.19 | 0.22 |

The first attempt quadrupled the trade count: continuously scaled exposure
rebalances on every small drift in the target. A 25% rebalance deadband fixed
that - 624 trades down to 64, profit factor back above 1 - **and it still lost**.

The likely reason is specific to this sample: volatility scaling treats high
volatility as danger, but gold spent the window in a strong bull market where
the high-volatility periods were exactly where the gains were. Cutting exposure
there cut the winners.

**It ships disabled.** The mechanism is sound and may pay on a different asset or
a different regime, but it has to be re-measured before being switched on, not
assumed to help because it sounds sophisticated.
