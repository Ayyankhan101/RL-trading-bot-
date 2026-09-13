# Quantitative Trading Research: BTC, Gold, and Funding Carry

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/tests-255%20passing-green.svg)](tests/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A research repository that tests trading strategies honestly and reports what
> it finds, including when the finding is that a strategy does not work.

**Every number in this README is generated from files in `results/` by
`tools/render_readme_table.py`. None are typed by hand.**

---

## What works, ranked by evidence

| Strategy | Return (walk-forward) | Sharpe | Max DD | Deflated Sharpe | Verdict |
|---|---|---|---|---|---|
| **Funding carry (BTC)** | **+29.24%** | 6.99 | -2.12% | **1.00** | Validated |
| **Multi-sleeve carry book** | **+26.93%** | 3.66 | -1.89% | 0.84 | Strong |
| **Gold, hourly** | **+27.26%** | 1.43 | -10.65% | 0.54 | Promising, unproven |
| Gold, 15-minute | +7.56% | 5.28 | -3.08% | 0.45 | Too little data |
| RSI rule (BTC 15m) | -6.40% | -0.13 | -22.99% | — | Fails |
| **RL agent on BTC price** | **-43.03%** | -2.19 | -50.04% | 0.00 | **Fails** |

Deflated Sharpe (Bailey & López de Prado 2014) corrects for how many
configurations were searched. Only the funding carry clears its threshold.

## The one finding that explains all the others

**A strategy's edge has to exceed the cost of capturing it.** That single ratio
decided every result here:

| Market | Median 15m move | Round trip | Edge / cost | Outcome |
|---|---|---|---|---|
| BTC (Binance taker) | 0.069% | 0.300% | **0.23x** | Impossible |
| Gold, tokenized (Binance) | 0.032% | 0.300% | 0.11x | Impossible |
| **Gold (CFD broker)** | **0.074%** | **0.005%** | **14.8x** | Viable |

BTC price prediction at 15 minutes cannot work on a retail crypto exchange, and
no model fixes that — the agent must overcome a cost 4-30x larger than the
opportunity. Gold moves 0.84x as much for 1/60th of the cost, which is why the
same class of signal is tradeable there.

## What was tested and rejected

Each of these was built, measured, and turned off. The negative results are the
point, not an embarrassment:

| Idea | Result |
|---|---|
| Short selling | -0.018%/trade vs +0.066% long. Rejected. |
| Cost-aware reward (Zhang et al. 2020) | Worst of three reward functions across 18 configs. |
| Taker execution at any horizon | 0 of 20 signal/horizon pairs clear it. |
| Aggressive ruin-budgeted leverage | +93.5% in-sample, **-10.5%** out-of-sample. |
| Volatility regime adaptation | 27.26% → 9.59%, and drawdown got worse. |
| Rewriting in C | 99.1% of runtime is already in PyTorch's C++ kernels. |

## Documentation

| Document | Contents |
|---|---|
| [docs/RESEARCH_FINDINGS.md](docs/RESEARCH_FINDINGS.md) | Why BTC price prediction fails; the cost arithmetic |
| [docs/RETURN_FRONTIER.md](docs/RETURN_FRONTIER.md) | What return is achievable, and the ruin table |
| [docs/GOLD_STRATEGY.md](docs/GOLD_STRATEGY.md) | Gold economics, the $20 account problem, regime test |

## Quick start

```bash
pip install -r requirements.txt
pytest -q                                    # 255 tests

# Fetch data (nothing large is committed; it is all reproducible)
python tools/fetch_history.py --interval 15m --years 2 --out data/BTC_15m.csv
python tools/fetch_funding.py --out data/BTC_basis.csv
python tools/fetch_multi_funding.py --out data/perp_funding.csv
python tools/fetch_gold.py

# Validate
python tools/signal_analysis.py --data data/GOLD_1h.csv   # is there an edge?
python basis_backtest.py --data data/BTC_basis.csv        # funding carry
python carry_backtest.py --data data/perp_funding.csv     # multi-sleeve book
python backtest.py --data data/GOLD_1h.csv                # gold

# Live paper trading (no keys, no orders)
python live_basis.py --once      # BTC funding carry
python live_gold.py --once       # gold
python live_gold.py --status
```

---

## The working strategy: funding-rate carry

The price-prediction bot does not work (see below — it is kept because the
research that killed it is the point). **The strategy that does work makes no
price forecast at all.**

Hold long spot BTC and short an equal notional of the perpetual future. Price
moves cancel between the legs. Perpetual longs pay shorts a funding fee every 8
hours, and the short leg collects it.

```bash
python tools/fetch_funding.py --out data/BTC_basis.csv
python basis_backtest.py --data data/BTC_basis.csv   # walk-forward
python live_basis.py --once                          # live paper account
python live_basis.py --status
```

Walk-forward, out-of-sample, 4 folds over 22 months of real Binance funding:

| Metric | Funding carry | RL agent (15m) |
|---|---|---|
| Total return | **+29.24%** | -43.03% |
| Annualized | **+19.31%** | -28.86% |
| Sharpe | **6.99** | -2.19 |
| Max drawdown | **-2.12%** | -50.04% |
| Positive weeks | **62/76 = 81.6%** | — |
| Mean week | **+0.339%** | — |
| Worst week | **-0.479%** | — |
| **Deflated Sharpe** | **1.00** | 0.00 |

Per fold: +8.61%, +8.95%, +0.99%, +8.15%. No liquidations.

**This is the only strategy in the repo that clears the selection-bias
threshold.** At 3x leverage the P&L decomposes to funding +$3,048, basis +$35,
fees -$87 — the near-zero basis term is the delta-neutrality doing its job.

### What it does not do

- **0.34%/week, not 10%/week.** Reaching 10%/week needs **107x leverage**, which
  is liquidated by a 0.43% adverse move. BTC covers that in an hour.
- **Funding is decaying**: 2024 had 99% of intervals positive, 2025 87%,
  **2026 73%**. Assume forward returns below the backtest.
- **A Sharpe of 7 hides tails the sample does not contain.** No exchange failed
  during these 22 months. FTX depositors had excellent Sharpes too.
- 22 months, one asset, one venue.

---

## Read this first: what the data actually supports

Full evidence in **[docs/RESEARCH_FINDINGS.md](docs/RESEARCH_FINDINGS.md)**,
reproducible via `python tools/signal_analysis.py`.

**There is a real signal, and it is mean reversion.** 10 of 23 features keep
their information-coefficient sign with |IC| > 0.02 across both halves of the
data, all negative — strength predicts weakness.

**The signal is smaller than the cost of trading it, unless you post limits.**

| | Round trip | Signal/horizon pairs that clear it |
|---|---|---|
| Taker (market orders) | 0.30% | **0 of 20** |
| Maker (limit orders) | 0.04% | **7 of 20** |

The measured gross edge is +0.02% to +0.10% per trade. That is why the taker
version lost 65%: it was paying 0.30% to capture 0.06%. Maker execution is
mandatory arithmetic, not an optimization — and it is modeled honestly here,
with real missed fills (~15% of orders), because a cheaper fee without the
misses would manufacture the entire improvement.

**Realistic ceiling: roughly +10% to +30% annualized gross**, before
implementation slippage, walk-forward decay, and correction for selection bias —
so expect materially lower, possibly near zero. Not 10% per week. 10%/week
compounds to 142x/year against a ~66%/year best-in-industry record, on a measured
edge 4-30x below taker costs.

## Why 15 minutes is the hard case

Measured on 3,000 real 15m BTCUSDT bars:

| | |
|---|---|
| Median per-bar move | **0.080%** |
| Median bar range (high−low) | 0.192% |
| Round-trip cost (0.10% fee + 0.05% slippage, both legs) | **0.30%** |
| Bars whose move exceeds the cost of trading them | **9.4%** |

| Hold length | Median absolute move | vs 0.30% cost |
|---|---|---|
| 1 bar (15m) | 0.080% | 0.27x |
| 4 bars (1h) | 0.159% | 0.53x |
| 8 bars (2h) | 0.222% | 0.74x |
| **24 bars (6h)** | **0.389%** | **1.30x** |
| 96 bars (24h) | 0.922% | 3.07x |

At this resolution the fee structure is the central fact about the strategy, not
a footnote. A policy free to trade every bar pays 0.30% for an average 0.08%
opportunity, 96 times a day — that is a losing machine regardless of how good the
signal is. Two settings exist to make the arithmetic survivable:

- **`min_hold_bars: 24`** — discretionary exits are blocked for 6 hours, the
  horizon at which the median move finally exceeds its own cost. Stops are
  exempt; risk control is not a trading signal.
- **ATR-scaled stops** — `3 × ATR` and `6 × ATR` at entry, roughly 0.57% and
  1.14% on current 15m volatility. The old fixed 5%/15% levels were written for
  4-hour bars; a 15% target is weeks away on a 15m chart and would never fire.

Neither supplies an edge. They stop the cost structure from guaranteeing a loss
before the model gets a say.

## 📊 Results

<!-- RESULTS:START -->
_Generated by `backtest.py` on 2026-09-13T17:08:40 (commit 8c84f8cd)._

**Protocol:** walk-forward, 5 folds. Each fold trains only on bars preceding its test window, with a 200-bar purge gap between them; the reported curve is the concatenation of the untouched test segments.

**Data:** data/BTC_15m.csv — 69,727 15-minute bars, 2024-09-17 to 2026-09-13. Out-of-sample span: 2025-01-18 to 2026-09-13.

**Execution:** maker — 0.020% fee per fill and no slippage (a resting limit fills at its own price), charged on entry and exit. Agent variant `double`, seed 42.

### Out-of-sample results

| Metric | RL Agent |
|---|---|
| Total Return | -43.03% |
| Annualized Return | -28.86% |
| Sharpe Ratio | -2.19 |
| Sortino Ratio | -3.02 |
| Max Drawdown | -50.04% |
| Calmar Ratio | -0.58 |
| Round Trips | 1,316 |
| Win Rate | 42.7% |
| Profit Factor | 0.51 |

### Comparison with benchmarks

| Strategy | Return | Annualized | Sharpe | Max DD | Calmar | Round Trips | Win Rate |
|---|---|---|---|---|---|---|---|
| RL Bot | -43.03% | -28.86% | -2.19 | -50.04% | -0.58 | 1,316 | 42.7% |
| Buy & Hold | -24.73% | -15.80% | -0.25 | -51.60% | -0.31 | 1 | 0.0% |
| RSI Strategy | -13.11% | -8.15% | -0.51 | -22.91% | -0.36 | 602 | 22.8% |

### Per-fold out-of-sample results

| Fold | Test window | Return | Sharpe | Max DD | Round Trips | Win Rate |
|---|---|---|---|---|---|---|
| 1 | 2025-01-18 → 2025-05-19 | -13.54% | -2.63 | -22.02% | 318 | 50.3% |
| 2 | 2025-05-19 → 2025-09-17 | -6.30% | -1.77 | -9.48% | 123 | 22.0% |
| 3 | 2025-09-17 → 2026-01-16 | -5.10% | -0.89 | -14.78% | 225 | 56.9% |
| 4 | 2026-01-16 → 2026-05-17 | -10.97% | -2.87 | -14.00% | 311 | 50.8% |
| 5 | 2026-05-17 → 2026-09-13 | -16.76% | -3.03 | -25.95% | 339 | 26.3% |
<!-- RESULTS:END -->

### Live paper-trading demo

<!-- LIVE:START -->
_Snapshot of `results/live/status.json`, updated 2026-09-13T16:15:54._

Paper account trading **BTCUSD 15m** bars from `binance`, model origin `bootstrap`. Last bar 2026-09-13 10:45:00 at $76,677.29.

| Metric | Value |
|---|---|
| Equity | $9,338.19 |
| Total return | -6.62% |
| Buy & hold, same window | -0.64% |
| Difference | -5.98% |
| Round trips | 23 |
| Win rate | 34.8% |
| Max drawdown so far | -11.07% |
| Position | 0.115720 BTC |
| Halted | False |
| Online-learning cycles | 2 (1 promoted) |
<!-- LIVE:END -->

This is a paper account driven by real exchange bars, with the model trained
only on bars that precede the replayed window. Read it next to the buy & hold
row: making money in a rising market is not the same as being worth running.

### What these numbers say

**Every strategy tested loses money over this window, and the RL agent loses
most.** Full evidence in [docs/RESEARCH_FINDINGS.md](docs/RESEARCH_FINDINGS.md).

| Strategy | Return | Sharpe | Max DD | Trades |
|---|---|---|---|---|
| RSI rule | **-6.40%** | -0.13 | -22.99% | 298 |
| Mean-reversion rule | -8.15% | -0.24 | -29.02% | 320 |
| Buy & hold | -24.73% | -0.25 | -51.60% | 1 |
| RL agent | -43.03% | -2.19 | -50.04% | 1,316 |

**The RL agent is beaten by the simple rule it was meant to learn, and by doing
nothing.** That is the honest state of this project.

**Progress is real but insufficient.** Across three rounds of fixes the agent
went -88.37% → -65.60% → **-43.03%**, and its win rate 29.0% → 42.7%. Wider
ATR stops, the Moody-Saffell reward, n-step returns, volatility targeting and
maker execution each moved the number. None of them produced an edge.

**The idealized edge did not survive implementation.** The offline decile test
showed +30.9% annualized. Implemented with causal expanding-window quantiles,
real stops, a minimum hold and fills that can miss, the same rule returned
-8.15%. The gap was lookahead in the quantiles, unconstrained exits, and
guaranteed fills.

**Results by market regime** explain what these strategies actually are:

| Window | Regime | Buy & hold | Mean reversion | RSI |
|---|---|---|---|---|
| 2024-09 → 2025-01 | up | **+74.36%** | +21.29% | +32.75% |
| 2025-01 → 2025-07 | flat | +2.42% | **+4.13%** | -3.55% |
| 2025-07 → 2026-01 | down | -17.52% | -12.17% | **-7.03%** |
| 2026-01 → 2026-09 | flat | 0.00% | -3.47% | **+5.34%** |

A long/flat strategy makes money only when the market rises, and then
underperforms simply holding. It cushions declines but cannot profit from them.
Mean reversion adds value in exactly one regime — flat markets, +4.13% against
+2.42% — which is where the measured edge lives.

**Deflated Sharpe: 0.00**, against a luck threshold of 1.30 over 12 searched
configurations. Nothing here is distinguishable from noise.

Read the per-fold spread, not the headline: one number from one seed on one asset
over one window is a single draw from a noisy distribution. The cost model here
(fee + fixed slippage, fills at the bar close, no market impact or partial fills)
is a simplification, and a live venue would be worse. Nothing here is a forecast.

---

## Live paper trading

`live_trade.py` runs the agent against **real, newly-closed exchange bars** in a
simulated account. It places no orders and needs no API keys.

```
python live_trade.py --once --catch-up 400   # replay the last 400 real bars, then exit
python live_trade.py --poll 300              # poll forever, trade each new closed bar
python live_trade.py --status                # print the saved account
python live_trade.py --reset --once          # start a fresh paper account
```

**Data.** 15-minute OHLC bars from Binance, with Kraken and Coinbase as automatic
fallbacks, plus the real Fear & Greed Index from alternative.me. The cache is
backfilled from one venue so the series is continuous — splicing a stale CSV onto
recent bars would leave a hole in the middle, and every rolling indicator
spanning that hole would be wrong.

**Only closed bars are traded.** The newest kline an exchange returns is still
forming; its high, low and close will change. Acting on it is the live
equivalent of reading bar `t+1` in a backtest, so the feed discards it.

**Same execution rules as the backtest.** Fee, adverse slippage, capped
exposure, minimum order size, intrabar stop-loss / take-profit, and the
max-drawdown kill switch all come from the same `config.yaml` block. The
observation is built by `features/observation.py`, which the backtest
environment also calls — so the live policy cannot be fed a different input than
it trained on.

**State survives restarts.** The account, equity curve, trades, decisions and
model checkpoint live in `results/live/`, so stopping and restarting resumes the
same account instead of starting a fresh, flattering one.

### Self-improvement: continuous learning behind a gate

The bot learns on **every closed bar** — 96 updates a day at 15m — but a learned
update never reaches the trading model without proving itself first.

**1. Shadow learner (continuous).** Every bar's transition is fed to a *shadow*
copy of the live model, which takes a gradient step on it immediately. The
trading model is untouched. Once a day (`gate_every_bars: 96`) the shadow is
scored against the live model on recent bars and replaces it only on a clear
win. A rejected shadow is re-forked from the champion, so a losing direction
cannot compound across cycles.

**2. From-scratch challenger (weekly).** Independently, a fresh model is trained
on a recent window that excludes the newest bars, then scored on those excluded
bars. This catches drift that a shadow nudged incrementally for weeks may have
baked in.

Why the gate exists: an ungated per-bar update is fit to a single noisy
observation. Applied directly to the trader, thousands of them in sequence walk
the policy somewhere nobody chose — and nothing ever notices. Here they
accumulate in a candidate that has to beat the incumbent to graduate.

The slower cycle runs every `retrain_every_bars` (672 bars ≈ one week of 15m data):

1. The live model — the **champion** — is copied into a **challenger**.
2. The challenger is fine-tuned on a recent window that **excludes** the most
   recent bars.
3. Both models are scored greedily on those excluded bars, which neither has
   trained on.
4. The challenger replaces the champion **only** if it beats it by
   `min_improvement` on that held-out slice. Otherwise it is discarded.

Every decision — promoted or rejected, from either learner — is appended to
`results/live/promotions.json` tagged by `source` (`shadow` or `retrain`), so the
model's history says which kind of update won and when, rather than being a black
box that claims to have "got better". The first run with no checkpoint
bootstraps a model on cached history, holding out every bar it is about to
replay.

```yaml
online_learning:
  enabled: true
  bootstrap: true

  per_bar_updates: true      # one gradient step per closed bar, on a shadow model
  gate_every_bars: 96        # score the shadow once a day
  shadow_warmup_bars: 500    # no promotion before it has seen enough

  retrain_every_bars: 672    # the weekly from-scratch challenger
  train_window_bars: 5000
  eval_window_bars: 1000     # held out from that challenger's training
  episodes: 8

  metric: "sharpe_ratio"
  min_improvement: 0.05
```

**What this does and does not promise.** The gate prevents the live model from
being replaced by something measurably worse on held-out data. It does not
guarantee the bot becomes profitable, and it cannot: a 1,000-bar validation slice
is a noisy measurement, and a model that clears it can still lose money on the
next 1,000 bars. "Learns constantly" and "improves constantly" are different
claims; only the first is implemented. Given the backtest results below, expect the live account to
struggle too. The dashboard shows the paper account against buy & hold over the
same window so the comparison is always in view.

---

## Architecture

```
Data            data/BTC_15m.csv  ->  utils/data_utils.py
                  sorted oldest-first, validated, DatetimeIndex

Features        features/technical_indicators.py
                  causal indicators; volume columns skipped when absent

Environment     environment/trading_env.py
                  observe bar t -> act -> fill at t's close (+ slippage)
                  -> advance -> stops checked against t+1's range

Agent           agents/torch_dqn.py
                  Q-network (+ dueling head), target net, PER buffer
                  epsilon decays once per EPISODE

Evaluation      backtest.py  ->  utils/metrics.py
                  walk-forward folds, baselines, results/ artifacts

Live            live/feed.py     closed bars + real Fear & Greed
                live/paper_broker.py   simulated fills, same cost model
                live/trader.py   per-bar loop, persisted account
                live/learner.py  champion/challenger promotion gate

Dashboard       dashboard/   (Vite + React + TypeScript)
                  reads results/, renders nothing it cannot source
```

## Project structure

```
├── strategies/
│   ├── basis_carry.py           # delta-neutral funding carry (validated)
│   ├── carry_book.py            # multi-sleeve cross-sectional carry
│   ├── risk_budget.py           # leverage from ruin probability, not Sharpe
│   ├── account_sizing.py        # lots for a real account; the $20 problem
│   └── regime.py                # volatility regimes (tested, ships disabled)
├── environment/
│   ├── trading_env.py           # execution, costs, risk controls, reward
│   ├── execution.py             # taker / maker / CFD fill models
│   ├── risk.py                  # ATR stops and minimum hold, shared live+backtest
│   └── sizing.py                # volatility targeting, fractional Kelly
├── agents/
│   ├── torch_dqn.py             # DQN / Double / Dueling + prioritized replay
│   └── rewards.py               # differential Sharpe, cost-aware, log return
├── features/
│   ├── technical_indicators.py  # causal indicators
│   └── observation.py           # shared by backtest and live, so they cannot drift
├── live/
│   ├── feed.py                  # exchange bars, funding, closed bars only
│   ├── gold_feed.py             # COMEX gold, session gaps preserved
│   ├── calendar.py              # economic events, blackout windows
│   ├── paper_broker.py          # simulated account, backtest's cost model
│   ├── trader.py                # live loop, persisted state
│   ├── gold_trader.py           # gold variant with event blackouts
│   ├── basis_trader.py          # funding carry live
│   └── learner.py               # per-bar shadow learning behind a gate
├── tools/
│   ├── fetch_*.py               # data acquisition
│   ├── signal_analysis.py       # is there an edge, and does it beat the cost?
│   ├── sweep.py                 # validation-only parameter search
│   └── render_readme_table.py   # generates the results tables in this file
├── tests/                       # 255 tests
├── docs/                        # the three research documents
├── dashboard/                   # Vite + React + TypeScript
├── backtest.py                  # walk-forward for price strategies
├── basis_backtest.py            # walk-forward for funding carry
├── carry_backtest.py            # walk-forward for the multi-sleeve book
├── live_trade.py / live_gold.py / live_basis.py
└── config.yaml
```

## Configuration

`config.yaml` is read in full — every key below is used by code.

```yaml
trading:
  initial_balance: 10000
  transaction_cost: 0.001     # per fill, both legs
  max_position_size: 0.95
  min_position_size: 0.02     # rejects dust orders

model:
  variant: "double"           # dqn | double | dueling
  learning_rate: 0.0005
  gamma: 0.99
  epsilon_decay: 0.92         # applied once per EPISODE
  prioritized: true

risk:
  max_drawdown_threshold: 0.35
  stop_loss: 0.05
  take_profit: 0.15

backtesting:
  slippage: 0.0005
  walk_forward:
    n_folds: 5
    train_episodes: 40
    purge_bars: 60
```

CLI flags override config values **in memory** — `config.yaml` is never rewritten
by a training run.

## Dataset

`data/BTC_15m.csv` — **70,111 fifteen-minute bars, 2024-09-13 to 2026-09-13, zero
gaps**, fetched from Binance by `tools/fetch_history.py` and joined to the real
Fear & Greed Index. `data/BTC.csv` (11,618 four-hour bars, 2020-2025) ships with
the repo as a reference series and is what the tests run against.

Neither file has a **volume column**, so volume-derived indicators are skipped
rather than faked, and the live feed drops volume to keep the live feature vector
identical to the trained one.

## Testing

```bash
pytest -q          # 255 tests
```

The suite exists to pin the properties that make a backtest trustworthy. The
ones that caught real bugs:

| Test | Property |
|---|---|
| `test_indicators_are_causal` | Truncating the series cannot change earlier values |
| `test_observations_do_not_depend_on_future_bars` | Future bars change no past observation |
| `test_forming_bar_is_dropped` | The live feed never trades a still-forming bar |
| `test_session_gaps_are_not_filled` | Indicators never smear across a closed market |
| `test_equity_curve_reconciles_with_round_trip_pnl` | Realized P&L explains final equity |
| `test_round_trip_with_zero_price_move_loses_the_fees` | A flat round trip is a loss |
| `test_sharpe_uses_the_supplied_annualization` | Bars annualized by real spacing |
| `test_a_fold_with_no_trades_reports_zero_sharpe` | Caught a Sharpe of -5e17 |
| `test_maker_mode_both_fills_and_misses` | A fill model that never misses is wrong |
| `test_price_direction_does_not_matter` | The carry hedge is actually delta-neutral |
| `test_65x_is_near_certain_ruin_within_a_week` | The headline risk claim, as a test |
| `test_one_sleeve_liquidation_does_not_wipe_the_book` | A sleeve loses only its own capital |
| `test_shadow_never_mutates_the_champion` | Per-bar learning cannot reach the trader |
| `test_live_trader_matches_the_backtest` | Live and backtest agree on identical data |
| `test_action_names_cover_the_full_action_space` | Caught a live-loop `KeyError: 3` |

## Technology

- **Deep learning:** PyTorch (DQN, Double, Dueling, prioritized replay, n-step)
- **Data:** pandas, NumPy
- **Config:** PyYAML
- **Tests:** pytest
- **Dashboard:** Vite, React, TypeScript, Recharts
- **Data sources:** Binance (spot, perpetuals, funding), Yahoo Finance (COMEX
  gold), alternative.me (Fear & Greed), ForexFactory (economic calendar)

## Methods, with sources

| Method | Source |
|---|---|
| Differential Sharpe ratio reward | Moody & Saffell, *J. Forecasting* (1998); IEEE TNN (2001) |
| Cost-aware reward, volatility targeting | Zhang, Zohren & Roberts, *J. Financial Data Science* (2020) |
| n-step returns | Sutton & Barto ch. 7; Hessel et al., *Rainbow*, AAAI (2018) |
| Prioritized experience replay | Schaul et al., ICLR (2016) |
| Deflated Sharpe ratio | Bailey & López de Prado, *J. Portfolio Management* (2014) |
| Purged walk-forward validation | López de Prado, *Advances in Financial ML*, ch. 7 |
| Fractional Kelly sizing | Kelly, *Bell System Tech. J.* (1956); Thorp (2006) |
| Volatility clustering | Engle, ARCH (1982) |
| Time-series momentum sizing | Moskowitz, Ooi & Pedersen, *JFE* (2012) |
| Efficiency ratio | Kaufman, *Smarter Trading* (1995) |

## Not implemented

Called out so the feature list stays honest. These are **not** in the codebase:
market regime *prediction*, multi-agent ensembles, VAR-targeted position sizing,
LSTM/attention architectures, short selling, and live order routing. Regime
*detection* exists but ships disabled because it measured worse.

Live trading is **paper only** throughout — every account simulates fills and
never sends an order to an exchange.

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

For education and research only. This is a backtest on historical data for one
asset, and a backtest is not a track record. Do not trade money you cannot
afford to lose.
