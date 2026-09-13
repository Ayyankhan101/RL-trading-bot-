"""
Performance metrics for the RL Trading Bot.

Every annualized figure takes ``periods_per_year`` explicitly. The bundled
dataset is 4-hourly (2190 bars/year), so the daily-equities convention of 252
would overstate Sharpe, Sortino and volatility by roughly 2.9x.

Trade-level metrics (win rate, profit factor, expectancy) expect **round trips**
- completed entry/exit pairs with a net P&L - not individual orders. Counting
entry orders in the denominator structurally caps win rate near 50%.
"""

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

DAILY_PERIODS_PER_YEAR = 365.0


def _as_array(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def calculate_returns(portfolio_values: Sequence[float]) -> np.ndarray:
    """Simple period-over-period returns of an equity curve."""
    values = np.asarray(portfolio_values, dtype=np.float64)
    if len(values) < 2:
        return np.array([])

    prev = values[:-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        returns = np.where(prev > 0, (values[1:] - prev) / prev, 0.0)
    return returns[np.isfinite(returns)]


def calculate_total_return(initial_value: float, final_value: float) -> float:
    if initial_value <= 0:
        return 0.0
    return (final_value - initial_value) / initial_value


def calculate_annualized_return(total_return: float, n_periods: int,
                                periods_per_year: float = DAILY_PERIODS_PER_YEAR) -> float:
    """
    Compound (geometric) annualized return.

    The previous implementation scaled linearly - ``total_return * 252 / n`` -
    which is not an annualized return at all and diverges badly over multi-year
    windows.
    """
    if n_periods <= 0 or periods_per_year <= 0:
        return 0.0
    growth = 1.0 + total_return
    if growth <= 0:
        return -1.0
    years = n_periods / periods_per_year
    if years <= 0:
        return 0.0
    return float(growth ** (1.0 / years) - 1.0)


def calculate_sharpe_ratio(returns: Sequence[float], risk_free_rate: float = 0.02,
                           periods_per_year: float = DAILY_PERIODS_PER_YEAR) -> float:
    """Annualized Sharpe ratio of per-period returns."""
    returns = _as_array(returns)
    if len(returns) < 2:
        return 0.0

    period_rf = risk_free_rate / periods_per_year
    excess = returns - period_rf
    std = np.std(excess, ddof=1)

    # A flat equity curve has no risk to adjust for. Testing against exact zero
    # is not enough: a curve that never traded leaves float noise in the std,
    # and dividing the constant risk-free shortfall by it produced a Sharpe of
    # -5e17 on a fold where the agent took no trades at all.
    if std <= abs(np.mean(returns)) * 1e-9 or std < 1e-15:
        return 0.0

    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def calculate_sortino_ratio(returns: Sequence[float], risk_free_rate: float = 0.02,
                            periods_per_year: float = DAILY_PERIODS_PER_YEAR) -> float:
    """
    Annualized Sortino ratio.

    Downside deviation averages the squared shortfalls over **all** periods, not
    only the negative ones - dividing by the count of losses instead inflates the
    deviation and understates the ratio.
    """
    returns = _as_array(returns)
    if len(returns) < 2:
        return 0.0

    std = np.std(returns, ddof=1)
    if std <= abs(np.mean(returns)) * 1e-9 or std < 1e-15:
        # A flat equity curve has no risk to adjust for. Without this guard the
        # constant risk-free shortfall divides by a vanishing downside deviation
        # and reports a large negative ratio for a portfolio that never moved.
        return 0.0

    period_rf = risk_free_rate / periods_per_year
    excess = returns - period_rf
    downside = np.minimum(excess, 0.0)
    downside_deviation = np.sqrt(np.mean(downside ** 2))

    if downside_deviation == 0:
        return 0.0

    return float(np.mean(excess) / downside_deviation * np.sqrt(periods_per_year))


def calculate_max_drawdown(portfolio_values: Sequence[float]) -> Dict[str, Any]:
    """Maximum peak-to-trough decline, with the indices that produced it."""
    values = np.asarray(portfolio_values, dtype=np.float64)
    if len(values) < 2:
        return {'max_drawdown': 0.0, 'peak_index': 0, 'trough_index': 0, 'recovery_index': None}

    peak = np.maximum.accumulate(values)
    with np.errstate(divide='ignore', invalid='ignore'):
        drawdown = np.where(peak > 0, (values - peak) / peak, 0.0)

    trough_idx = int(np.argmin(drawdown))
    # The peak that *caused* this trough is where the running max was set, which
    # is the last index at that level, not the first.
    peak_idx = int(np.argmax(values[:trough_idx + 1])) if trough_idx > 0 else 0

    recovery_idx = None
    after = np.nonzero(values[trough_idx:] >= values[peak_idx])[0]
    if len(after):
        recovery_idx = int(trough_idx + after[0])

    return {
        'max_drawdown': float(np.min(drawdown)),
        'peak_index': peak_idx,
        'trough_index': trough_idx,
        'recovery_index': recovery_idx,
    }


def calculate_calmar_ratio(annualized_return: float, max_drawdown: float) -> float:
    """Annualized return over max drawdown. Undefined without a drawdown."""
    if abs(max_drawdown) < 1e-12:
        return 0.0
    return float(annualized_return / abs(max_drawdown))


def calculate_var(returns: Sequence[float], confidence_level: float = 0.05) -> float:
    """
    Historical Value at Risk, reported as a **positive loss magnitude**.

    A 0.05 confidence level means: on the worst 5% of periods the loss was at
    least this large.
    """
    returns = _as_array(returns)
    if len(returns) == 0:
        return 0.0
    return float(max(0.0, -np.percentile(returns, confidence_level * 100)))


def calculate_cvar(returns: Sequence[float], confidence_level: float = 0.05) -> float:
    """Expected shortfall beyond the VaR threshold, as a positive magnitude."""
    returns = _as_array(returns)
    if len(returns) == 0:
        return 0.0

    threshold = np.percentile(returns, confidence_level * 100)
    tail = returns[returns <= threshold]
    if len(tail) == 0:
        return 0.0
    return float(max(0.0, -np.mean(tail)))


def calculate_volatility(returns: Sequence[float], annualize: bool = True,
                         periods_per_year: float = DAILY_PERIODS_PER_YEAR) -> float:
    returns = _as_array(returns)
    if len(returns) < 2:
        return 0.0

    vol = float(np.std(returns, ddof=1))
    return vol * float(np.sqrt(periods_per_year)) if annualize else vol


def calculate_win_rate(trades: List[Dict[str, Any]]) -> float:
    """Share of **round trips** that closed with a positive net P&L."""
    closed = [t for t in trades if t.get('pnl') is not None]
    if not closed:
        return 0.0
    return sum(1 for t in closed if t['pnl'] > 0) / len(closed)


def calculate_profit_factor(trades: List[Dict[str, Any]]) -> Optional[float]:
    """
    Gross profit over gross loss.

    Returns ``None`` when there are no losing trades: the ratio is genuinely
    undefined there, and returning ``inf`` only propagates into report
    formatting as a crash or a nonsense figure.
    """
    closed = [t for t in trades if t.get('pnl') is not None]
    if not closed:
        return None

    gross_profit = sum(t['pnl'] for t in closed if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in closed if t['pnl'] < 0))

    if gross_loss == 0:
        return None
    return float(gross_profit / gross_loss)


def calculate_expectancy(trades: List[Dict[str, Any]]) -> float:
    """Average net P&L per round trip, in account currency."""
    closed = [t for t in trades if t.get('pnl') is not None]
    if not closed:
        return 0.0
    return float(np.mean([t['pnl'] for t in closed]))


def calculate_beta(returns: Sequence[float], benchmark_returns: Sequence[float]) -> float:
    """Beta against a benchmark return series (consistent ddof throughout)."""
    returns = np.asarray(returns, dtype=np.float64)
    benchmark_returns = np.asarray(benchmark_returns, dtype=np.float64)

    n = min(len(returns), len(benchmark_returns))
    if n < 2:
        return 0.0

    returns, benchmark_returns = returns[:n], benchmark_returns[:n]
    variance = np.var(benchmark_returns, ddof=1)
    if variance == 0:
        return 0.0

    covariance = np.cov(returns, benchmark_returns, ddof=1)[0, 1]
    return float(covariance / variance)


def calculate_alpha(returns: Sequence[float], benchmark_returns: Sequence[float],
                    risk_free_rate: float = 0.02,
                    periods_per_year: float = DAILY_PERIODS_PER_YEAR) -> float:
    """Annualized Jensen's alpha."""
    returns = np.asarray(returns, dtype=np.float64)
    benchmark_returns = np.asarray(benchmark_returns, dtype=np.float64)

    n = min(len(returns), len(benchmark_returns))
    if n < 2:
        return 0.0

    returns, benchmark_returns = returns[:n], benchmark_returns[:n]
    beta = calculate_beta(returns, benchmark_returns)
    period_rf = risk_free_rate / periods_per_year

    excess = np.mean(returns) - period_rf
    benchmark_excess = np.mean(benchmark_returns) - period_rf
    return float((excess - beta * benchmark_excess) * periods_per_year)


def calculate_comprehensive_metrics(
    portfolio_values: Sequence[float],
    trades: Optional[List[Dict[str, Any]]] = None,
    benchmark_values: Optional[Sequence[float]] = None,
    periods_per_year: float = DAILY_PERIODS_PER_YEAR,
    risk_free_rate: float = 0.02,
    n_trials: int = 1,
) -> Dict[str, Any]:
    """Full metric set for one equity curve. All keys are plain floats/ints."""
    values = np.asarray(portfolio_values, dtype=np.float64)
    if len(values) < 2:
        return {}

    returns = calculate_returns(values)
    total_return = calculate_total_return(values[0], values[-1])
    n_periods = len(values) - 1
    annualized_return = calculate_annualized_return(total_return, n_periods, periods_per_year)
    drawdown = calculate_max_drawdown(values)

    metrics: Dict[str, Any] = {
        'initial_value': float(values[0]),
        'final_value': float(values[-1]),
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'volatility': calculate_volatility(returns, True, periods_per_year),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        'max_drawdown': drawdown['max_drawdown'],
        'max_drawdown_peak_index': drawdown['peak_index'],
        'max_drawdown_trough_index': drawdown['trough_index'],
        'calmar_ratio': calculate_calmar_ratio(annualized_return, drawdown['max_drawdown']),
        'var_95': calculate_var(returns, 0.05),
        'cvar_95': calculate_cvar(returns, 0.05),
        'n_periods': int(n_periods),
        'periods_per_year': float(periods_per_year),
    }

    # How much of the Sharpe survives the number of configurations tried.
    metrics.update(deflated_sharpe_ratio(returns, n_trials=n_trials,
                                         periods_per_year=periods_per_year))

    if trades:
        closed = [t for t in trades if t.get('pnl') is not None]
        wins = [t['pnl'] for t in closed if t['pnl'] > 0]
        losses = [t['pnl'] for t in closed if t['pnl'] < 0]

        metrics.update({
            'total_trades': len(closed),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': calculate_win_rate(closed),
            'profit_factor': calculate_profit_factor(closed),
            'expectancy': calculate_expectancy(closed),
            'avg_winning_trade': float(np.mean(wins)) if wins else 0.0,
            'avg_losing_trade': float(np.mean(losses)) if losses else 0.0,
            'largest_win': float(max(wins)) if wins else 0.0,
            'largest_loss': float(min(losses)) if losses else 0.0,
        })
    else:
        metrics.update({'total_trades': 0, 'win_rate': 0.0, 'profit_factor': None})

    if benchmark_values is not None and len(benchmark_values) >= 2:
        benchmark_returns = calculate_returns(benchmark_values)
        metrics['beta'] = calculate_beta(returns, benchmark_returns)
        metrics['alpha'] = calculate_alpha(returns, benchmark_returns,
                                           risk_free_rate, periods_per_year)

    return metrics


def _fmt(value: Any, kind: str = 'float') -> str:
    if value is None:
        return 'n/a'
    if not isinstance(value, (int, float)) or not np.isfinite(value):
        return 'n/a'
    if kind == 'pct':
        return f"{value * 100:.2f}%"
    if kind == 'money':
        return f"${value:,.2f}"
    if kind == 'int':
        return f"{int(value):,}"
    return f"{value:.2f}"


def print_performance_report(metrics: Dict[str, Any], title: str = "PERFORMANCE") -> None:
    """Print a metrics dict. Missing or undefined values render as 'n/a'."""
    if not metrics:
        print("No metrics to report.")
        return

    rows = [
        ("Initial Value", _fmt(metrics.get('initial_value'), 'money')),
        ("Final Value", _fmt(metrics.get('final_value'), 'money')),
        ("Total Return", _fmt(metrics.get('total_return'), 'pct')),
        ("Annualized Return", _fmt(metrics.get('annualized_return'), 'pct')),
        ("Volatility (ann.)", _fmt(metrics.get('volatility'), 'pct')),
        ("Sharpe Ratio", _fmt(metrics.get('sharpe_ratio'))),
        ("Sortino Ratio", _fmt(metrics.get('sortino_ratio'))),
        ("Max Drawdown", _fmt(metrics.get('max_drawdown'), 'pct')),
        ("Calmar Ratio", _fmt(metrics.get('calmar_ratio'))),
        ("VaR 95% (per bar)", _fmt(metrics.get('var_95'), 'pct')),
        ("CVaR 95% (per bar)", _fmt(metrics.get('cvar_95'), 'pct')),
        ("Round Trips", _fmt(metrics.get('total_trades'), 'int')),
        ("Win Rate", _fmt(metrics.get('win_rate'), 'pct')),
        ("Profit Factor", _fmt(metrics.get('profit_factor'))),
        ("Expectancy / Trade", _fmt(metrics.get('expectancy'), 'money')),
        ("Deflated Sharpe", _fmt(metrics.get('deflated_sharpe'))),
        ("Luck threshold SR", _fmt(metrics.get('threshold_sharpe'))),
        ("Configs searched", _fmt(metrics.get('n_trials'), 'int')),
    ]

    print(f"\n{title}")
    print("-" * 44)
    for label, value in rows:
        print(f"{label:<22}{value:>21}")
    print("-" * 44)


# --------------------------------------------------------------- multiple testing

EULER_MASCHERONI = 0.5772156649015329


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    plow, phigh = 0.02425, 1 - 0.02425

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def expected_max_sharpe(n_trials: int, sharpe_variance: float = 1.0) -> float:
    """
    Expected maximum Sharpe ratio from ``n_trials`` independent backtests of a
    strategy with no real edge.

    Bailey & Lopez de Prado, "The Deflated Sharpe Ratio: Correcting for Selection
    Bias, Backtest Overfitting and Non-Normality" (Journal of Portfolio
    Management, 2014), equation 4:

        SR_0 = sqrt(V[SR]) * [ (1 - g) * Z^-1(1 - 1/N) + g * Z^-1(1 - 1/(N*e)) ]

    with g the Euler-Mascheroni constant. Try enough configurations and a
    positive Sharpe appears by luck alone; this is the bar luck clears.
    """
    if n_trials < 2:
        return 0.0

    gamma = EULER_MASCHERONI
    term = ((1 - gamma) * _normal_ppf(1 - 1.0 / n_trials)
            + gamma * _normal_ppf(1 - 1.0 / (n_trials * math.e)))
    return float(math.sqrt(max(sharpe_variance, 0.0)) * term)


def deflated_sharpe_ratio(returns: Sequence[float], n_trials: int = 1,
                          periods_per_year: float = DAILY_PERIODS_PER_YEAR,
                          benchmark_sharpe: Optional[float] = None) -> Dict[str, Any]:
    """
    Probability that the observed Sharpe ratio reflects skill rather than
    selection bias and non-normal returns.

    Bailey & Lopez de Prado (2014), equation 9:

        DSR = Z[ (SR - SR_0) * sqrt(T - 1) /
                 sqrt(1 - g3*SR + (g4 - 1)/4 * SR^2) ]

    where SR is the **non-annualized** per-period Sharpe, g3 the skewness and g4
    the kurtosis of the return series, and T the number of observations.

    A DSR below ~0.95 means the result is not distinguishable from what searching
    over that many configurations would produce by chance.
    """
    returns = _as_array(returns)
    if len(returns) < 10:
        return {'deflated_sharpe': 0.0, 'observed_sharpe': 0.0,
                'threshold_sharpe': 0.0, 'n_trials': int(n_trials)}

    std = np.std(returns, ddof=1)
    if std == 0:
        return {'deflated_sharpe': 0.0, 'observed_sharpe': 0.0,
                'threshold_sharpe': 0.0, 'n_trials': int(n_trials)}

    sharpe = float(np.mean(returns) / std)          # per period, not annualized
    n = len(returns)

    centred = returns - np.mean(returns)
    skew = float(np.mean(centred ** 3) / (std ** 3))
    kurtosis = float(np.mean(centred ** 4) / (std ** 4))

    threshold = expected_max_sharpe(n_trials, sharpe_variance=1.0 / max(n - 1, 1))

    denominator = 1.0 - skew * sharpe + ((kurtosis - 1.0) / 4.0) * sharpe ** 2
    if denominator <= 0:
        deflated = 0.0
    else:
        statistic = (sharpe - threshold) * math.sqrt(max(n - 1, 1)) / math.sqrt(denominator)
        deflated = _normal_cdf(statistic)

    return {
        'deflated_sharpe': float(deflated),
        'observed_sharpe': sharpe * float(np.sqrt(periods_per_year)),
        'threshold_sharpe': threshold * float(np.sqrt(periods_per_year)),
        'skew': skew,
        'kurtosis': kurtosis,
        'n_trials': int(n_trials),
        'n_observations': int(n),
    }
