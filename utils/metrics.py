"""
Performance metrics calculation for the RL Trading Bot
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


def calculate_returns(portfolio_values: List[float]) -> np.ndarray:
    """
    Calculate daily returns from portfolio values
    
    Args:
        portfolio_values: List of portfolio values over time
        
    Returns:
        Array of daily returns
    """
    if len(portfolio_values) < 2:
        return np.array([])
    
    values = np.array(portfolio_values)
    returns = np.diff(values) / values[:-1]
    return returns


def calculate_total_return(initial_value: float, final_value: float) -> float:
    """
    Calculate total return percentage
    
    Args:
        initial_value: Starting portfolio value
        final_value: Ending portfolio value
        
    Returns:
        Total return as percentage
    """
    if initial_value <= 0:
        return 0.0
    
    return (final_value - initial_value) / initial_value


def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio
    
    Args:
        returns: Array of daily returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # Convert annual risk-free rate to daily
    daily_rf_rate = risk_free_rate / 252
    
    excess_returns = returns - daily_rf_rate
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    # Annualize the Sharpe ratio
    sharpe = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)
    return sharpe


def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sortino ratio (focuses on downside risk)
    
    Args:
        returns: Array of daily returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0
    
    daily_rf_rate = risk_free_rate / 252
    excess_returns = returns - daily_rf_rate
    
    # Calculate downside deviation
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0
    
    downside_deviation = np.sqrt(np.mean(negative_returns ** 2))
    
    if downside_deviation == 0:
        return 0.0
    
    sortino = np.mean(excess_returns) / downside_deviation * np.sqrt(252)
    return sortino


def calculate_max_drawdown(portfolio_values: List[float]) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown and its duration
    
    Args:
        portfolio_values: List of portfolio values over time
        
    Returns:
        Tuple of (max_drawdown, start_index, end_index)
    """
    if len(portfolio_values) < 2:
        return 0.0, 0, 0
    
    values = np.array(portfolio_values)
    peak = np.maximum.accumulate(values)
    drawdown = (values - peak) / peak
    
    max_dd = np.min(drawdown)
    max_dd_idx = np.argmin(drawdown)
    
    # Find the peak before the max drawdown
    peak_idx = np.argmax(peak[:max_dd_idx + 1])
    
    return max_dd, peak_idx, max_dd_idx


def calculate_calmar_ratio(total_return: float, max_drawdown: float) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown)
    
    Args:
        total_return: Total return percentage
        max_drawdown: Maximum drawdown percentage
        
    Returns:
        Calmar ratio
    """
    if abs(max_drawdown) < 1e-8:
        return float('inf') if total_return > 0 else 0.0
    
    return total_return / abs(max_drawdown)


def calculate_var(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """
    Calculate Value at Risk (VaR)
    
    Args:
        returns: Array of daily returns
        confidence_level: Confidence level (default: 5% for 95% VaR)
        
    Returns:
        VaR value
    """
    if len(returns) == 0:
        return 0.0
    
    return np.percentile(returns, confidence_level * 100)


def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.05) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) - Expected Shortfall
    
    Args:
        returns: Array of daily returns
        confidence_level: Confidence level
        
    Returns:
        CVaR value
    """
    if len(returns) == 0:
        return 0.0
    
    var = calculate_var(returns, confidence_level)
    cvar = np.mean(returns[returns <= var])
    return cvar


def calculate_win_rate(trades: List[Dict]) -> float:
    """
    Calculate win rate from trade history
    
    Args:
        trades: List of trade dictionaries with 'pnl' key
        
    Returns:
        Win rate as percentage
    """
    if not trades:
        return 0.0
    
    winning_trades = sum(1 for trade in trades if trade.get('pnl', 0) > 0)
    return winning_trades / len(trades)


def calculate_profit_factor(trades: List[Dict]) -> float:
    """
    Calculate profit factor (gross profit / gross loss)
    
    Args:
        trades: List of trade dictionaries with 'pnl' key
        
    Returns:
        Profit factor
    """
    if not trades:
        return 0.0
    
    gross_profit = sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) > 0)
    gross_loss = abs(sum(trade.get('pnl', 0) for trade in trades if trade.get('pnl', 0) < 0))
    
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    
    return gross_profit / gross_loss


def calculate_volatility(returns: np.ndarray, annualize: bool = True) -> float:
    """
    Calculate volatility (standard deviation of returns)
    
    Args:
        returns: Array of daily returns
        annualize: Whether to annualize the volatility
        
    Returns:
        Volatility
    """
    if len(returns) == 0:
        return 0.0
    
    vol = np.std(returns)
    
    if annualize:
        vol *= np.sqrt(252)  # Annualize assuming 252 trading days
    
    return vol


def calculate_beta(strategy_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Calculate beta (correlation with benchmark)
    
    Args:
        strategy_returns: Strategy returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Beta coefficient
    """
    if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    
    # Ensure same length
    min_len = min(len(strategy_returns), len(benchmark_returns))
    strategy_returns = strategy_returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]
    
    if np.var(benchmark_returns) == 0:
        return 0.0
    
    covariance = np.cov(strategy_returns, benchmark_returns)[0, 1]
    benchmark_variance = np.var(benchmark_returns)
    
    return covariance / benchmark_variance


def calculate_alpha(strategy_returns: np.ndarray, 
                   benchmark_returns: np.ndarray,
                   risk_free_rate: float = 0.02) -> float:
    """
    Calculate alpha (excess return over CAPM prediction)
    
    Args:
        strategy_returns: Strategy returns
        benchmark_returns: Benchmark returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Alpha
    """
    if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    
    daily_rf_rate = risk_free_rate / 252
    
    # Calculate beta
    beta = calculate_beta(strategy_returns, benchmark_returns)
    
    # Calculate average returns
    avg_strategy_return = np.mean(strategy_returns)
    avg_benchmark_return = np.mean(benchmark_returns)
    
    # Calculate alpha
    alpha = avg_strategy_return - (daily_rf_rate + beta * (avg_benchmark_return - daily_rf_rate))
    
    # Annualize alpha
    return alpha * 252


def calculate_information_ratio(strategy_returns: np.ndarray, 
                              benchmark_returns: np.ndarray) -> float:
    """
    Calculate information ratio (active return / tracking error)
    
    Args:
        strategy_returns: Strategy returns
        benchmark_returns: Benchmark returns
        
    Returns:
        Information ratio
    """
    if len(strategy_returns) == 0 or len(benchmark_returns) == 0:
        return 0.0
    
    # Ensure same length
    min_len = min(len(strategy_returns), len(benchmark_returns))
    strategy_returns = strategy_returns[:min_len]
    benchmark_returns = benchmark_returns[:min_len]
    
    # Calculate active returns
    active_returns = strategy_returns - benchmark_returns
    
    if np.std(active_returns) == 0:
        return 0.0
    
    # Calculate information ratio
    ir = np.mean(active_returns) / np.std(active_returns) * np.sqrt(252)
    return ir


def calculate_comprehensive_metrics(portfolio_values: List[float],
                                  trades: List[Dict] = None,
                                  benchmark_values: List[float] = None,
                                  risk_free_rate: float = 0.02) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics
    
    Args:
        portfolio_values: List of portfolio values over time
        trades: List of trade dictionaries
        benchmark_values: Benchmark portfolio values
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Dictionary of performance metrics
    """
    if len(portfolio_values) < 2:
        return {}
    
    # Calculate returns
    returns = calculate_returns(portfolio_values)
    
    # Basic metrics
    metrics = {
        'total_return': calculate_total_return(portfolio_values[0], portfolio_values[-1]),
        'annualized_return': calculate_total_return(portfolio_values[0], portfolio_values[-1]) * (252 / len(portfolio_values)),
        'volatility': calculate_volatility(returns),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate),
        'var_95': calculate_var(returns, 0.05),
        'cvar_95': calculate_cvar(returns, 0.05),
    }
    
    # Drawdown metrics
    max_dd, dd_start, dd_end = calculate_max_drawdown(portfolio_values)
    metrics['max_drawdown'] = max_dd
    metrics['calmar_ratio'] = calculate_calmar_ratio(metrics['total_return'], max_dd)
    
    # Trade-based metrics
    if trades:
        metrics['win_rate'] = calculate_win_rate(trades)
        metrics['profit_factor'] = calculate_profit_factor(trades)
        metrics['total_trades'] = len(trades)
        
        # Average trade metrics
        trade_pnls = [trade.get('pnl', 0) for trade in trades]
        if trade_pnls:
            metrics['avg_trade_pnl'] = np.mean(trade_pnls)
            metrics['best_trade'] = max(trade_pnls)
            metrics['worst_trade'] = min(trade_pnls)
    
    # Benchmark comparison
    if benchmark_values and len(benchmark_values) >= len(portfolio_values):
        benchmark_returns = calculate_returns(benchmark_values[:len(portfolio_values)])
        
        metrics['beta'] = calculate_beta(returns, benchmark_returns)
        metrics['alpha'] = calculate_alpha(returns, benchmark_returns, risk_free_rate)
        metrics['information_ratio'] = calculate_information_ratio(returns, benchmark_returns)
        
        # Benchmark metrics for comparison
        metrics['benchmark_total_return'] = calculate_total_return(benchmark_values[0], benchmark_values[len(portfolio_values)-1])
        metrics['benchmark_volatility'] = calculate_volatility(benchmark_returns)
        metrics['benchmark_sharpe'] = calculate_sharpe_ratio(benchmark_returns, risk_free_rate)
        benchmark_max_dd, _, _ = calculate_max_drawdown(benchmark_values[:len(portfolio_values)])
        metrics['benchmark_max_drawdown'] = benchmark_max_dd
    
    return metrics


def print_performance_report(metrics: Dict[str, float]) -> None:
    """
    Print a formatted performance report
    
    Args:
        metrics: Dictionary of performance metrics
    """
    print("=" * 60)
    print("PERFORMANCE REPORT")
    print("=" * 60)
    
    # Return metrics
    print("\n📈 RETURN METRICS")
    print("-" * 30)
    if 'total_return' in metrics:
        print(f"Total Return:        {metrics['total_return']:>10.2%}")
    if 'annualized_return' in metrics:
        print(f"Annualized Return:   {metrics['annualized_return']:>10.2%}")
    
    # Risk metrics
    print("\n⚠️  RISK METRICS")
    print("-" * 30)
    if 'volatility' in metrics:
        print(f"Volatility:          {metrics['volatility']:>10.2%}")
    if 'max_drawdown' in metrics:
        print(f"Max Drawdown:        {metrics['max_drawdown']:>10.2%}")
    if 'var_95' in metrics:
        print(f"VaR (95%):           {metrics['var_95']:>10.2%}")
    if 'cvar_95' in metrics:
        print(f"CVaR (95%):          {metrics['cvar_95']:>10.2%}")
    
    # Risk-adjusted metrics
    print("\n📊 RISK-ADJUSTED METRICS")
    print("-" * 30)
    if 'sharpe_ratio' in metrics:
        print(f"Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
    if 'sortino_ratio' in metrics:
        print(f"Sortino Ratio:       {metrics['sortino_ratio']:>10.2f}")
    if 'calmar_ratio' in metrics:
        print(f"Calmar Ratio:        {metrics['calmar_ratio']:>10.2f}")
    
    # Trading metrics
    if 'total_trades' in metrics:
        print("\n💼 TRADING METRICS")
        print("-" * 30)
        print(f"Total Trades:        {metrics['total_trades']:>10.0f}")
        if 'win_rate' in metrics:
            print(f"Win Rate:            {metrics['win_rate']:>10.2%}")
        if 'profit_factor' in metrics:
            print(f"Profit Factor:       {metrics['profit_factor']:>10.2f}")
        if 'avg_trade_pnl' in metrics:
            print(f"Avg Trade P&L:       ${metrics['avg_trade_pnl']:>9.2f}")
    
    # Benchmark comparison
    if 'beta' in metrics:
        print("\n🏆 BENCHMARK COMPARISON")
        print("-" * 30)
        print(f"Beta:                {metrics['beta']:>10.2f}")
        if 'alpha' in metrics:
            print(f"Alpha:               {metrics['alpha']:>10.2%}")
        if 'information_ratio' in metrics:
            print(f"Information Ratio:   {metrics['information_ratio']:>10.2f}")
    
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    print("Testing performance metrics...")
    
    # Generate sample portfolio values
    np.random.seed(42)
    initial_value = 10000
    returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year
    portfolio_values = [initial_value]
    
    for ret in returns:
        portfolio_values.append(portfolio_values[-1] * (1 + ret))
    
    # Calculate metrics
    metrics = calculate_comprehensive_metrics(portfolio_values)
    
    # Print report
    print_performance_report(metrics)
