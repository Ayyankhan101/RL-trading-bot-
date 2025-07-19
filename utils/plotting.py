"""
Visualization functions for the RL Trading Bot
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Optional


# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_price_and_indicators(data: pd.DataFrame, 
                             title: str = "Bitcoin Price with Technical Indicators",
                             save_path: Optional[str] = None) -> None:
    """
    Plot Bitcoin price with technical indicators
    
    Args:
        data: DataFrame with OHLCV and indicator data
        title: Plot title
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(4, 1, figsize=(15, 12))
    
    # Price and moving averages
    axes[0].plot(data.index, data['close'], label='Close Price', linewidth=2)
    if 'SMA_20' in data.columns:
        axes[0].plot(data.index, data['SMA_20'], label='SMA 20', alpha=0.7)
    if 'EMA_12' in data.columns:
        axes[0].plot(data.index, data['EMA_12'], label='EMA 12', alpha=0.7)
    
    # Bollinger Bands
    if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
        axes[0].fill_between(data.index, data['BB_upper'], data['BB_lower'], 
                           alpha=0.2, label='Bollinger Bands')
    
    axes[0].set_title(title)
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # RSI
    if 'RSI' in data.columns:
        axes[1].plot(data.index, data['RSI'], color='orange', linewidth=2)
        axes[1].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        axes[1].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        axes[1].set_ylabel('RSI')
        axes[1].set_ylim(0, 100)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # MACD
    if all(col in data.columns for col in ['MACD', 'MACD_signal']):
        axes[2].plot(data.index, data['MACD'], label='MACD', linewidth=2)
        axes[2].plot(data.index, data['MACD_signal'], label='Signal', linewidth=2)
        if 'MACD_histogram' in data.columns:
            axes[2].bar(data.index, data['MACD_histogram'], alpha=0.3, label='Histogram')
        axes[2].set_ylabel('MACD')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    # Volume
    axes[3].bar(data.index, data['volume'], alpha=0.6, color='gray')
    axes[3].set_ylabel('Volume')
    axes[3].set_xlabel('Date')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_training_progress(rewards: List[float], 
                          portfolio_values: List[float],
                          losses: List[float] = None,
                          save_path: Optional[str] = None) -> None:
    """
    Plot training progress metrics
    
    Args:
        rewards: List of episode rewards
        portfolio_values: List of portfolio values
        losses: List of training losses
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(rewards, color='blue', alpha=0.7)
    axes[0, 0].plot(pd.Series(rewards).rolling(50).mean(), color='red', linewidth=2, label='50-episode MA')
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Portfolio values
    axes[0, 1].plot(portfolio_values, color='green', alpha=0.7)
    axes[0, 1].plot(pd.Series(portfolio_values).rolling(50).mean(), color='red', linewidth=2, label='50-episode MA')
    axes[0, 1].set_title('Portfolio Value')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Portfolio Value ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Training losses
    if losses:
        axes[1, 0].plot(losses, color='orange', alpha=0.7)
        axes[1, 0].plot(pd.Series(losses).rolling(100).mean(), color='red', linewidth=2, label='100-step MA')
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_xlabel('Training Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Reward distribution
    axes[1, 1].hist(rewards, bins=50, alpha=0.7, color='purple')
    axes[1, 1].set_title('Reward Distribution')
    axes[1, 1].set_xlabel('Reward')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_backtest_results(data: pd.DataFrame, 
                         trades: List[Dict],
                         portfolio_values: List[float],
                         benchmark_values: List[float] = None,
                         save_path: Optional[str] = None) -> None:
    """
    Plot backtesting results
    
    Args:
        data: Price data DataFrame
        trades: List of trade dictionaries
        portfolio_values: Portfolio value over time
        benchmark_values: Benchmark (buy & hold) values
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # Price chart with trades
    axes[0].plot(data.index, data['close'], label='BTC Price', linewidth=2, alpha=0.8)
    
    # Mark buy and sell trades
    buy_trades = [t for t in trades if t['action'] == 'buy']
    sell_trades = [t for t in trades if t['action'] == 'sell']
    
    if buy_trades:
        buy_dates = [t['timestamp'] for t in buy_trades]
        buy_prices = [t['price'] for t in buy_trades]
        axes[0].scatter(buy_dates, buy_prices, color='green', marker='^', 
                       s=100, label='Buy', zorder=5)
    
    if sell_trades:
        sell_dates = [t['timestamp'] for t in sell_trades]
        sell_prices = [t['price'] for t in sell_trades]
        axes[0].scatter(sell_dates, sell_prices, color='red', marker='v', 
                       s=100, label='Sell', zorder=5)
    
    axes[0].set_title('Trading Strategy Performance')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Portfolio value vs benchmark
    axes[1].plot(portfolio_values, label='RL Strategy', linewidth=2, color='blue')
    if benchmark_values:
        axes[1].plot(benchmark_values, label='Buy & Hold', linewidth=2, color='orange', alpha=0.7)
    
    axes[1].set_title('Portfolio Value Comparison')
    axes[1].set_ylabel('Portfolio Value ($)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Returns distribution
    if len(portfolio_values) > 1:
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        axes[2].hist(returns, bins=50, alpha=0.7, color='purple', density=True)
        axes[2].axvline(np.mean(returns), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(returns):.4f}')
        axes[2].set_title('Daily Returns Distribution')
        axes[2].set_xlabel('Daily Return')
        axes[2].set_ylabel('Density')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_interactive_dashboard(data: pd.DataFrame, 
                               portfolio_values: List[float],
                               trades: List[Dict] = None) -> go.Figure:
    """
    Create interactive Plotly dashboard
    
    Args:
        data: Price data DataFrame
        portfolio_values: Portfolio values over time
        trades: List of trade dictionaries
        
    Returns:
        Plotly figure object
    """
    # Create subplots
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Price & Indicators', 'RSI', 'MACD', 'Portfolio Value'),
        vertical_spacing=0.08,
        row_heights=[0.4, 0.2, 0.2, 0.2]
    )
    
    # Price candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='BTC Price'
        ),
        row=1, col=1
    )
    
    # Moving averages
    if 'SMA_20' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['SMA_20'], name='SMA 20', 
                      line=dict(color='orange', width=1)),
            row=1, col=1
        )
    
    # Bollinger Bands
    if all(col in data.columns for col in ['BB_upper', 'BB_lower']):
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_upper'], name='BB Upper',
                      line=dict(color='gray', width=1, dash='dash')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=data['BB_lower'], name='BB Lower',
                      line=dict(color='gray', width=1, dash='dash'),
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )
    
    # Trade markers
    if trades:
        buy_trades = [t for t in trades if t['action'] == 'buy']
        sell_trades = [t for t in trades if t['action'] == 'sell']
        
        if buy_trades:
            fig.add_trace(
                go.Scatter(
                    x=[t['timestamp'] for t in buy_trades],
                    y=[t['price'] for t in buy_trades],
                    mode='markers',
                    marker=dict(symbol='triangle-up', size=10, color='green'),
                    name='Buy'
                ),
                row=1, col=1
            )
        
        if sell_trades:
            fig.add_trace(
                go.Scatter(
                    x=[t['timestamp'] for t in sell_trades],
                    y=[t['price'] for t in sell_trades],
                    mode='markers',
                    marker=dict(symbol='triangle-down', size=10, color='red'),
                    name='Sell'
                ),
                row=1, col=1
            )
    
    # RSI
    if 'RSI' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI',
                      line=dict(color='purple', width=2)),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'MACD' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], name='MACD',
                      line=dict(color='blue', width=2)),
            row=3, col=1
        )
        if 'MACD_signal' in data.columns:
            fig.add_trace(
                go.Scatter(x=data.index, y=data['MACD_signal'], name='Signal',
                          line=dict(color='red', width=2)),
                row=3, col=1
            )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(x=data.index[:len(portfolio_values)], y=portfolio_values,
                  name='Portfolio Value', line=dict(color='green', width=3)),
        row=4, col=1
    )
    
    # Update layout
    fig.update_layout(
        title='RL Trading Bot Dashboard',
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig


def plot_performance_metrics(metrics: Dict[str, float],
                           benchmark_metrics: Dict[str, float] = None,
                           save_path: Optional[str] = None) -> None:
    """
    Plot performance metrics comparison
    
    Args:
        metrics: Strategy performance metrics
        benchmark_metrics: Benchmark performance metrics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Metrics to plot
    metric_names = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
    metric_labels = ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)']
    
    for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
        row, col = i // 2, i % 2
        
        values = []
        labels = []
        
        if metric in metrics:
            values.append(metrics[metric] * 100 if 'return' in metric or 'drawdown' in metric or 'rate' in metric else metrics[metric])
            labels.append('Strategy')
        
        if benchmark_metrics and metric in benchmark_metrics:
            values.append(benchmark_metrics[metric] * 100 if 'return' in metric or 'drawdown' in metric or 'rate' in metric else benchmark_metrics[metric])
            labels.append('Benchmark')
        
        if values:
            colors = ['blue', 'orange'][:len(values)]
            bars = axes[row, col].bar(labels, values, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.2f}', ha='center', va='bottom')
        
        axes[row, col].set_title(label)
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    # Example usage with sample data
    print("Creating sample plots...")
    
    # Generate sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.lognormal(10, 1, 100),
        'RSI': np.random.uniform(20, 80, 100),
        'MACD': np.random.randn(100),
        'MACD_signal': np.random.randn(100)
    }, index=dates)
    
    # Plot sample data
    plot_price_and_indicators(sample_data, "Sample Bitcoin Data")
