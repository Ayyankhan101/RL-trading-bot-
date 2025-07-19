"""
Simple Trading Demo - Works without TensorFlow
Demonstrates the trading bot functionality using basic algorithms
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import sys

# Add current directory to path
sys.path.append('.')

from utils.data_utils import create_sample_data, add_fear_greed_index
from features.technical_indicators import add_technical_indicators
from utils.plotting import plot_price_and_indicators
from utils.metrics import calculate_comprehensive_metrics, print_performance_report


class SimpleTradingAgent:
    """Simple trading agent using basic rules instead of deep learning"""
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.btc_held = 0
        self.trades = []
        self.portfolio_values = [initial_balance]
        
    def make_decision(self, data_row):
        """
        Enhanced trading decision based on multiple technical indicators
        Returns: 0=Hold, 1=Buy, 2=Sell
        """
        # Get technical indicators
        rsi = data_row.get('RSI', 50)
        macd = data_row.get('MACD', 0)
        macd_signal = data_row.get('MACD_signal', 0)
        price_momentum_5 = data_row.get('Price_momentum_5', 0)
        price_momentum_10 = data_row.get('Price_momentum_10', 0)
        sma_20 = data_row.get('SMA_20', data_row.get('close', 0))
        ema_12 = data_row.get('EMA_12', data_row.get('close', 0))
        bb_upper = data_row.get('BB_upper', float('inf'))
        bb_lower = data_row.get('BB_lower', 0)
        volume_ratio = data_row.get('Volume_ratio', 1)
        fear_greed = data_row.get('Fear & Greed Index', 50)
        current_price = data_row.get('close', 0)
        
        # Calculate signals
        buy_signals = 0
        sell_signals = 0
        
        # RSI signals
        if rsi < 30:  # Oversold
            buy_signals += 2
        elif rsi > 70:  # Overbought
            sell_signals += 2
        
        # MACD signals
        if macd > macd_signal and macd > 0:  # MACD above signal and positive
            buy_signals += 1
        elif macd < macd_signal and macd < 0:  # MACD below signal and negative
            sell_signals += 1
        
        # Price momentum signals
        if price_momentum_5 > 0.02 and price_momentum_10 > 0.01:  # Strong positive momentum
            buy_signals += 1
        elif price_momentum_5 < -0.03 or price_momentum_10 < -0.02:  # Negative momentum
            sell_signals += 1
        
        # Moving average signals
        if current_price > sma_20 and ema_12 > sma_20:  # Price above MA and EMA > SMA
            buy_signals += 1
        elif current_price < sma_20 and ema_12 < sma_20:  # Price below MA and EMA < SMA
            sell_signals += 1
        
        # Bollinger Bands signals
        if current_price < bb_lower:  # Price below lower band (oversold)
            buy_signals += 1
        elif current_price > bb_upper:  # Price above upper band (overbought)
            sell_signals += 1
        
        # Volume confirmation
        if volume_ratio > 1.5:  # High volume
            if buy_signals > sell_signals:
                buy_signals += 1
            elif sell_signals > buy_signals:
                sell_signals += 1
        
        # Fear & Greed Index
        if fear_greed < 25:  # Extreme fear - contrarian buy
            buy_signals += 1
        elif fear_greed > 75:  # Extreme greed - contrarian sell
            sell_signals += 1
        
        # Decision logic
        if buy_signals >= 3 and buy_signals > sell_signals:
            return 1  # Buy
        elif sell_signals >= 3 and sell_signals > buy_signals:
            return 2  # Sell
        
        return 0  # Hold
    
    def execute_trade(self, action, price, timestamp):
        """Execute trading action"""
        if action == 1 and self.balance > 0:  # Buy
            btc_to_buy = self.balance * 0.95 / price  # Use 95% of balance
            transaction_cost = btc_to_buy * price * 0.001  # 0.1% fee
            
            if self.balance >= (btc_to_buy * price + transaction_cost):
                self.btc_held += btc_to_buy
                self.balance -= (btc_to_buy * price + transaction_cost)
                
                self.trades.append({
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': price,
                    'amount': btc_to_buy,
                    'balance': self.balance,
                    'btc_held': self.btc_held
                })
        
        elif action == 2 and self.btc_held > 0:  # Sell
            btc_to_sell = self.btc_held * 0.95  # Sell 95% of holdings
            transaction_cost = btc_to_sell * price * 0.001  # 0.1% fee
            
            self.balance += (btc_to_sell * price - transaction_cost)
            self.btc_held -= btc_to_sell
            
            self.trades.append({
                'timestamp': timestamp,
                'action': 'sell',
                'price': price,
                'amount': btc_to_sell,
                'balance': self.balance,
                'btc_held': self.btc_held
            })
    
    def get_portfolio_value(self, current_price):
        """Calculate total portfolio value"""
        return self.balance + (self.btc_held * current_price)


def run_simple_backtest():
    """Run a simple backtest demonstration with live data"""
    print("🤖 Live Bitcoin Trading Bot Demo")
    print("=" * 50)
    
    # Download live Bitcoin data
    print("📊 Downloading live Bitcoin data from Yahoo Finance...")
    try:
        from utils.data_utils import download_bitcoin_data
        data = download_bitcoin_data('BTC-USD', '2y', '1d')  # Last 2 years of daily data
        
        if data.empty:
            print("⚠️  Failed to download live data, using sample data instead...")
            data = create_sample_data()
        else:
            print(f"✅ Downloaded {len(data)} days of live Bitcoin data")
            # Handle timezone-aware timestamps properly
            try:
                min_date = data['timestamp'].min()
                max_date = data['timestamp'].max()
                if hasattr(min_date, 'date'):
                    print(f"📅 Date range: {min_date.date()} to {max_date.date()}")
                else:
                    print(f"📅 Date range: {min_date} to {max_date}")
            except:
                print(f"📅 Data contains {len(data)} recent records")
            print(f"💰 Price range: ${data['close'].min():,.2f} - ${data['close'].max():,.2f}")
            
            # Set timestamp as index for plotting (handle both cases)
            if 'timestamp' in data.columns:
                data.set_index('timestamp', inplace=True)
            elif data.index.name != 'timestamp':
                # If timestamp is already index, ensure it's named properly
                data.index.name = 'timestamp'
            
    except Exception as e:
        print(f"⚠️  Error downloading live data: {e}")
        print("📊 Using sample data instead...")
        data = create_sample_data()
    
    # Add technical indicators
    print("📈 Adding technical indicators...")
    data = add_technical_indicators(data)
    data = add_fear_greed_index(data)
    data = data.dropna()
    
    print(f"✅ Data prepared: {len(data)} rows")
    
    # Initialize agent
    agent = SimpleTradingAgent(initial_balance=10000)
    
    # Run backtest
    print("🚀 Running backtest simulation...")
    
    for i, (timestamp, row) in enumerate(data.iterrows()):
        # Make trading decision
        action = agent.make_decision(row)
        
        # Execute trade
        if action != 0:
            agent.execute_trade(action, row['close'], timestamp)
        
        # Record portfolio value
        portfolio_value = agent.get_portfolio_value(row['close'])
        agent.portfolio_values.append(portfolio_value)
    
    # Calculate performance metrics
    print("📊 Calculating performance metrics...")
    
    # Calculate buy & hold benchmark
    initial_price = data['close'].iloc[0]
    final_price = data['close'].iloc[-1]
    btc_amount = agent.initial_balance / initial_price
    benchmark_final = btc_amount * final_price
    
    # Create benchmark portfolio values
    benchmark_values = []
    for price in data['close']:
        benchmark_values.append(btc_amount * price)
    
    # Add trade P&L
    for i, trade in enumerate(agent.trades):
        if trade['action'] == 'sell' and i > 0:
            # Find corresponding buy trade
            for j in range(i-1, -1, -1):
                if agent.trades[j]['action'] == 'buy':
                    buy_price = agent.trades[j]['price']
                    sell_price = trade['price']
                    pnl = (sell_price - buy_price) * trade['amount']
                    trade['pnl'] = pnl
                    break
        else:
            trade['pnl'] = 0
    
    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(
        agent.portfolio_values,
        agent.trades,
        benchmark_values
    )
    
    # Print results
    print("\n" + "="*60)
    print("📈 LIVE BITCOIN TRADING RESULTS")
    print("="*60)
    
    final_value = agent.portfolio_values[-1]
    total_return = (final_value - agent.initial_balance) / agent.initial_balance
    benchmark_return = (benchmark_final - agent.initial_balance) / agent.initial_balance
    
    # Current market info
    current_price = data['close'].iloc[-1]
    price_change_24h = (current_price - data['close'].iloc[-2]) / data['close'].iloc[-2]
    
    print(f"🔴 LIVE MARKET DATA")
    print(f"Current BTC Price:   ${current_price:,.2f}")
    print(f"24h Change:          {price_change_24h:+.2%}")
    print(f"Period High:         ${data['close'].max():,.2f}")
    print(f"Period Low:          ${data['close'].min():,.2f}")
    print(f"")
    print(f"💼 TRADING PERFORMANCE")
    print(f"Initial Balance:     ${agent.initial_balance:,.2f}")
    print(f"Final Value:         ${final_value:,.2f}")
    print(f"Total Return:        {total_return:.2%}")
    print(f"Benchmark Return:    {benchmark_return:.2%}")
    print(f"Outperformance:      {(total_return - benchmark_return):+.2%}")
    print(f"Total Trades:        {len(agent.trades)}")
    
    if len(agent.trades) > 0:
        winning_trades = [t for t in agent.trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in agent.trades if t.get('pnl', 0) < 0]
        total_pnl = sum(t.get('pnl', 0) for t in agent.trades)
        
        print(f"Winning Trades:      {len(winning_trades)}")
        print(f"Losing Trades:       {len(losing_trades)}")
        print(f"Win Rate:            {len(winning_trades)/len(agent.trades):.2%}")
        print(f"Total P&L:           ${total_pnl:,.2f}")
        
        if winning_trades:
            avg_win = sum(t.get('pnl', 0) for t in winning_trades) / len(winning_trades)
            print(f"Average Win:         ${avg_win:,.2f}")
        
        if losing_trades:
            avg_loss = sum(t.get('pnl', 0) for t in losing_trades) / len(losing_trades)
            print(f"Average Loss:        ${avg_loss:,.2f}")
    
    # Print detailed metrics
    print_performance_report(metrics)
    
    # Plot results with improved clarity and readability
    print("\n📊 Generating clear, readable plots...")
    
    # Set up matplotlib for better display
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'figure.titlesize': 16
    })
    
    # Create separate, larger plots for better readability
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Bitcoin Price Chart with Trading Signals
    ax1 = plt.subplot(3, 2, (1, 2))  # Top row, spanning both columns
    ax1.plot(data.index, data['close'], label='Bitcoin Price', linewidth=2, color='#1f77b4')
    
    # Add trading signals
    if agent.trades:
        buy_trades = [t for t in agent.trades if t['action'] == 'buy']
        sell_trades = [t for t in agent.trades if t['action'] == 'sell']
        
        if buy_trades:
            buy_dates = [t['timestamp'] for t in buy_trades]
            buy_prices = [t['price'] for t in buy_trades]
            ax1.scatter(buy_dates, buy_prices, color='green', marker='^', 
                       s=80, label=f'Buy Signals ({len(buy_trades)})', alpha=0.8, zorder=5)
        
        if sell_trades:
            sell_dates = [t['timestamp'] for t in sell_trades]
            sell_prices = [t['price'] for t in sell_trades]
            ax1.scatter(sell_dates, sell_prices, color='red', marker='v', 
                       s=80, label=f'Sell Signals ({len(sell_trades)})', alpha=0.8, zorder=5)
    
    ax1.set_title('Bitcoin Price with Trading Signals', fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel('Price (USD)', fontsize=12)
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax1.tick_params(axis='x', rotation=45)
    
    # Format y-axis for better readability
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 2. Portfolio Performance Comparison
    ax2 = plt.subplot(3, 2, 3)
    ax2.plot(data.index, agent.portfolio_values[1:], label='Trading Strategy', 
             linewidth=3, color='#ff7f0e', alpha=0.9)
    ax2.plot(data.index, benchmark_values, label='Buy & Hold', 
             linewidth=2, color='#2ca02c', alpha=0.8, linestyle='--')
    
    ax2.set_title('Portfolio Performance Comparison', fontsize=14, fontweight='bold', pad=15)
    ax2.set_ylabel('Portfolio Value (USD)', fontsize=12)
    ax2.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax2.tick_params(axis='x', rotation=45)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # 3. RSI Technical Indicator
    ax3 = plt.subplot(3, 2, 4)
    if 'RSI' in data.columns:
        ax3.plot(data.index, data['RSI'], color='purple', linewidth=2, alpha=0.8)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Overbought (70)')
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, linewidth=2, label='Oversold (30)')
        ax3.fill_between(data.index, 30, 70, alpha=0.1, color='gray', label='Normal Range')
        
        ax3.set_title('RSI Technical Indicator', fontsize=14, fontweight='bold', pad=15)
        ax3.set_ylabel('RSI Value', fontsize=12)
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
        ax3.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax3.tick_params(axis='x', rotation=45)
    
    # 4. Trading Performance Metrics
    ax4 = plt.subplot(3, 2, 5)
    
    # Create a summary metrics visualization
    metrics_labels = ['Total Return', 'Annualized Return', 'Sharpe Ratio', 'Max Drawdown']
    strategy_values = [
        total_return * 100,
        metrics.get('annualized_return', 0) * 100,
        metrics.get('sharpe_ratio', 0),
        abs(metrics.get('max_drawdown', 0)) * 100
    ]
    benchmark_values_metrics = [
        benchmark_return * 100,
        (benchmark_return / 2) * 100,  # Approximate annualized
        0.5,  # Approximate Sharpe for buy & hold
        20  # Approximate max drawdown
    ]
    
    x = range(len(metrics_labels))
    width = 0.35
    
    bars1 = ax4.bar([i - width/2 for i in x], strategy_values, width, 
                    label='Trading Strategy', color='#ff7f0e', alpha=0.8)
    bars2 = ax4.bar([i + width/2 for i in x], benchmark_values_metrics, width,
                    label='Buy & Hold', color='#2ca02c', alpha=0.8)
    
    ax4.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=15)
    ax4.set_ylabel('Value (%)', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics_labels, rotation=45, ha='right')
    ax4.legend(frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%' if abs(height) > 1 else f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)
    
    # 5. Trade Distribution
    ax5 = plt.subplot(3, 2, 6)
    
    if agent.trades:
        # Calculate P&L for each trade
        trade_pnls = []
        for trade in agent.trades:
            if 'pnl' in trade:
                trade_pnls.append(trade['pnl'])
        
        if trade_pnls:
            winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
            losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
            
            # Create histogram
            ax5.hist(winning_trades, bins=20, alpha=0.7, color='green', 
                    label=f'Winning Trades ({len(winning_trades)})', edgecolor='black')
            ax5.hist(losing_trades, bins=20, alpha=0.7, color='red', 
                    label=f'Losing Trades ({len(losing_trades)})', edgecolor='black')
            
            ax5.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold', pad=15)
            ax5.set_xlabel('Profit/Loss (USD)', fontsize=12)
            ax5.set_ylabel('Number of Trades', fontsize=12)
            ax5.legend(frameon=True, fancybox=True, shadow=True)
            ax5.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.5)
            ax5.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    
    # Adjust layout with more spacing
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
    
    # Save with high quality
    plt.savefig('demo_results.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('demo_results.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Display the plot
    plt.show()
    
    print(f"\n✅ Demo completed successfully!")
    print(f"📊 Results saved to: demo_results.png")
    
    return agent, data, metrics


def run_dashboard_demo():
    """Run a simple dashboard demo"""
    print("\n🌐 Starting Streamlit Dashboard Demo...")
    print("Run the following command to start the dashboard:")
    print("streamlit run dashboard.py")
    print("\nNote: The dashboard will work with sample data even without TensorFlow")


if __name__ == "__main__":
    try:
        # Run the simple backtest demo
        agent, data, metrics = run_simple_backtest()
        
        # Show dashboard info
        run_dashboard_demo()
        
        print("\n" + "="*60)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("✅ Simple trading algorithm implemented")
        print("✅ Backtesting completed")
        print("✅ Performance metrics calculated")
        print("✅ Visualization generated")
        print("\nNext steps:")
        print("1. Install TensorFlow for deep learning features")
        print("2. Run: streamlit run dashboard.py")
        print("3. Explore the Jupyter notebooks")
        print("4. Train actual RL models when TensorFlow is available")
        
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print("Please ensure all required packages are installed")
        import traceback
        traceback.print_exc()
