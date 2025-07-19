"""
Live Bitcoin Trading Demo with Real-Time Data
Enhanced version with live market data and advanced analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
import time

# Add current directory to path
sys.path.append('.')

from utils.data_utils import download_bitcoin_data, add_fear_greed_index
from features.technical_indicators import add_technical_indicators
from utils.plotting import plot_price_and_indicators
from utils.metrics import calculate_comprehensive_metrics, print_performance_report


class LiveTradingBot:
    """Enhanced trading bot with live data capabilities"""
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.btc_held = 0
        self.trades = []
        self.portfolio_values = [initial_balance]
        self.signals_history = []
        
    def analyze_market_conditions(self, data_row):
        """Analyze current market conditions"""
        rsi = data_row.get('RSI', 50)
        macd = data_row.get('MACD', 0)
        volatility = data_row.get('Volatility', 0)
        volume_ratio = data_row.get('Volume_ratio', 1)
        fear_greed = data_row.get('Fear & Greed Index', 50)
        
        conditions = {
            'trend': 'neutral',
            'volatility': 'normal',
            'sentiment': 'neutral',
            'volume': 'normal'
        }
        
        # Trend analysis
        if rsi > 60 and macd > 0:
            conditions['trend'] = 'bullish'
        elif rsi < 40 and macd < 0:
            conditions['trend'] = 'bearish'
        
        # Volatility analysis
        if volatility > data_row.get('close', 0) * 0.05:
            conditions['volatility'] = 'high'
        elif volatility < data_row.get('close', 0) * 0.02:
            conditions['volatility'] = 'low'
        
        # Sentiment analysis
        if fear_greed < 25:
            conditions['sentiment'] = 'extreme_fear'
        elif fear_greed < 45:
            conditions['sentiment'] = 'fear'
        elif fear_greed > 75:
            conditions['sentiment'] = 'extreme_greed'
        elif fear_greed > 55:
            conditions['sentiment'] = 'greed'
        
        # Volume analysis
        if volume_ratio > 2:
            conditions['volume'] = 'very_high'
        elif volume_ratio > 1.5:
            conditions['volume'] = 'high'
        elif volume_ratio < 0.5:
            conditions['volume'] = 'low'
        
        return conditions
    
    def make_decision(self, data_row):
        """Enhanced decision making with market condition analysis"""
        # Get market conditions
        conditions = self.analyze_market_conditions(data_row)
        
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
        
        # Calculate signals with market condition weighting
        buy_signals = 0
        sell_signals = 0
        confidence = 0
        
        # RSI signals (weighted by trend)
        if rsi < 30:
            buy_signals += 3 if conditions['trend'] != 'bearish' else 2
            confidence += 0.2
        elif rsi > 70:
            sell_signals += 3 if conditions['trend'] != 'bullish' else 2
            confidence += 0.2
        
        # MACD signals
        if macd > macd_signal and macd > 0:
            buy_signals += 2
            confidence += 0.15
        elif macd < macd_signal and macd < 0:
            sell_signals += 2
            confidence += 0.15
        
        # Momentum signals (stronger in trending markets)
        momentum_weight = 2 if conditions['trend'] != 'neutral' else 1
        if price_momentum_5 > 0.02 and price_momentum_10 > 0.01:
            buy_signals += momentum_weight
            confidence += 0.1
        elif price_momentum_5 < -0.03 or price_momentum_10 < -0.02:
            sell_signals += momentum_weight
            confidence += 0.1
        
        # Moving average signals
        if current_price > sma_20 and ema_12 > sma_20:
            buy_signals += 1
        elif current_price < sma_20 and ema_12 < sma_20:
            sell_signals += 1
        
        # Bollinger Bands (stronger in low volatility)
        bb_weight = 2 if conditions['volatility'] == 'low' else 1
        if current_price < bb_lower:
            buy_signals += bb_weight
            confidence += 0.1
        elif current_price > bb_upper:
            sell_signals += bb_weight
            confidence += 0.1
        
        # Volume confirmation (stronger signals with high volume)
        if conditions['volume'] in ['high', 'very_high']:
            if buy_signals > sell_signals:
                buy_signals += 1
                confidence += 0.1
            elif sell_signals > buy_signals:
                sell_signals += 1
                confidence += 0.1
        
        # Sentiment signals (contrarian approach)
        if conditions['sentiment'] == 'extreme_fear':
            buy_signals += 2
            confidence += 0.15
        elif conditions['sentiment'] == 'extreme_greed':
            sell_signals += 2
            confidence += 0.15
        
        # Record signal analysis
        signal_data = {
            'timestamp': data_row.name if hasattr(data_row, 'name') else datetime.now(),
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'confidence': confidence,
            'conditions': conditions,
            'rsi': rsi,
            'macd': macd,
            'price': current_price
        }
        self.signals_history.append(signal_data)
        
        # Decision logic with confidence threshold
        min_signals = 3
        min_confidence = 0.3
        
        if (buy_signals >= min_signals and 
            buy_signals > sell_signals and 
            confidence >= min_confidence):
            return 1  # Buy
        elif (sell_signals >= min_signals and 
              sell_signals > buy_signals and 
              confidence >= min_confidence):
            return 2  # Sell
        
        return 0  # Hold
    
    def execute_trade(self, action, price, timestamp):
        """Execute trading action with enhanced logging"""
        if action == 1 and self.balance > 0:  # Buy
            btc_to_buy = self.balance * 0.95 / price
            transaction_cost = btc_to_buy * price * 0.001
            
            if self.balance >= (btc_to_buy * price + transaction_cost):
                self.btc_held += btc_to_buy
                self.balance -= (btc_to_buy * price + transaction_cost)
                
                trade = {
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': price,
                    'amount': btc_to_buy,
                    'balance': self.balance,
                    'btc_held': self.btc_held,
                    'portfolio_value': self.get_portfolio_value(price),
                    'transaction_cost': transaction_cost
                }
                self.trades.append(trade)
                return True
        
        elif action == 2 and self.btc_held > 0:  # Sell
            btc_to_sell = self.btc_held * 0.95
            transaction_cost = btc_to_sell * price * 0.001
            
            self.balance += (btc_to_sell * price - transaction_cost)
            self.btc_held -= btc_to_sell
            
            trade = {
                'timestamp': timestamp,
                'action': 'sell',
                'price': price,
                'amount': btc_to_sell,
                'balance': self.balance,
                'btc_held': self.btc_held,
                'portfolio_value': self.get_portfolio_value(price),
                'transaction_cost': transaction_cost
            }
            self.trades.append(trade)
            return True
        
        return False
    
    def get_portfolio_value(self, current_price):
        """Calculate total portfolio value"""
        return self.balance + (self.btc_held * current_price)
    
    def get_trading_summary(self):
        """Get comprehensive trading summary"""
        if not self.trades:
            return {}
        
        buy_trades = [t for t in self.trades if t['action'] == 'buy']
        sell_trades = [t for t in self.trades if t['action'] == 'sell']
        
        # Calculate P&L for completed trades
        total_pnl = 0
        completed_trades = 0
        
        for sell_trade in sell_trades:
            # Find the most recent buy trade before this sell
            for buy_trade in reversed(buy_trades):
                if buy_trade['timestamp'] < sell_trade['timestamp']:
                    pnl = (sell_trade['price'] - buy_trade['price']) * sell_trade['amount']
                    total_pnl += pnl
                    sell_trade['pnl'] = pnl
                    completed_trades += 1
                    break
        
        return {
            'total_trades': len(self.trades),
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'completed_trades': completed_trades,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / max(1, completed_trades),
            'total_fees': sum(t.get('transaction_cost', 0) for t in self.trades)
        }


def run_live_trading_demo():
    """Run enhanced live trading demo"""
    print("🚀 LIVE BITCOIN TRADING BOT - ENHANCED DEMO")
    print("=" * 60)
    
    # Download live data
    print("📡 Fetching live Bitcoin data...")
    try:
        data = download_bitcoin_data('BTC-USD', '1y', '1d')  # Last 1 year
        
        if data.empty:
            print("❌ Failed to fetch live data")
            return
        
        print(f"✅ Downloaded {len(data)} days of live Bitcoin data")
        # Handle timestamp column properly
        if 'timestamp' in data.columns:
            print(f"📅 Period: {data['timestamp'].min().date()} to {data['timestamp'].max().date()}")
            data.set_index('timestamp', inplace=True)
        else:
            print(f"📅 Period: {data.index.min().date()} to {data.index.max().date()}")
            
        print(f"💰 Price range: ${data['close'].min():,.2f} - ${data['close'].max():,.2f}")
        
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return
    
    # Add technical indicators
    print("🔧 Calculating technical indicators...")
    data = add_technical_indicators(data)
    data = add_fear_greed_index(data)
    data = data.dropna()
    
    print(f"✅ Technical analysis complete - {len(data)} data points ready")
    
    # Initialize enhanced trading bot
    bot = LiveTradingBot(initial_balance=10000)
    
    # Run simulation
    print("🤖 Running live trading simulation...")
    print("⏳ Processing market data...")
    
    for i, (timestamp, row) in enumerate(data.iterrows()):
        # Show progress
        if i % 50 == 0:
            progress = (i / len(data)) * 100
            print(f"Progress: {progress:.1f}% - Processing {timestamp.date()}")
        
        # Make trading decision
        action = bot.make_decision(row)
        
        # Execute trade if decision made
        if action != 0:
            success = bot.execute_trade(action, row['close'], timestamp)
            if success:
                action_name = "BUY" if action == 1 else "SELL"
                print(f"🔄 {action_name} executed at ${row['close']:,.2f} on {timestamp.date()}")
        
        # Record portfolio value
        portfolio_value = bot.get_portfolio_value(row['close'])
        bot.portfolio_values.append(portfolio_value)
    
    # Calculate performance
    print("\n📊 Calculating performance metrics...")
    
    # Benchmark calculation
    initial_price = data['close'].iloc[0]
    final_price = data['close'].iloc[-1]
    btc_amount = bot.initial_balance / initial_price
    benchmark_final = btc_amount * final_price
    benchmark_return = (benchmark_final - bot.initial_balance) / bot.initial_balance
    
    # Trading performance
    final_value = bot.portfolio_values[-1]
    total_return = (final_value - bot.initial_balance) / bot.initial_balance
    
    # Get trading summary
    summary = bot.get_trading_summary()
    
    # Display results
    print("\n" + "="*70)
    print("📈 LIVE BITCOIN TRADING RESULTS")
    print("="*70)
    
    # Market overview
    current_price = data['close'].iloc[-1]
    price_change = (current_price - data['close'].iloc[0]) / data['close'].iloc[0]
    
    print(f"🔴 MARKET OVERVIEW")
    print(f"Current BTC Price:    ${current_price:,.2f}")
    print(f"Period Change:        {price_change:+.2%}")
    print(f"Period High:          ${data['close'].max():,.2f}")
    print(f"Period Low:           ${data['close'].min():,.2f}")
    print(f"Average Volume:       {data['volume'].mean():,.0f}")
    
    print(f"\n💼 TRADING PERFORMANCE")
    print(f"Initial Balance:      ${bot.initial_balance:,.2f}")
    print(f"Final Portfolio:      ${final_value:,.2f}")
    print(f"Strategy Return:      {total_return:+.2%}")
    print(f"Buy & Hold Return:    {benchmark_return:+.2%}")
    print(f"Outperformance:       {(total_return - benchmark_return):+.2%}")
    
    print(f"\n📋 TRADING ACTIVITY")
    print(f"Total Trades:         {summary.get('total_trades', 0)}")
    print(f"Buy Orders:           {summary.get('buy_trades', 0)}")
    print(f"Sell Orders:          {summary.get('sell_trades', 0)}")
    print(f"Completed Trades:     {summary.get('completed_trades', 0)}")
    print(f"Total P&L:            ${summary.get('total_pnl', 0):,.2f}")
    print(f"Total Fees:           ${summary.get('total_fees', 0):,.2f}")
    
    if summary.get('completed_trades', 0) > 0:
        print(f"Avg P&L per Trade:    ${summary.get('avg_pnl_per_trade', 0):,.2f}")
    
    # Generate visualization
    print(f"\n📊 Generating performance charts...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Price chart with trades
    axes[0, 0].plot(data.index, data['close'], label='BTC Price', alpha=0.8, linewidth=1)
    
    if bot.trades:
        buy_trades = [t for t in bot.trades if t['action'] == 'buy']
        sell_trades = [t for t in bot.trades if t['action'] == 'sell']
        
        if buy_trades:
            buy_dates = [t['timestamp'] for t in buy_trades]
            buy_prices = [t['price'] for t in buy_trades]
            axes[0, 0].scatter(buy_dates, buy_prices, color='green', marker='^', 
                             s=60, label=f'Buy ({len(buy_trades)})', zorder=5)
        
        if sell_trades:
            sell_dates = [t['timestamp'] for t in sell_trades]
            sell_prices = [t['price'] for t in sell_trades]
            axes[0, 0].scatter(sell_dates, sell_prices, color='red', marker='v', 
                             s=60, label=f'Sell ({len(sell_trades)})', zorder=5)
    
    axes[0, 0].set_title('Bitcoin Price with Trading Signals', fontsize=14)
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Portfolio performance
    portfolio_dates = data.index[:len(bot.portfolio_values)]
    benchmark_values = [bot.initial_balance * (data['close'].iloc[i] / data['close'].iloc[0]) 
                       for i in range(len(portfolio_dates))]
    
    axes[0, 1].plot(portfolio_dates, bot.portfolio_values[:len(portfolio_dates)], 
                   label='Trading Strategy', linewidth=2, color='blue')
    axes[0, 1].plot(portfolio_dates, benchmark_values, 
                   label='Buy & Hold', linewidth=2, color='orange', alpha=0.7)
    axes[0, 1].set_title('Portfolio Performance Comparison', fontsize=14)
    axes[0, 1].set_ylabel('Portfolio Value ($)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # RSI with signals
    if 'RSI' in data.columns:
        axes[1, 0].plot(data.index, data['RSI'], color='purple', linewidth=1)
        axes[1, 0].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        axes[1, 0].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        axes[1, 0].set_title('RSI Indicator', fontsize=14)
        axes[1, 0].set_ylabel('RSI')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Trading signals confidence
    if bot.signals_history:
        signal_dates = [s['timestamp'] for s in bot.signals_history]
        confidences = [s['confidence'] for s in bot.signals_history]
        axes[1, 1].plot(signal_dates, confidences, color='green', alpha=0.7)
        axes[1, 1].set_title('Trading Signal Confidence', fontsize=14)
        axes[1, 1].set_ylabel('Confidence Score')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('live_trading_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Analysis complete!")
    print(f"📊 Charts saved as: live_trading_results.png")
    print(f"🎯 Ready for live trading implementation!")
    
    return bot, data


if __name__ == "__main__":
    try:
        bot, data = run_live_trading_demo()
        
        print("\n" + "="*70)
        print("🎉 LIVE TRADING DEMO COMPLETED!")
        print("="*70)
        print("✅ Live Bitcoin data processed")
        print("✅ Advanced trading algorithm executed")
        print("✅ Performance analysis completed")
        print("✅ Comprehensive charts generated")
        print("\n🚀 Next steps:")
        print("1. Run: streamlit run dashboard.py (for web interface)")
        print("2. Explore signal analysis in the charts")
        print("3. Fine-tune trading parameters")
        print("4. Consider paper trading implementation")
        
    except Exception as e:
        print(f"❌ Error in live trading demo: {e}")
        import traceback
        traceback.print_exc()
