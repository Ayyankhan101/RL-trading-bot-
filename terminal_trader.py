#!/usr/bin/env python3
"""
Real-time Terminal Trading Interface
Live Bitcoin trading signals and analysis in the terminal
"""

import sys
import os
import time
import threading
from datetime import datetime, timedelta
from collections import deque
import signal

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import pandas as pd
    import numpy as np
    from utils.data_utils import download_bitcoin_data
    from features.technical_indicators import add_technical_indicators
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all dependencies are installed")
    sys.exit(1)

class TerminalTrader:
    """Real-time terminal-based trading interface"""
    
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.btc_held = 0
        self.trades = []
        self.price_history = deque(maxlen=100)
        self.signal_history = deque(maxlen=20)
        self.running = True
        
        # Optimized trading parameters for better win rate
        self.rsi_oversold = 25  # More strict oversold (was 30)
        self.rsi_overbought = 75  # More strict overbought (was 70)
        self.min_trade_amount = 100  # Minimum $100 per trade
        
        # Enhanced signal confirmation
        self.trend_confirmation_periods = 3  # Require 3 periods of trend
        self.volume_threshold = 1.2  # Require 20% above average volume
        self.stop_loss_pct = 0.03  # 3% stop loss
        self.take_profit_pct = 0.06  # 6% take profit (2:1 risk-reward)
        
        # Display settings
        self.refresh_rate = 5  # seconds
        self.last_update = None
        
    def clear_screen(self):
        """Clear terminal screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def get_live_data(self):
        """Fetch latest Bitcoin data"""
        try:
            # Get recent data (last 7 days with hourly intervals for freshness)
            data = download_bitcoin_data('BTC-USD', '7d', '1h')
            if not data.empty:
                # Add technical indicators
                data = add_technical_indicators(data)
                return data
        except Exception as e:
            print(f"❌ Error fetching data: {e}")
        return None
    
    def analyze_market_condition(self, data):
        """Analyze current market conditions"""
        if data is None or len(data) < 20:
            return "UNKNOWN", "Insufficient data"
        
        latest = data.iloc[-1]
        prev_24h = data.iloc[-24] if len(data) >= 24 else data.iloc[0]
        
        # Price analysis
        current_price = latest['close']
        price_change_24h = ((current_price - prev_24h['close']) / prev_24h['close']) * 100
        
        # Technical indicators
        rsi = latest.get('RSI', 50)
        sma_20 = latest.get('SMA_20', current_price)
        sma_50 = latest.get('SMA_50', current_price)
        
        # Volume analysis
        avg_volume = data['volume'].tail(24).mean()
        current_volume = latest['volume']
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Market condition analysis
        conditions = []
        
        # Trend analysis
        if current_price > sma_20 > sma_50:
            conditions.append("BULLISH_TREND")
        elif current_price < sma_20 < sma_50:
            conditions.append("BEARISH_TREND")
        else:
            conditions.append("SIDEWAYS")
        
        # RSI conditions
        if rsi < self.rsi_oversold:
            conditions.append("OVERSOLD")
        elif rsi > self.rsi_overbought:
            conditions.append("OVERBOUGHT")
        
        # Volume conditions
        if volume_ratio > 1.5:
            conditions.append("HIGH_VOLUME")
        elif volume_ratio < 0.5:
            conditions.append("LOW_VOLUME")
        
        # Volatility
        volatility = data['close'].tail(24).std() / current_price * 100
        if volatility > 5:
            conditions.append("HIGH_VOLATILITY")
        
        return conditions, {
            'price': current_price,
            'price_change_24h': price_change_24h,
            'rsi': rsi,
            'volume_ratio': volume_ratio,
            'volatility': volatility,
            'sma_20': sma_20,
            'sma_50': sma_50
        }
    
    def generate_trading_signal(self, data, market_info):
        """Enhanced trading signal with better win rate optimization"""
        if data is None or len(data) < 20:
            return "HOLD", "Insufficient data", 0
        
        conditions, metrics = market_info
        current_price = metrics['price']
        rsi = metrics['rsi']
        price_change_24h = metrics['price_change_24h']
        volume_ratio = metrics['volume_ratio']
        
        # Enhanced signal scoring system
        buy_score = 0
        sell_score = 0
        confidence = 0
        reasons = []
        
        # 1. STRICT RSI signals (more selective)
        if rsi < 20:  # Extremely oversold
            buy_score += 4
            confidence += 40
            reasons.append(f"Extremely oversold RSI({rsi:.1f})")
        elif rsi < self.rsi_oversold:  # Oversold
            buy_score += 2
            confidence += 25
            reasons.append(f"Oversold RSI({rsi:.1f})")
        elif rsi > 80:  # Extremely overbought
            sell_score += 4
            confidence += 40
            reasons.append(f"Extremely overbought RSI({rsi:.1f})")
        elif rsi > self.rsi_overbought:  # Overbought
            sell_score += 2
            confidence += 25
            reasons.append(f"Overbought RSI({rsi:.1f})")
        
        # 2. TREND CONFIRMATION (require stronger trends)
        if "BULLISH_TREND" in conditions and price_change_24h > 2:
            buy_score += 2
            confidence += 20
            reasons.append("Strong bullish trend")
        elif "BEARISH_TREND" in conditions and price_change_24h < -2:
            sell_score += 2
            confidence += 20
            reasons.append("Strong bearish trend")
        
        # 3. VOLUME CONFIRMATION (require significant volume)
        if volume_ratio >= self.volume_threshold:
            confidence += 25
            reasons.append(f"High volume confirmation({volume_ratio:.1f}x)")
        else:
            confidence -= 15  # Penalize low volume
            reasons.append("Low volume - reducing confidence")
        
        # 4. MOMENTUM CONFIRMATION
        if price_change_24h < -7:  # Very strong dip
            buy_score += 3
            confidence += 25
            reasons.append(f"Strong dip({price_change_24h:+.1f}%)")
        elif price_change_24h < -3:  # Moderate dip
            buy_score += 1
            confidence += 10
        elif price_change_24h > 7:  # Very strong rally
            sell_score += 2
            confidence += 15
            reasons.append(f"Strong rally({price_change_24h:+.1f}%)")
        
        # 5. MULTI-TIMEFRAME CONFIRMATION
        if len(data) >= 24:
            # Check if trend is consistent over multiple periods
            recent_prices = data['close'].tail(self.trend_confirmation_periods)
            if len(recent_prices) >= 3:
                trend_up = all(recent_prices.iloc[i] > recent_prices.iloc[i-1] for i in range(1, len(recent_prices)))
                trend_down = all(recent_prices.iloc[i] < recent_prices.iloc[i-1] for i in range(1, len(recent_prices)))
                
                if trend_up and buy_score > 0:
                    buy_score += 1
                    confidence += 15
                    reasons.append("Consistent uptrend")
                elif trend_down and sell_score > 0:
                    sell_score += 1
                    confidence += 15
                    reasons.append("Consistent downtrend")
        
        # 6. VOLATILITY FILTER (avoid trading in high volatility)
        volatility = metrics.get('volatility', 0)
        if volatility > 8:  # High volatility
            confidence -= 20
            reasons.append("High volatility - reducing confidence")
        
        # 7. GENERATE SIGNAL WITH HIGHER THRESHOLDS
        confidence = max(0, min(confidence, 100))
        
        if buy_score >= 4 and buy_score > sell_score:  # Require higher score
            signal = "BUY"
            reason = "BUY: " + ", ".join(reasons[:3])
        elif sell_score >= 4 and sell_score > buy_score:  # Require higher score
            signal = "SELL"
            reason = "SELL: " + ", ".join(reasons[:3])
        else:
            signal = "HOLD"
            reason = f"HOLD: Insufficient signals (buy:{buy_score}, sell:{sell_score})"
        
        return signal, reason, confidence
    
    def execute_trade(self, signal, price, confidence):
        """Enhanced trade execution with better risk management"""
        if confidence < 70:  # Require higher confidence (was 60)
            return False, f"Low confidence signal ({confidence:.0f}% < 70%)"
        
        current_value = self.get_portfolio_value(price)
        
        if signal == "BUY" and self.balance > self.min_trade_amount:
            # More conservative position sizing (15% instead of 25%)
            trade_amount = min(self.balance * 0.15, self.balance - 100)  # Keep $100 buffer
            btc_amount = trade_amount / price
            
            self.balance -= trade_amount
            self.btc_held += btc_amount
            
            trade = {
                'timestamp': datetime.now(),
                'action': 'BUY',
                'price': price,
                'amount': btc_amount,
                'value': trade_amount,
                'confidence': confidence,
                'stop_loss': price * (1 - self.stop_loss_pct),
                'take_profit': price * (1 + self.take_profit_pct)
            }
            self.trades.append(trade)
            return True, f"Bought {btc_amount:.6f} BTC for ${trade_amount:.2f} (SL: ${trade['stop_loss']:.0f}, TP: ${trade['take_profit']:.0f})"
        
        elif signal == "SELL" and self.btc_held > 0:
            # More conservative exit (30% instead of 50%)
            btc_to_sell = self.btc_held * 0.3
            trade_value = btc_to_sell * price
            
            self.balance += trade_value
            self.btc_held -= btc_to_sell
            
            trade = {
                'timestamp': datetime.now(),
                'action': 'SELL',
                'price': price,
                'amount': btc_to_sell,
                'value': trade_value,
                'confidence': confidence
            }
            self.trades.append(trade)
            return True, f"Sold {btc_to_sell:.6f} BTC for ${trade_value:.2f}"
        
        return False, "No trade executed"
    
    def get_portfolio_value(self, current_price):
        """Calculate total portfolio value"""
        return self.balance + (self.btc_held * current_price)
    
    def display_header(self):
        """Display terminal header"""
        print("🚀" + "="*78 + "🚀")
        print("🔴 LIVE BITCOIN TRADING TERMINAL - REAL-TIME ANALYSIS 🔴".center(80))
        print("🚀" + "="*78 + "🚀")
        print()
    
    def display_market_data(self, data, market_info):
        """Display current market data"""
        if data is None:
            print("❌ No market data available")
            return
        
        conditions, metrics = market_info
        current_price = metrics['price']
        price_change_24h = metrics['price_change_24h']
        rsi = metrics['rsi']
        volume_ratio = metrics['volume_ratio']
        volatility = metrics['volatility']
        
        # Price display with color coding
        price_color = "🟢" if price_change_24h > 0 else "🔴" if price_change_24h < 0 else "⚪"
        
        print("📊 LIVE MARKET DATA")
        print("-" * 50)
        print(f"{price_color} Bitcoin Price:     ${current_price:,.2f}")
        print(f"📈 24h Change:        {price_change_24h:+.2f}%")
        print(f"📊 RSI:              {rsi:.1f}")
        print(f"📈 Volume Ratio:      {volume_ratio:.2f}x")
        print(f"⚡ Volatility:        {volatility:.2f}%")
        print(f"🕐 Last Update:       {datetime.now().strftime('%H:%M:%S')}")
        print()
        
        # Market conditions
        print("🎯 MARKET CONDITIONS")
        print("-" * 50)
        for condition in conditions:
            emoji = {
                'BULLISH_TREND': '🟢',
                'BEARISH_TREND': '🔴',
                'SIDEWAYS': '⚪',
                'OVERSOLD': '🟢',
                'OVERBOUGHT': '🔴',
                'HIGH_VOLUME': '📈',
                'LOW_VOLUME': '📉',
                'HIGH_VOLATILITY': '⚡'
            }.get(condition, '📊')
            print(f"{emoji} {condition.replace('_', ' ')}")
        print()
    
    def display_trading_signal(self, signal, reason, confidence):
        """Display trading signal"""
        signal_emoji = {
            'BUY': '🟢 BUY',
            'SELL': '🔴 SELL',
            'HOLD': '⚪ HOLD'
        }
        
        confidence_bar = "█" * (confidence // 10) + "░" * (10 - confidence // 10)
        
        print("🎯 TRADING SIGNAL")
        print("-" * 50)
        print(f"Signal:     {signal_emoji.get(signal, signal)}")
        print(f"Confidence: {confidence:.0f}% [{confidence_bar}]")
        print(f"Reason:     {reason}")
        print()
    
    def display_portfolio(self, current_price):
        """Display portfolio status"""
        portfolio_value = self.get_portfolio_value(current_price)
        total_return = ((portfolio_value - self.initial_balance) / self.initial_balance) * 100
        
        print("💼 PORTFOLIO STATUS")
        print("-" * 50)
        print(f"💰 Cash Balance:      ${self.balance:,.2f}")
        print(f"₿  Bitcoin Held:      {self.btc_held:.6f} BTC")
        print(f"💎 BTC Value:         ${self.btc_held * current_price:,.2f}")
        print(f"📊 Total Value:       ${portfolio_value:,.2f}")
        print(f"📈 Total Return:      {total_return:+.2f}%")
        print(f"🔄 Total Trades:      {len(self.trades)}")
        print()
    
    def display_recent_trades(self):
        """Display recent trades"""
        if not self.trades:
            return
        
        print("📋 RECENT TRADES")
        print("-" * 50)
        recent_trades = self.trades[-5:]  # Last 5 trades
        
        for trade in recent_trades:
            action_emoji = "🟢" if trade['action'] == 'BUY' else "🔴"
            time_str = trade['timestamp'].strftime('%H:%M:%S')
            print(f"{action_emoji} {time_str} {trade['action']} {trade['amount']:.6f} BTC @ ${trade['price']:.2f}")
        print()
    
    def run_live_terminal(self):
        """Main live terminal loop"""
        print("🚀 Starting Live Bitcoin Trading Terminal...")
        print("Press Ctrl+C to stop")
        print()
        
        # Set up signal handler for graceful exit
        def signal_handler(sig, frame):
            self.running = False
            print("\n\n🛑 Stopping trading terminal...")
            sys.exit(0)
        
        import signal as sig_module
        sig_module.signal(sig_module.SIGINT, signal_handler)
        
        while self.running:
            try:
                # Clear screen for fresh display
                self.clear_screen()
                
                # Display header
                self.display_header()
                
                # Fetch live data
                print("📡 Fetching live market data...")
                data = self.get_live_data()
                
                if data is not None and not data.empty:
                    # Analyze market
                    market_info = self.analyze_market_condition(data)
                    
                    # Generate trading signal
                    signal, reason, confidence = self.generate_trading_signal(data, market_info)
                    
                    # Display all information
                    self.display_market_data(data, market_info)
                    self.display_trading_signal(signal, reason, confidence)
                    
                    current_price = market_info[1]['price']
                    self.display_portfolio(current_price)
                    
                    # Execute trade if signal is strong
                    if signal in ['BUY', 'SELL']:
                        executed, message = self.execute_trade(signal, current_price, confidence)
                        if executed:
                            print(f"⚡ TRADE EXECUTED: {message}")
                            print()
                    
                    self.display_recent_trades()
                    
                    # Store price history
                    self.price_history.append(current_price)
                    self.signal_history.append((signal, confidence))
                    
                else:
                    print("❌ Unable to fetch market data")
                
                # Display next update countdown
                print(f"🔄 Next update in {self.refresh_rate} seconds... (Ctrl+C to stop)")
                
                # Wait for next update
                time.sleep(self.refresh_rate)
                
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as e:
                print(f"❌ Error in main loop: {e}")
                time.sleep(5)
        
        print("\n🛑 Trading terminal stopped.")

def main():
    """Main function"""
    print("🚀 Bitcoin Live Trading Terminal")
    print("=" * 50)
    
    # Get initial balance from user
    try:
        balance_input = input("💰 Enter initial balance (default $10,000): $").strip()
        initial_balance = float(balance_input) if balance_input else 10000
    except ValueError:
        initial_balance = 10000
    
    print(f"✅ Starting with ${initial_balance:,.2f}")
    print()
    
    # Create and run terminal trader
    trader = TerminalTrader(initial_balance)
    trader.run_live_terminal()

if __name__ == "__main__":
    main()
