"""
Simplified Bitcoin Trading Environment
Lightweight trading environment without OpenAI Gym dependency
"""

import numpy as np
import pandas as pd
import yaml
from typing import Tuple, Dict, Any

class SimpleTradingEnv:
    """
    Simplified Trading Environment for Bitcoin
    
    Actions:
    0: Hold
    1: Buy
    2: Sell
    """
    
    def __init__(self, data: pd.DataFrame, config_path: str = "config.yaml"):
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.data = data.copy()
        self.initial_balance = self.config['trading']['initial_balance']
        self.transaction_cost = self.config['trading']['transaction_cost']
        self.lookback_window = self.config['features']['lookback_window']
        
        # Environment parameters
        self.current_step = 0
        self.max_steps = len(data) - self.lookback_window - 1
        
        # Portfolio state
        self.balance = self.initial_balance
        self.btc_held = 0
        self.total_trades = 0
        self.successful_trades = 0
        
        # Performance tracking
        self.portfolio_values = []
        self.trades_history = []
        self.rewards_history = []
        
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state"""
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.btc_held = 0
        self.total_trades = 0
        self.successful_trades = 0
        
        self.portfolio_values = [self.initial_balance]
        self.trades_history = []
        self.rewards_history = []
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Execute one time step within the environment"""
        prev_portfolio_value = self._get_portfolio_value()
        
        # Execute action
        self._execute_action(action)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        current_portfolio_value = self._get_portfolio_value()
        reward = self._calculate_reward(prev_portfolio_value, current_portfolio_value, action)
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Update tracking
        self.portfolio_values.append(current_portfolio_value)
        self.rewards_history.append(reward)
        
        # Additional info
        info = {
            'portfolio_value': current_portfolio_value,
            'balance': self.balance,
            'btc_held': self.btc_held,
            'total_trades': self.total_trades,
            'win_rate': self.successful_trades / max(1, self.total_trades),
            'current_price': self._get_current_price()
        }
        
        return self._get_observation(), reward, done, info
    
    def _execute_action(self, action: int):
        """Execute the trading action"""
        current_price = self._get_current_price()
        
        if action == 1:  # Buy
            if self.balance > 100:  # Minimum balance to trade
                # Use 90% of available balance
                amount_to_invest = self.balance * 0.9
                btc_to_buy = amount_to_invest / current_price
                transaction_cost = amount_to_invest * self.transaction_cost
                
                if self.balance >= (amount_to_invest + transaction_cost):
                    self.btc_held += btc_to_buy
                    self.balance -= (amount_to_invest + transaction_cost)
                    self.total_trades += 1
                    
                    self.trades_history.append({
                        'step': self.current_step,
                        'action': 'BUY',
                        'price': current_price,
                        'amount': btc_to_buy,
                        'cost': transaction_cost
                    })
        
        elif action == 2:  # Sell
            if self.btc_held > 0:
                # Sell all BTC
                sell_value = self.btc_held * current_price
                transaction_cost = sell_value * self.transaction_cost
                
                self.balance += (sell_value - transaction_cost)
                
                # Check if trade was profitable
                if len(self.trades_history) > 0:
                    last_buy = None
                    for trade in reversed(self.trades_history):
                        if trade['action'] == 'BUY':
                            last_buy = trade
                            break
                    
                    if last_buy and current_price > last_buy['price']:
                        self.successful_trades += 1
                
                self.trades_history.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'price': current_price,
                    'amount': self.btc_held,
                    'cost': transaction_cost
                })
                
                self.btc_held = 0
                self.total_trades += 1
        
        # Action 0 (Hold) requires no execution
    
    def _calculate_reward(self, prev_value: float, current_value: float, action: int) -> float:
        """Calculate reward based on portfolio performance"""
        # Portfolio return
        if prev_value > 0:
            portfolio_return = (current_value - prev_value) / prev_value
        else:
            portfolio_return = 0
        
        # Base reward
        reward = portfolio_return * 100  # Scale up the reward
        
        # Penalty for excessive trading
        if action != 0:
            reward -= 0.1
        
        # Bonus for profitable trades
        if action == 2 and len(self.trades_history) >= 2:
            last_trade = self.trades_history[-1]
            if last_trade['action'] == 'SELL':
                # Find corresponding buy
                for trade in reversed(self.trades_history[:-1]):
                    if trade['action'] == 'BUY':
                        if last_trade['price'] > trade['price']:
                            reward += 1.0  # Bonus for profitable trade
                        break
        
        return reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        if self.current_step >= len(self.data):
            self.current_step = len(self.data) - 1
        
        current_row = self.data.iloc[self.current_step]
        
        # Basic features
        features = []
        
        # Price features (normalized)
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in current_row:
                features.append(current_row[col] / 50000.0)  # Normalize by typical BTC price
        
        # Volume (normalized)
        if 'volume' in current_row:
            features.append(current_row['volume'] / 1e9)  # Normalize volume
        
        # Technical indicators (if available)
        tech_indicators = ['RSI', 'MACD', 'SMA_20', 'EMA_12']
        for indicator in tech_indicators:
            if indicator in current_row and not pd.isna(current_row[indicator]):
                if indicator == 'RSI':
                    features.append(current_row[indicator] / 100.0)  # RSI is 0-100
                else:
                    features.append(current_row[indicator] / 50000.0)  # Normalize by price
            else:
                features.append(0.0)
        
        # Fear & Greed Index
        if 'Fear & Greed Index' in current_row and not pd.isna(current_row['Fear & Greed Index']):
            features.append(current_row['Fear & Greed Index'] / 100.0)
        else:
            features.append(0.5)  # Neutral
        
        # Portfolio features
        total_value = self._get_portfolio_value()
        features.extend([
            self.balance / self.initial_balance,  # Normalized balance
            (self.btc_held * self._get_current_price()) / self.initial_balance,  # Normalized BTC value
            total_value / self.initial_balance,  # Normalized total value
            min(self.total_trades / 100.0, 1.0),  # Normalized trade count
        ])
        
        return np.array(features, dtype=np.float32)
    
    def _get_current_price(self) -> float:
        """Get current Bitcoin price"""
        if self.current_step >= len(self.data):
            return self.data.iloc[-1]['close']
        return self.data.iloc[self.current_step]['close']
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        return self.balance + (self.btc_held * self._get_current_price())
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if len(self.portfolio_values) < 2:
            return {}
        
        returns = np.diff(self.portfolio_values) / np.array(self.portfolio_values[:-1])
        returns = returns[~np.isnan(returns)]  # Remove NaN values
        
        total_return = (self.portfolio_values[-1] - self.portfolio_values[0]) / self.portfolio_values[0]
        
        if len(returns) > 0:
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        max_drawdown = self._calculate_max_drawdown()
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'win_rate': self.successful_trades / max(1, self.total_trades),
            'final_portfolio_value': self.portfolio_values[-1]
        }
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        if len(self.portfolio_values) < 2:
            return 0
        
        peak = np.maximum.accumulate(self.portfolio_values)
        drawdown = (np.array(self.portfolio_values) - peak) / peak
        return np.min(drawdown)
