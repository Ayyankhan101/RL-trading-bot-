"""
Bitcoin Trading Environment for Reinforcement Learning
Custom OpenAI Gym environment for cryptocurrency trading
"""

import gym
from gym import spaces
import numpy as np
import pandas as pd
import yaml
from typing import Tuple, Dict, Any

class BitcoinTradingEnv(gym.Env):
    """
    Custom Trading Environment for Bitcoin
    
    Actions:
    0: Hold
    1: Buy
    2: Sell
    
    State Space:
    - OHLCV data
    - Technical indicators
    - Fear & Greed Index
    - Portfolio information
    """
    
    def __init__(self, data: pd.DataFrame, config_path: str = "config.yaml"):
        super(BitcoinTradingEnv, self).__init__()
        
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
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(3)  # Hold, Buy, Sell
        
        # State space: OHLCV + indicators + portfolio info
        n_features = len(self._get_observation())
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        
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
            if self.balance > 0:
                # Calculate position size (use all available balance)
                btc_to_buy = (self.balance * 0.95) / current_price  # Keep 5% as buffer
                transaction_cost = btc_to_buy * current_price * self.transaction_cost
                
                if self.balance >= (btc_to_buy * current_price + transaction_cost):
                    self.btc_held += btc_to_buy
                    self.balance -= (btc_to_buy * current_price + transaction_cost)
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
                self.btc_held = 0
                self.total_trades += 1
                
                # Check if trade was profitable
                if len(self.trades_history) > 0 and self.trades_history[-1]['action'] == 'BUY':
                    if current_price > self.trades_history[-1]['price']:
                        self.successful_trades += 1
                
                self.trades_history.append({
                    'step': self.current_step,
                    'action': 'SELL',
                    'price': current_price,
                    'amount': self.btc_held,
                    'cost': transaction_cost
                })
        
        # Action 0 (Hold) requires no execution
    
    def _calculate_reward(self, prev_value: float, current_value: float, action: int) -> float:
        """Calculate reward based on portfolio performance and risk"""
        # Base reward: portfolio value change
        portfolio_return = (current_value - prev_value) / prev_value
        
        # Risk-adjusted reward using Sharpe-like ratio
        if len(self.portfolio_values) > 30:  # Need enough history
            returns = np.diff(self.portfolio_values[-30:]) / self.portfolio_values[-31:-1]
            volatility = np.std(returns) if np.std(returns) > 0 else 0.01
            risk_adjusted_return = portfolio_return / volatility
        else:
            risk_adjusted_return = portfolio_return
        
        # Penalty for excessive trading
        trading_penalty = -0.001 if action != 0 else 0
        
        # Bonus for holding during favorable conditions
        current_price = self._get_current_price()
        fear_greed = self._get_fear_greed_index()
        
        # Reward buying during fear and selling during greed
        sentiment_bonus = 0
        if action == 1 and fear_greed < 30:  # Buy during fear
            sentiment_bonus = 0.005
        elif action == 2 and fear_greed > 70:  # Sell during greed
            sentiment_bonus = 0.005
        
        total_reward = risk_adjusted_return + trading_penalty + sentiment_bonus
        
        return total_reward
    
    def _get_observation(self) -> np.ndarray:
        """Get current state observation"""
        # Get current data window
        start_idx = self.current_step - self.lookback_window
        end_idx = self.current_step + 1
        
        window_data = self.data.iloc[start_idx:end_idx]
        
        # OHLCV features (normalized)
        ohlcv_features = []
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in window_data.columns:
                values = window_data[col].values
                normalized = (values - values.mean()) / (values.std() + 1e-8)
                ohlcv_features.extend(normalized)
        
        # Technical indicators
        tech_features = []
        current_row = window_data.iloc[-1]
        
        # Add technical indicators if they exist
        indicator_cols = ['RSI', 'MACD', 'BB_upper', 'BB_lower', 'SMA_20', 'EMA_12']
        for col in indicator_cols:
            if col in current_row:
                tech_features.append(current_row[col])
        
        # Fear & Greed Index
        fear_greed = self._get_fear_greed_index()
        tech_features.append(fear_greed / 100.0)  # Normalize to 0-1
        
        # Portfolio features
        portfolio_features = [
            self.balance / self.initial_balance,  # Normalized balance
            self.btc_held * self._get_current_price() / self.initial_balance,  # Normalized BTC value
            self._get_portfolio_value() / self.initial_balance,  # Normalized total value
            self.total_trades / 100.0,  # Normalized trade count
        ]
        
        # Combine all features
        observation = np.concatenate([
            np.array(ohlcv_features),
            np.array(tech_features),
            np.array(portfolio_features)
        ])
        
        return observation.astype(np.float32)
    
    def _get_current_price(self) -> float:
        """Get current Bitcoin price"""
        return self.data.iloc[self.current_step]['close']
    
    def _get_fear_greed_index(self) -> float:
        """Get current Fear & Greed Index"""
        if 'Fear & Greed Index' in self.data.columns:
            value = self.data.iloc[self.current_step]['Fear & Greed Index']
            return value if not pd.isna(value) else 50.0  # Default to neutral
        return 50.0
    
    def _get_portfolio_value(self) -> float:
        """Calculate total portfolio value"""
        return self.balance + (self.btc_held * self._get_current_price())
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics"""
        if len(self.portfolio_values) < 2:
            return {}
        
        returns = np.diff(self.portfolio_values) / self.portfolio_values[:-1]
        
        total_return = (self.portfolio_values[-1] - self.portfolio_values[0]) / self.portfolio_values[0]
        sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
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
        peak = np.maximum.accumulate(self.portfolio_values)
        drawdown = (np.array(self.portfolio_values) - peak) / peak
        return np.min(drawdown)
