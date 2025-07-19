"""
Technical Indicators for Bitcoin Trading
Implements various technical analysis indicators
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive technical indicators to the dataframe
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with added technical indicators
    """
    data = df.copy()
    
    # Simple Moving Averages
    data['SMA_10'] = data['close'].rolling(window=10).mean()
    data['SMA_20'] = data['close'].rolling(window=20).mean()
    data['SMA_50'] = data['close'].rolling(window=50).mean()
    
    # Exponential Moving Averages
    data['EMA_12'] = data['close'].ewm(span=12).mean()
    data['EMA_26'] = data['close'].ewm(span=26).mean()
    
    # RSI (Relative Strength Index)
    data['RSI'] = calculate_rsi(data['close'])
    
    # MACD
    macd_data = calculate_macd(data['close'])
    data['MACD'] = macd_data['MACD']
    data['MACD_signal'] = macd_data['Signal']
    data['MACD_histogram'] = macd_data['Histogram']
    
    # Bollinger Bands
    bb_data = calculate_bollinger_bands(data['close'])
    data['BB_upper'] = bb_data['Upper']
    data['BB_middle'] = bb_data['Middle']
    data['BB_lower'] = bb_data['Lower']
    data['BB_width'] = bb_data['Width']
    
    # Stochastic Oscillator
    stoch_data = calculate_stochastic(data['high'], data['low'], data['close'])
    data['STOCH_k'] = stoch_data['%K']
    data['STOCH_d'] = stoch_data['%D']
    
    # Average True Range (ATR)
    data['ATR'] = calculate_atr(data['high'], data['low'], data['close'])
    
    # Volume indicators
    data['Volume_SMA'] = data['volume'].rolling(window=20).mean()
    data['Volume_ratio'] = data['volume'] / data['Volume_SMA']
    
    # Price momentum
    data['Price_momentum_5'] = data['close'].pct_change(5)
    data['Price_momentum_10'] = data['close'].pct_change(10)
    
    # Volatility
    data['Volatility'] = data['close'].rolling(window=20).std()
    
    return data

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
    """Calculate MACD indicator"""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    
    return {
        'MACD': macd,
        'Signal': signal_line,
        'Histogram': histogram
    }

def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
    """Calculate Bollinger Bands"""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    width = upper - lower
    
    return {
        'Upper': upper,
        'Middle': sma,
        'Lower': lower,
        'Width': width
    }

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                        k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
    """Calculate Stochastic Oscillator"""
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_window).mean()
    
    return {
        '%K': k_percent,
        '%D': d_percent
    }

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Average True Range"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=window).mean()
    
    return atr
