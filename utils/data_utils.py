"""
Data handling utilities for the RL Trading Bot
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Tuple
import os


def download_bitcoin_data(symbol: str = "BTC-USD", 
                         period: str = "5y", 
                         interval: str = "1d") -> pd.DataFrame:
    """
    Download Bitcoin data from Yahoo Finance
    
    Args:
        symbol: Trading symbol (default: BTC-USD)
        period: Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period, interval=interval)
        
        # Clean column names
        data.columns = [col.lower() for col in data.columns]
        
        # Reset index to make timestamp a column
        data.reset_index(inplace=True)
        # Handle both 'Date' and 'date' column names from yfinance
        if 'Date' in data.columns:
            data.rename(columns={'Date': 'timestamp'}, inplace=True)
        elif 'date' in data.columns:
            data.rename(columns={'date': 'timestamp'}, inplace=True)
        
        print(f"Downloaded {len(data)} rows of {symbol} data")
        return data
        
    except Exception as e:
        print(f"Error downloading data: {e}")
        return pd.DataFrame()


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from CSV file
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        data = pd.read_csv(file_path)
        
        # Convert timestamp column if it exists
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        elif 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data.rename(columns={'date': 'timestamp'}, inplace=True)
            
        print(f"Loaded {len(data)} rows from {file_path}")
        return data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()


def save_data(data: pd.DataFrame, file_path: str) -> bool:
    """
    Save DataFrame to CSV file
    
    Args:
        data: DataFrame to save
        file_path: Output file path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        data.to_csv(file_path, index=False)
        print(f"Saved {len(data)} rows to {file_path}")
        return True
        
    except Exception as e:
        print(f"Error saving data: {e}")
        return False


def split_data(data: pd.DataFrame, 
               train_ratio: float = 0.7, 
               val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets
    
    Args:
        data: Input DataFrame
        train_ratio: Ratio for training data
        val_ratio: Ratio for validation data
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    
    train_data = data.iloc[:train_end].copy()
    val_data = data.iloc[train_end:val_end].copy()
    test_data = data.iloc[val_end:].copy()
    
    print(f"Data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    return train_data, val_data, test_data


def normalize_features(data: pd.DataFrame, 
                      feature_columns: list, 
                      method: str = 'minmax') -> pd.DataFrame:
    """
    Normalize feature columns
    
    Args:
        data: Input DataFrame
        feature_columns: List of columns to normalize
        method: Normalization method ('minmax', 'zscore')
        
    Returns:
        DataFrame with normalized features
    """
    normalized_data = data.copy()
    
    for col in feature_columns:
        if col in data.columns:
            if method == 'minmax':
                min_val = data[col].min()
                max_val = data[col].max()
                normalized_data[col] = (data[col] - min_val) / (max_val - min_val + 1e-8)
            elif method == 'zscore':
                mean_val = data[col].mean()
                std_val = data[col].std()
                normalized_data[col] = (data[col] - mean_val) / (std_val + 1e-8)
    
    return normalized_data


def add_fear_greed_index(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add Fear & Greed Index (simulated for demo purposes)
    In production, this would fetch real data from an API
    
    Args:
        data: Input DataFrame
        
    Returns:
        DataFrame with Fear & Greed Index
    """
    # Simulate Fear & Greed Index based on price volatility and momentum
    data = data.copy()
    
    # Calculate price momentum and volatility
    data['price_change'] = data['close'].pct_change(periods=7)
    data['volatility'] = data['close'].rolling(window=14).std()
    
    # Normalize to 0-100 scale (inverted for fear/greed)
    price_momentum_norm = (data['price_change'] - data['price_change'].mean()) / data['price_change'].std()
    volatility_norm = (data['volatility'] - data['volatility'].mean()) / data['volatility'].std()
    
    # Combine factors (higher values = more greed, lower = more fear)
    fear_greed_raw = 50 + (price_momentum_norm * 20) - (volatility_norm * 15)
    
    # Clip to 0-100 range
    data['Fear & Greed Index'] = np.clip(fear_greed_raw, 0, 100)
    
    # Clean up temporary columns
    data.drop(['price_change', 'volatility'], axis=1, inplace=True)
    
    return data


def validate_data(data: pd.DataFrame) -> dict:
    """
    Validate data quality and return statistics
    
    Args:
        data: Input DataFrame
        
    Returns:
        Dictionary with validation results
    """
    validation_results = {
        'total_rows': len(data),
        'missing_values': data.isnull().sum().to_dict(),
        'duplicate_rows': data.duplicated().sum(),
        'date_range': None,
        'data_quality_score': 0
    }
    
    # Check date range
    if 'timestamp' in data.columns:
        validation_results['date_range'] = {
            'start': data['timestamp'].min(),
            'end': data['timestamp'].max(),
            'days': (data['timestamp'].max() - data['timestamp'].min()).days
        }
    
    # Calculate data quality score (0-100)
    total_cells = len(data) * len(data.columns)
    missing_cells = data.isnull().sum().sum()
    duplicate_penalty = validation_results['duplicate_rows'] * len(data.columns)
    
    quality_score = max(0, 100 - (missing_cells / total_cells * 100) - (duplicate_penalty / total_cells * 100))
    validation_results['data_quality_score'] = round(quality_score, 2)
    
    return validation_results


def create_sample_data() -> pd.DataFrame:
    """
    Create sample Bitcoin data for testing purposes
    
    Returns:
        DataFrame with sample OHLCV data
    """
    # Generate 2 years of daily data
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    n_days = len(dates)
    
    # Simulate Bitcoin price with trend and volatility
    np.random.seed(42)
    
    # Starting price
    initial_price = 40000
    
    # Generate price series with random walk + trend
    returns = np.random.normal(0.001, 0.03, n_days)  # Daily returns
    prices = [initial_price]
    
    for i in range(1, n_days):
        new_price = prices[-1] * (1 + returns[i])
        prices.append(max(new_price, 1000))  # Minimum price floor
    
    # Create OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
        'close': prices,
        'volume': np.random.lognormal(15, 1, n_days)  # Log-normal volume distribution
    })
    
    # Ensure high >= close >= low and high >= open >= low
    for i in range(len(data)):
        high = max(data.loc[i, 'open'], data.loc[i, 'close'], data.loc[i, 'high'])
        low = min(data.loc[i, 'open'], data.loc[i, 'close'], data.loc[i, 'low'])
        data.loc[i, 'high'] = high
        data.loc[i, 'low'] = low
    
    return data


if __name__ == "__main__":
    # Example usage
    print("Creating sample Bitcoin data...")
    sample_data = create_sample_data()
    
    print("\nValidating data...")
    validation = validate_data(sample_data)
    print(f"Data quality score: {validation['data_quality_score']}")
    
    print("\nSaving sample data...")
    save_data(sample_data, "data/sample_btc_data.csv")
