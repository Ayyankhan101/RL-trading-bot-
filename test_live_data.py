#!/usr/bin/env python3
"""
Test script to verify live Bitcoin data fetching
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_utils import download_bitcoin_data
from datetime import datetime, timedelta
import pandas as pd

def test_live_data_fetch():
    """Test fetching the most recent Bitcoin data"""
    print("🧪 Testing Live Bitcoin Data Fetching")
    print("=" * 50)
    
    # Test different periods to ensure we get the most recent data
    periods = ['1d', '7d', '1mo', '3mo']
    
    for period in periods:
        print(f"\n📊 Testing {period} period...")
        try:
            data = download_bitcoin_data('BTC-USD', period, '1d')
            
            if not data.empty:
                print(f"✅ Successfully downloaded {len(data)} rows")
                print(f"📅 Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
                print(f"💰 Latest price: ${data['close'].iloc[-1]:,.2f}")
                print(f"📈 Price change: {((data['close'].iloc[-1] / data['close'].iloc[0]) - 1) * 100:.2f}%")
                
                # Check if data is recent (within last few days)
                latest_date = pd.to_datetime(data['timestamp'].max())
                days_old = (datetime.now() - latest_date).days
                print(f"🕐 Data freshness: {days_old} days old")
                
                if days_old <= 3:
                    print("✅ Data is fresh and recent!")
                else:
                    print("⚠️  Data might be outdated")
                    
                return data
            else:
                print("❌ No data received")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    return None

def test_current_bitcoin_price():
    """Get the most current Bitcoin price available"""
    print("\n🔴 CURRENT BITCOIN MARKET DATA")
    print("=" * 40)
    
    try:
        # Get very recent data (last 5 days with hourly intervals for freshness)
        data = download_bitcoin_data('BTC-USD', '5d', '1h')
        
        if not data.empty:
            latest_price = data['close'].iloc[-1]
            prev_price = data['close'].iloc[-25] if len(data) > 25 else data['close'].iloc[0]  # ~24h ago
            change_24h = ((latest_price / prev_price) - 1) * 100
            
            print(f"💰 Current BTC Price: ${latest_price:,.2f}")
            print(f"📈 24h Change: {change_24h:+.2f}%")
            print(f"📅 Last Update: {data['timestamp'].iloc[-1]}")
            print(f"📊 Volume: {data['volume'].iloc[-1]:,.0f}")
            print(f"🔺 High: ${data['high'].max():,.2f}")
            print(f"🔻 Low: ${data['low'].min():,.2f}")
            
            return data
        else:
            print("❌ Could not fetch current price data")
            
    except Exception as e:
        print(f"❌ Error fetching current price: {e}")
    
    return None

if __name__ == "__main__":
    # Test live data fetching
    data = test_live_data_fetch()
    
    # Get current Bitcoin price
    current_data = test_current_bitcoin_price()
    
    if data is not None:
        print("\n🎉 Live data fetching is working!")
        print("✅ Ready to run trading bot with real Bitcoin data")
    else:
        print("\n❌ Live data fetching needs debugging")
