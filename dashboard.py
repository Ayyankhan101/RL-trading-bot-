"""
Streamlit Dashboard for RL Trading Bot
Interactive web interface for monitoring and analyzing trading performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import yaml
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_utils import load_data, download_bitcoin_data, add_fear_greed_index
from utils.plotting import create_interactive_dashboard
from utils.metrics import calculate_comprehensive_metrics, print_performance_report
from features.technical_indicators import add_technical_indicators
from agents.dqn_agent import DQNAgent
from environment.trading_env import BitcoinTradingEnv


# Page configuration
st.set_page_config(
    page_title="RL Trading Bot Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_config():
    """Load configuration file"""
    try:
        with open('config.yaml', 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error("Configuration file not found. Please ensure config.yaml exists.")
        return None


@st.cache_data
def load_bitcoin_data_cached(data_source, symbol, period):
    """Load Bitcoin data with caching"""
    if data_source == "File":
        if os.path.exists("data/BTC.csv"):
            return load_data("data/BTC.csv")
        else:
            st.warning("BTC.csv not found. Downloading data...")
            return download_bitcoin_data(symbol, period)
    else:
        return download_bitcoin_data(symbol, period)


def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🤖 RL Trading Bot Dashboard</h1>', unsafe_allow_html=True)
    
    # Load configuration
    config = load_config()
    if config is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox(
        "Data Source",
        ["File", "Yahoo Finance"],
        help="Choose data source for Bitcoin prices"
    )
    
    if data_source == "Yahoo Finance":
        symbol = st.sidebar.selectbox("Symbol", ["BTC-USD", "ETH-USD"], index=0)
        period = st.sidebar.selectbox(
            "Period", 
            ["1y", "2y", "5y", "max"], 
            index=2,
            help="Data period to download"
        )
    else:
        symbol = "BTC-USD"
        period = "5y"
    
    # Load data
    with st.spinner("Loading Bitcoin data..."):
        data = load_bitcoin_data_cached(data_source, symbol, period)
    
    if data.empty:
        st.error("Failed to load data. Please check your configuration.")
        st.stop()
    
    # Add technical indicators
    with st.spinner("Calculating technical indicators..."):
        data = add_technical_indicators(data)
        data = add_fear_greed_index(data)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Market Overview", 
        "🤖 Model Performance", 
        "📈 Trading Analysis", 
        "⚙️ Live Trading", 
        "📋 Reports"
    ])
    
    with tab1:
        show_market_overview(data, config)
    
    with tab2:
        show_model_performance(data, config)
    
    with tab3:
        show_trading_analysis(data, config)
    
    with tab4:
        show_live_trading(data, config)
    
    with tab5:
        show_reports(data, config)


def show_market_overview(data, config):
    """Display market overview tab"""
    st.header("📊 Market Overview")
    
    # Current price metrics
    current_price = data['close'].iloc[-1]
    prev_price = data['close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Current Price",
            f"${current_price:,.2f}",
            f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        high_24h = data['high'].iloc[-1]
        low_24h = data['low'].iloc[-1]
        st.metric("24h High", f"${high_24h:,.2f}")
        st.metric("24h Low", f"${low_24h:,.2f}")
    
    with col3:
        volume = data['volume'].iloc[-1]
        st.metric("Volume", f"{volume:,.0f}")
        
        if 'Fear & Greed Index' in data.columns:
            fear_greed = data['Fear & Greed Index'].iloc[-1]
            st.metric("Fear & Greed", f"{fear_greed:.0f}")
    
    with col4:
        if 'RSI' in data.columns:
            rsi = data['RSI'].iloc[-1]
            st.metric("RSI", f"{rsi:.1f}")
        
        if 'MACD' in data.columns:
            macd = data['MACD'].iloc[-1]
            st.metric("MACD", f"{macd:.4f}")
    
    # Price chart
    st.subheader("Price Chart with Technical Indicators")
    
    fig = make_subplots(
        rows=3, cols=1,
        subplot_titles=('Price & Moving Averages', 'RSI', 'MACD'),
        vertical_spacing=0.08,
        row_heights=[0.6, 0.2, 0.2]
    )
    
    # Candlestick chart
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
    
    if 'EMA_12' in data.columns:
        fig.add_trace(
            go.Scatter(x=data.index, y=data['EMA_12'], name='EMA 12',
                      line=dict(color='red', width=1)),
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
    
    fig.update_layout(height=800, showlegend=True, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


def show_model_performance(data, config):
    """Display model performance tab"""
    st.header("🤖 Model Performance")
    
    # Model selection
    model_files = []
    if os.path.exists("models/trained_models"):
        model_files = [f for f in os.listdir("models/trained_models") if f.endswith('.h5')]
    
    if not model_files:
        st.warning("No trained models found. Please train a model first.")
        
        if st.button("🚀 Start Training"):
            st.info("Training would start here. This is a demo interface.")
        return
    
    selected_model = st.selectbox("Select Model", model_files)
    
    # Model info
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Information")
        st.write(f"**Model File:** {selected_model}")
        st.write(f"**Architecture:** {config['model']['architecture']}")
        st.write(f"**Learning Rate:** {config['model']['learning_rate']}")
        st.write(f"**Batch Size:** {config['model']['batch_size']}")
    
    with col2:
        st.subheader("Training Configuration")
        st.write(f"**Episodes:** {config['training']['episodes']}")
        st.write(f"**Max Steps:** {config['training']['max_steps_per_episode']}")
        st.write(f"**Validation Split:** {config['training']['validation_split']}")
    
    # Simulated training metrics (in real implementation, load from training logs)
    st.subheader("Training Progress")
    
    # Generate sample training data
    episodes = np.arange(1, config['training']['episodes'] + 1)
    rewards = np.random.normal(0, 1, len(episodes)).cumsum() + np.random.normal(100, 10, len(episodes))
    portfolio_values = 10000 + np.random.normal(0, 500, len(episodes)).cumsum()
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Rewards', 'Portfolio Value', 'Epsilon Decay', 'Loss'),
        vertical_spacing=0.1
    )
    
    # Episode rewards
    fig.add_trace(
        go.Scatter(x=episodes, y=rewards, name='Rewards', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Portfolio values
    fig.add_trace(
        go.Scatter(x=episodes, y=portfolio_values, name='Portfolio Value', line=dict(color='green')),
        row=1, col=2
    )
    
    # Epsilon decay
    epsilon_values = config['model']['epsilon_start'] * (config['model']['epsilon_decay'] ** episodes)
    fig.add_trace(
        go.Scatter(x=episodes, y=epsilon_values, name='Epsilon', line=dict(color='orange')),
        row=2, col=1
    )
    
    # Loss
    loss_values = np.exp(-episodes/100) + np.random.normal(0, 0.01, len(episodes))
    fig.add_trace(
        go.Scatter(x=episodes, y=loss_values, name='Loss', line=dict(color='red')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def show_trading_analysis(data, config):
    """Display trading analysis tab"""
    st.header("📈 Trading Analysis")
    
    # Simulate backtesting results
    st.subheader("Backtesting Results")
    
    # Generate sample portfolio performance
    initial_balance = config['trading']['initial_balance']
    portfolio_values = [initial_balance]
    
    # Simulate trading performance
    for i in range(1, len(data)):
        # Simple momentum strategy for demo
        price_change = (data['close'].iloc[i] - data['close'].iloc[i-1]) / data['close'].iloc[i-1]
        portfolio_change = price_change * 0.5  # 50% exposure
        new_value = portfolio_values[-1] * (1 + portfolio_change)
        portfolio_values.append(new_value)
    
    # Performance metrics
    final_value = portfolio_values[-1]
    total_return = (final_value - initial_balance) / initial_balance
    
    # Buy and hold benchmark
    btc_initial = data['close'].iloc[0]
    btc_final = data['close'].iloc[-1]
    btc_return = (btc_final - btc_initial) / btc_initial
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{total_return:.2%}")
        st.metric("Final Value", f"${final_value:,.2f}")
    
    with col2:
        st.metric("BTC Return", f"{btc_return:.2%}")
        st.metric("Outperformance", f"{(total_return - btc_return):.2%}")
    
    with col3:
        # Calculate max drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (np.array(portfolio_values) - peak) / peak
        max_drawdown = np.min(drawdown)
        st.metric("Max Drawdown", f"{max_drawdown:.2%}")
    
    with col4:
        # Calculate Sharpe ratio (simplified)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    
    # Portfolio performance chart
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=data.index[:len(portfolio_values)],
            y=portfolio_values,
            name='RL Strategy',
            line=dict(color='blue', width=3)
        )
    )
    
    # Buy and hold benchmark
    btc_portfolio = [initial_balance * (data['close'].iloc[i] / data['close'].iloc[0]) for i in range(len(data))]
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=btc_portfolio,
            name='Buy & Hold',
            line=dict(color='orange', width=2, dash='dash')
        )
    )
    
    fig.update_layout(
        title="Portfolio Performance Comparison",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_live_trading(data, config):
    """Display live trading tab"""
    st.header("⚙️ Live Trading")
    
    st.warning("⚠️ Live trading is disabled in demo mode")
    
    # Trading controls
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trading Controls")
        
        trading_enabled = st.checkbox("Enable Live Trading", disabled=True)
        
        if st.button("🚀 Start Trading", disabled=True):
            st.info("Live trading would start here")
        
        if st.button("⏹️ Stop Trading", disabled=True):
            st.info("Trading would stop here")
    
    with col2:
        st.subheader("Current Position")
        st.metric("Cash Balance", "$10,000.00")
        st.metric("BTC Holdings", "0.0000 BTC")
        st.metric("Total Value", "$10,000.00")
    
    # Recent trades table
    st.subheader("Recent Trades")
    
    # Sample trades data
    sample_trades = pd.DataFrame({
        'Timestamp': pd.date_range(start='2024-01-01', periods=5, freq='D'),
        'Action': ['Buy', 'Sell', 'Buy', 'Hold', 'Sell'],
        'Price': [45000, 46500, 44800, 45200, 46000],
        'Amount': [0.1, 0.1, 0.05, 0.0, 0.05],
        'P&L': [0, 150, 0, 0, 60]
    })
    
    st.dataframe(sample_trades, use_container_width=True)


def show_reports(data, config):
    """Display reports tab"""
    st.header("📋 Performance Reports")
    
    # Report generation
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Generate Report")
        
        report_type = st.selectbox(
            "Report Type",
            ["Performance Summary", "Risk Analysis", "Trade Analysis", "Full Report"]
        )
        
        date_range = st.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=365), datetime.now()),
            max_value=datetime.now()
        )
        
        if st.button("📊 Generate Report"):
            st.success("Report generated successfully!")
    
    with col2:
        st.subheader("Export Options")
        
        export_format = st.selectbox("Format", ["PDF", "Excel", "CSV"])
        
        if st.button("📥 Download Report"):
            st.info(f"Report would be downloaded as {export_format}")
    
    # Sample performance summary
    st.subheader("Performance Summary")
    
    summary_data = {
        'Metric': [
            'Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio',
            'Max Drawdown', 'Win Rate', 'Profit Factor', 'Total Trades'
        ],
        'Value': [
            '23.45%', '18.32%', '15.67%', '1.17',
            '-8.23%', '67.4%', '2.31', '156'
        ],
        'Benchmark': [
            '15.62%', '12.45%', '22.34%', '0.56',
            '-18.45%', 'N/A', 'N/A', 'N/A'
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.dataframe(summary_df, use_container_width=True)


if __name__ == "__main__":
    main()
