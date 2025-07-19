"""
Backtesting Script for RL Trading Bot
Comprehensive backtesting framework with performance analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
import os
from datetime import datetime
from typing import Dict, List, Tuple

from agents.dqn_agent import DQNAgent
from environment.trading_env import BitcoinTradingEnv
from features.technical_indicators import add_technical_indicators
from utils.data_utils import load_data, split_data, add_fear_greed_index
from utils.metrics import calculate_comprehensive_metrics, print_performance_report
from utils.plotting import plot_backtest_results, plot_performance_metrics


class Backtester:
    """Comprehensive backtesting framework"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize backtester with configuration"""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.results = {}
        self.trades = []
        self.portfolio_values = []
        self.benchmark_values = []
        
    def load_model_and_data(self, model_path: str, data_path: str) -> Tuple[DQNAgent, pd.DataFrame]:
        """Load trained model and prepare data"""
        print(f"Loading data from {data_path}...")
        data = load_data(data_path)
        
        # Add technical indicators
        data = add_technical_indicators(data)
        data = add_fear_greed_index(data)
        data = data.dropna()
        
        print(f"Data loaded: {len(data)} rows")
        
        # Filter data by date range if specified
        if 'start_date' in self.config['backtesting']:
            start_date = pd.to_datetime(self.config['backtesting']['start_date'])
            data = data[data.index >= start_date]
        
        if 'end_date' in self.config['backtesting']:
            end_date = pd.to_datetime(self.config['backtesting']['end_date'])
            data = data[data.index <= end_date]
        
        print(f"Filtered data: {len(data)} rows")
        
        # Load trained model
        print(f"Loading model from {model_path}...")
        
        # Create environment to get state size
        temp_env = BitcoinTradingEnv(data.head(100), "config.yaml")
        state_size = len(temp_env._get_observation())
        action_size = temp_env.action_space.n
        
        # Initialize and load agent
        agent = DQNAgent(state_size, action_size, "config.yaml")
        agent.load_model(model_path)
        
        return agent, data
    
    def run_backtest(self, agent: DQNAgent, data: pd.DataFrame) -> Dict:
        """Run backtesting simulation"""
        print("Starting backtesting simulation...")
        
        # Create environment
        env = BitcoinTradingEnv(data, "config.yaml")
        
        # Initialize tracking variables
        state = env.reset()
        total_reward = 0
        step_count = 0
        
        self.portfolio_values = [env.initial_balance]
        self.trades = []
        
        # Run simulation
        while True:
            # Get action from agent (no exploration)
            action = agent.act(state, training=False)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Record trade if action was buy or sell
            if action != 0:  # Not hold
                trade = {
                    'timestamp': data.index[env.current_step],
                    'action': 'buy' if action == 1 else 'sell',
                    'price': info['current_price'],
                    'portfolio_value': info['portfolio_value'],
                    'balance': info['balance'],
                    'btc_held': info['btc_held']
                }
                self.trades.append(trade)
            
            # Update tracking
            self.portfolio_values.append(info['portfolio_value'])
            total_reward += reward
            step_count += 1
            
            state = next_state
            
            if done:
                break
        
        # Calculate trade P&L
        self._calculate_trade_pnl()
        
        # Calculate benchmark (buy and hold)
        self._calculate_benchmark(data)
        
        print(f"Backtesting completed: {step_count} steps, {len(self.trades)} trades")
        
        return {
            'total_reward': total_reward,
            'final_portfolio_value': self.portfolio_values[-1],
            'total_trades': len(self.trades),
            'steps': step_count
        }
    
    def _calculate_trade_pnl(self):
        """Calculate P&L for each trade"""
        if len(self.trades) < 2:
            return
        
        buy_price = None
        
        for i, trade in enumerate(self.trades):
            if trade['action'] == 'buy':
                buy_price = trade['price']
                trade['pnl'] = 0  # No P&L on entry
            elif trade['action'] == 'sell' and buy_price is not None:
                # Calculate P&L for sell trade
                pnl = (trade['price'] - buy_price) * trade['btc_held']
                trade['pnl'] = pnl
                buy_price = None  # Reset for next trade pair
            else:
                trade['pnl'] = 0
    
    def _calculate_benchmark(self, data: pd.DataFrame):
        """Calculate buy and hold benchmark"""
        initial_balance = self.config['trading']['initial_balance']
        initial_price = data['close'].iloc[0]
        
        # Buy and hold strategy
        btc_amount = initial_balance / initial_price
        
        self.benchmark_values = []
        for i in range(len(self.portfolio_values)):
            if i < len(data):
                current_price = data['close'].iloc[i]
                benchmark_value = btc_amount * current_price
                self.benchmark_values.append(benchmark_value)
            else:
                self.benchmark_values.append(self.benchmark_values[-1])
    
    def analyze_performance(self) -> Dict:
        """Analyze backtesting performance"""
        print("Analyzing performance...")
        
        # Calculate comprehensive metrics
        strategy_metrics = calculate_comprehensive_metrics(
            self.portfolio_values,
            self.trades,
            self.benchmark_values
        )
        
        # Additional analysis
        if len(self.trades) > 0:
            # Trade analysis
            winning_trades = [t for t in self.trades if t.get('pnl', 0) > 0]
            losing_trades = [t for t in self.trades if t.get('pnl', 0) < 0]
            
            strategy_metrics.update({
                'winning_trades': len(winning_trades),
                'losing_trades': len(losing_trades),
                'avg_winning_trade': np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0,
                'avg_losing_trade': np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0,
                'largest_win': max([t.get('pnl', 0) for t in self.trades]),
                'largest_loss': min([t.get('pnl', 0) for t in self.trades])
            })
        
        self.results = strategy_metrics
        return strategy_metrics
    
    def generate_report(self, save_path: str = None):
        """Generate comprehensive backtesting report"""
        print("\n" + "="*80)
        print("BACKTESTING REPORT")
        print("="*80)
        
        # Print performance metrics
        print_performance_report(self.results)
        
        # Trade summary
        if len(self.trades) > 0:
            print("\n💼 TRADE SUMMARY")
            print("-" * 40)
            print(f"Total Trades:        {len(self.trades):>10}")
            print(f"Winning Trades:      {self.results.get('winning_trades', 0):>10}")
            print(f"Losing Trades:       {self.results.get('losing_trades', 0):>10}")
            print(f"Avg Winning Trade:   ${self.results.get('avg_winning_trade', 0):>9.2f}")
            print(f"Avg Losing Trade:    ${self.results.get('avg_losing_trade', 0):>9.2f}")
            print(f"Largest Win:         ${self.results.get('largest_win', 0):>9.2f}")
            print(f"Largest Loss:        ${self.results.get('largest_loss', 0):>9.2f}")
        
        print("="*80)
        
        # Save report to file
        if save_path:
            self._save_report_to_file(save_path)
    
    def _save_report_to_file(self, file_path: str):
        """Save report to text file"""
        with open(file_path, 'w') as f:
            f.write("BACKTESTING REPORT\n")
            f.write("="*50 + "\n\n")
            
            f.write("PERFORMANCE METRICS\n")
            f.write("-"*30 + "\n")
            for key, value in self.results.items():
                if isinstance(value, float):
                    if 'return' in key.lower() or 'ratio' in key.lower():
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            
            f.write(f"\nReport generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"Report saved to: {file_path}")
    
    def plot_results(self, data: pd.DataFrame, save_path: str = None):
        """Plot backtesting results"""
        print("Generating plots...")
        
        plot_backtest_results(
            data.iloc[:len(self.portfolio_values)],
            self.trades,
            self.portfolio_values,
            self.benchmark_values,
            save_path
        )
        
        # Performance metrics comparison
        if self.benchmark_values:
            benchmark_metrics = calculate_comprehensive_metrics(self.benchmark_values)
            plot_performance_metrics(
                self.results,
                benchmark_metrics,
                save_path.replace('.png', '_metrics.png') if save_path else None
            )
    
    def export_trades(self, file_path: str):
        """Export trades to CSV"""
        if not self.trades:
            print("No trades to export")
            return
        
        trades_df = pd.DataFrame(self.trades)
        trades_df.to_csv(file_path, index=False)
        print(f"Trades exported to: {file_path}")
    
    def export_portfolio_values(self, file_path: str):
        """Export portfolio values to CSV"""
        portfolio_df = pd.DataFrame({
            'step': range(len(self.portfolio_values)),
            'portfolio_value': self.portfolio_values,
            'benchmark_value': self.benchmark_values[:len(self.portfolio_values)] if self.benchmark_values else None
        })
        portfolio_df.to_csv(file_path, index=False)
        print(f"Portfolio values exported to: {file_path}")


def main():
    """Main backtesting function"""
    parser = argparse.ArgumentParser(description='Backtest RL Trading Bot')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, required=True, help='Path to test data CSV')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--output', type=str, default='backtest_results', help='Output directory')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--export', action='store_true', help='Export detailed results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Initialize backtester
    backtester = Backtester(args.config)
    
    try:
        # Load model and data
        agent, data = backtester.load_model_and_data(args.model, args.data)
        
        # Run backtest
        summary = backtester.run_backtest(agent, data)
        
        # Analyze performance
        metrics = backtester.analyze_performance()
        
        # Generate report
        report_path = os.path.join(args.output, 'backtest_report.txt')
        backtester.generate_report(report_path)
        
        # Generate plots
        if args.plot:
            plot_path = os.path.join(args.output, 'backtest_results.png')
            backtester.plot_results(data, plot_path)
        
        # Export detailed results
        if args.export:
            trades_path = os.path.join(args.output, 'trades.csv')
            portfolio_path = os.path.join(args.output, 'portfolio_values.csv')
            
            backtester.export_trades(trades_path)
            backtester.export_portfolio_values(portfolio_path)
        
        print(f"\nBacktesting completed successfully!")
        print(f"Results saved to: {args.output}")
        
    except Exception as e:
        print(f"Error during backtesting: {e}")
        raise


if __name__ == "__main__":
    main()
