"""
Training Script for Bitcoin RL Trading Bot
Main script to train the DQN agent on Bitcoin data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from datetime import datetime
import os

from agents.dqn_agent import DQNAgent
from environment.trading_env import BitcoinTradingEnv
from features.technical_indicators import add_technical_indicators

def load_and_prepare_data(data_path: str) -> pd.DataFrame:
    """Load and prepare Bitcoin data with technical indicators"""
    print(f"Loading data from {data_path}...")
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Convert timestamp to datetime if needed
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
    
    # Add technical indicators
    print("Adding technical indicators...")
    df = add_technical_indicators(df)
    
    # Remove rows with NaN values
    df = df.dropna()
    
    print(f"Data prepared: {len(df)} rows, {len(df.columns)} columns")
    return df

def train_agent(data: pd.DataFrame, config_path: str = "config.yaml"):
    """Train the DQN agent"""
    
    # Load configuration
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Split data for training and validation
    split_idx = int(len(data) * (1 - config['training']['validation_split']))
    train_data = data.iloc[:split_idx]
    val_data = data.iloc[split_idx:]
    
    print(f"Training data: {len(train_data)} rows")
    print(f"Validation data: {len(val_data)} rows")
    
    # Create environment
    env = BitcoinTradingEnv(train_data, config_path)
    
    # Initialize agent
    state_size = len(env._get_observation())
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, config_path)
    
    print(f"State size: {state_size}")
    print(f"Action size: {action_size}")
    
    # Training parameters
    episodes = config['training']['episodes']
    target_update_frequency = config['model']['target_update_frequency']
    
    # Training metrics
    episode_rewards = []
    episode_portfolio_values = []
    episode_trades = []
    
    print(f"\nStarting training for {episodes} episodes...")
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # Choose action
            action = agent.act(state)
            
            # Execute action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Update target network periodically
        if episode % target_update_frequency == 0:
            agent.update_target_network()
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_portfolio_values.append(info['portfolio_value'])
        episode_trades.append(info['total_trades'])
        
        # Print progress
        if episode % config['training']['log_frequency'] == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_portfolio = np.mean(episode_portfolio_values[-10:])
            stats = agent.get_training_stats()
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {total_reward:8.4f} | "
                  f"Avg Reward: {avg_reward:8.4f} | "
                  f"Portfolio: ${avg_portfolio:8.2f} | "
                  f"Epsilon: {stats['epsilon']:.4f} | "
                  f"Trades: {info['total_trades']:3d}")
    
    # Save trained model
    model_path = f"models/trained_models/dqn_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    agent.save_model(model_path)
    
    # Validation
    print("\nRunning validation...")
    val_env = BitcoinTradingEnv(val_data, config_path)
    val_metrics = validate_agent(agent, val_env)
    
    print(f"\nValidation Results:")
    for key, value in val_metrics.items():
        print(f"{key}: {value:.4f}")
    
    # Plot training results
    plot_training_results(episode_rewards, episode_portfolio_values, episode_trades)
    
    return agent, val_metrics

def validate_agent(agent: DQNAgent, env: BitcoinTradingEnv):
    """Validate the trained agent"""
    state = env.reset()
    total_reward = 0
    
    while True:
        action = agent.act(state, training=False)  # No exploration
        next_state, reward, done, info = env.step(action)
        
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # Get performance metrics
    metrics = env.get_performance_metrics()
    metrics['total_reward'] = total_reward
    
    return metrics

def plot_training_results(rewards, portfolio_values, trades):
    """Plot training results"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Episode rewards
    axes[0, 0].plot(rewards)
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Total Reward')
    
    # Portfolio values
    axes[0, 1].plot(portfolio_values)
    axes[0, 1].set_title('Portfolio Value')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Portfolio Value ($)')
    
    # Number of trades
    axes[1, 0].plot(trades)
    axes[1, 0].set_title('Number of Trades per Episode')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Number of Trades')
    
    # Moving average of rewards
    window = 50
    if len(rewards) >= window:
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        axes[1, 1].plot(moving_avg)
        axes[1, 1].set_title(f'Moving Average Reward (window={window})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Reward')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Train Bitcoin RL Trading Bot')
    parser.add_argument('--data', type=str, required=True, help='Path to Bitcoin CSV data')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--episodes', type=int, help='Number of training episodes (overrides config)')
    
    args = parser.parse_args()
    
    # Load and prepare data
    data = load_and_prepare_data(args.data)
    
    # Override episodes if specified
    if args.episodes:
        with open(args.config, 'r') as file:
            config = yaml.safe_load(file)
        config['training']['episodes'] = args.episodes
        with open(args.config, 'w') as file:
            yaml.dump(config, file)
    
    # Train agent
    agent, metrics = train_agent(data, args.config)
    
    print("\nTraining completed successfully!")
    print(f"Final validation metrics: {metrics}")

if __name__ == "__main__":
    main()
