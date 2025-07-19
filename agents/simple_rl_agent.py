"""
Simplified RL Trading Agent using Q-Learning
A lightweight implementation that works without TensorFlow
"""

import numpy as np
import pandas as pd
import pickle
from collections import defaultdict, deque
import random
import yaml

class SimpleRLAgent:
    def __init__(self, config_path="config.yaml"):
        """
        Initialize Simple RL Agent using Q-Learning
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Q-Learning parameters
        self.learning_rate = self.config['model']['learning_rate']
        self.epsilon = self.config['model']['epsilon_start']
        self.epsilon_min = self.config['model']['epsilon_end']
        self.epsilon_decay = self.config['model']['epsilon_decay']
        self.gamma = self.config['model']['gamma']
        
        # Q-table (state -> action -> value)
        self.q_table = defaultdict(lambda: defaultdict(float))
        
        # Experience replay
        self.memory = deque(maxlen=self.config['model']['memory_size'])
        
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.actions = [0, 1, 2]
        self.action_names = ['Hold', 'Buy', 'Sell']
        
        # Training metrics
        self.training_rewards = []
        self.training_actions = []
        
    def discretize_state(self, state):
        """Convert continuous state to discrete state for Q-table"""
        # Simple discretization - bin continuous values
        discrete_state = []
        
        for i, value in enumerate(state):
            if np.isnan(value):
                discrete_state.append(0)
            else:
                # Bin values into 10 discrete levels
                binned = int(np.clip(value * 10, -50, 50))
                discrete_state.append(binned)
        
        return tuple(discrete_state[:10])  # Use first 10 features to keep state space manageable
    
    def act(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        discrete_state = self.discretize_state(state)
        
        if training and np.random.random() <= self.epsilon:
            return random.choice(self.actions)
        
        # Choose action with highest Q-value
        q_values = [self.q_table[discrete_state][action] for action in self.actions]
        return self.actions[np.argmax(q_values)]
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def replay(self):
        """Train the agent using Q-Learning"""
        if len(self.memory) < 32:  # Minimum batch size
            return
        
        # Sample random batch
        batch = random.sample(self.memory, min(32, len(self.memory)))
        
        for state, action, reward, next_state, done in batch:
            discrete_state = self.discretize_state(state)
            discrete_next_state = self.discretize_state(next_state)
            
            # Current Q-value
            current_q = self.q_table[discrete_state][action]
            
            if done:
                target_q = reward
            else:
                # Max Q-value for next state
                next_q_values = [self.q_table[discrete_next_state][a] for a in self.actions]
                target_q = reward + self.gamma * max(next_q_values)
            
            # Update Q-value
            self.q_table[discrete_state][action] += self.learning_rate * (target_q - current_q)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        """Save the Q-table"""
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load the Q-table"""
        with open(filepath, 'rb') as f:
            loaded_q_table = pickle.load(f)
            self.q_table = defaultdict(lambda: defaultdict(float), loaded_q_table)
        print(f"Model loaded from {filepath}")
    
    def get_training_stats(self):
        """Get training statistics"""
        return {
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'q_table_size': len(self.q_table),
            'avg_reward': np.mean(self.training_rewards[-100:]) if self.training_rewards else 0
        }
