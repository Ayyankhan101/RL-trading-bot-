"""
Dueling Deep Q-Network (Dueling DQN) Agent for Bitcoin Trading
Implements Dueling DQN architecture with separate value and advantage streams
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from collections import deque
import random
import yaml


class DuelingDQNAgent:
    """Dueling DQN Agent with separate value and advantage estimation"""
    
    def __init__(self, state_size, action_size, config_path="config.yaml"):
        """
        Initialize Dueling DQN Agent
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Number of possible actions
            config_path (str): Path to configuration file
        """
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.state_size = state_size
        self.action_size = action_size
        
        # Hyperparameters
        self.learning_rate = self.config['model']['learning_rate']
        self.epsilon = self.config['model']['epsilon_start']
        self.epsilon_min = self.config['model']['epsilon_end']
        self.epsilon_decay = self.config['model']['epsilon_decay']
        self.gamma = self.config['model']['gamma']
        self.batch_size = self.config['model']['batch_size']
        
        # Experience replay
        self.memory = deque(maxlen=self.config['model']['memory_size'])
        
        # Neural networks
        self.q_network = self._build_dueling_model()
        self.target_network = self._build_dueling_model()
        self.update_target_network()
        
        # Training metrics
        self.loss_history = []
        self.reward_history = []
        
        print("Initialized Dueling DQN Agent")
    
    def _build_dueling_model(self):
        """Build the Dueling DQN model with separate value and advantage streams"""
        # Input layer
        inputs = layers.Input(shape=(self.state_size,))
        
        # Shared layers
        x = layers.Dense(
            self.config['network']['hidden_layers'][0], 
            activation=self.config['network']['activation']
        )(inputs)
        x = layers.Dropout(self.config['network']['dropout_rate'])(x)
        
        x = layers.Dense(
            self.config['network']['hidden_layers'][1],
            activation=self.config['network']['activation']
        )(x)
        x = layers.Dropout(self.config['network']['dropout_rate'])(x)
        
        # Value stream - estimates V(s)
        value_stream = layers.Dense(
            self.config['network']['hidden_layers'][2] // 2,
            activation=self.config['network']['activation']
        )(x)
        value_stream = layers.Dense(1, activation='linear', name='value')(value_stream)
        
        # Advantage stream - estimates A(s,a)
        advantage_stream = layers.Dense(
            self.config['network']['hidden_layers'][2] // 2,
            activation=self.config['network']['activation']
        )(x)
        advantage_stream = layers.Dense(
            self.action_size, 
            activation='linear', 
            name='advantage'
        )(advantage_stream)
        
        # Combine streams using dueling architecture
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        advantage_mean = tf.reduce_mean(advantage_stream, axis=1, keepdims=True)
        q_values = value_stream + advantage_stream - advantage_mean
        
        model = keras.Model(inputs=inputs, outputs=q_values)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, training=True):
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            action: Selected action
        """
        if training and np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self):
        """Train the model on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q values
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # Next Q values from target network
        next_q_values = self.target_network.predict(next_states, verbose=0)
        
        # Update Q values
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        
        # Train the model
        history = self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
        self.loss_history.append(history.history['loss'][0])
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.set_weights(self.q_network.get_weights())
    
    def save_model(self, filepath):
        """Save the trained model"""
        self.q_network.save(filepath)
        print(f"Dueling DQN model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.q_network = keras.models.load_model(filepath)
        self.target_network = keras.models.load_model(filepath)
        print(f"Dueling DQN model loaded from {filepath}")
    
    def get_training_stats(self):
        """Get training statistics"""
        return {
            'agent_type': 'Dueling DQN',
            'epsilon': self.epsilon,
            'memory_size': len(self.memory),
            'avg_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            'avg_reward': np.mean(self.reward_history[-100:]) if self.reward_history else 0
        }
    
    def get_value_and_advantage(self, state):
        """
        Get separate value and advantage estimates for analysis
        
        Args:
            state: Input state
            
        Returns:
            tuple: (value, advantages, q_values)
        """
        # Create a model that outputs intermediate layers
        value_model = keras.Model(
            inputs=self.q_network.input,
            outputs=self.q_network.get_layer('value').output
        )
        
        advantage_model = keras.Model(
            inputs=self.q_network.input,
            outputs=self.q_network.get_layer('advantage').output
        )
        
        state_input = state.reshape(1, -1)
        value = value_model.predict(state_input, verbose=0)[0][0]
        advantages = advantage_model.predict(state_input, verbose=0)[0]
        q_values = self.q_network.predict(state_input, verbose=0)[0]
        
        return value, advantages, q_values
    
    def analyze_decision(self, state):
        """
        Analyze the agent's decision-making process
        
        Args:
            state: Input state
            
        Returns:
            dict: Analysis of value, advantages, and final Q-values
        """
        value, advantages, q_values = self.get_value_and_advantage(state)
        
        analysis = {
            'state_value': value,
            'advantages': advantages.tolist(),
            'q_values': q_values.tolist(),
            'selected_action': np.argmax(q_values),
            'advantage_spread': np.max(advantages) - np.min(advantages),
            'confidence': np.max(q_values) - np.mean(q_values)
        }
        
        return analysis
