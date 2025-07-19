"""
Double Deep Q-Network (DDQN) Agent for Bitcoin Trading
Implements DDQN to reduce overestimation bias
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import yaml
from .dqn_agent import DQNAgent


class DDQNAgent(DQNAgent):
    """Double DQN Agent - extends DQN with target network for action selection"""
    
    def __init__(self, state_size, action_size, config_path="config.yaml"):
        """
        Initialize DDQN Agent
        
        Args:
            state_size (int): Dimension of state space
            action_size (int): Number of possible actions
            config_path (str): Path to configuration file
        """
        super().__init__(state_size, action_size, config_path)
        
        # DDQN uses the same networks but different update rule
        print("Initialized Double DQN Agent")
    
    def replay(self):
        """
        Train the model using Double DQN update rule
        Uses main network for action selection and target network for value estimation
        """
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        # Current Q values from main network
        current_q_values = self.q_network.predict(states, verbose=0)
        
        # DDQN: Use main network to select actions for next states
        next_q_values_main = self.q_network.predict(next_states, verbose=0)
        next_actions = np.argmax(next_q_values_main, axis=1)
        
        # Use target network to evaluate the selected actions
        next_q_values_target = self.target_network.predict(next_states, verbose=0)
        
        # Update Q values using DDQN rule
        for i in range(self.batch_size):
            if dones[i]:
                current_q_values[i][actions[i]] = rewards[i]
            else:
                # DDQN: Q(s,a) = r + γ * Q_target(s', argmax_a Q_main(s',a))
                current_q_values[i][actions[i]] = rewards[i] + self.gamma * next_q_values_target[i][next_actions[i]]
        
        # Train the main network
        history = self.q_network.fit(states, current_q_values, epochs=1, verbose=0)
        self.loss_history.append(history.history['loss'][0])
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def get_training_stats(self):
        """Get training statistics with DDQN identifier"""
        stats = super().get_training_stats()
        stats['agent_type'] = 'DDQN'
        return stats
