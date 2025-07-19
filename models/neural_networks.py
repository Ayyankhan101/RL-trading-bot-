"""
Neural Network Architectures for RL Trading Bot
Implements various deep learning models for trading
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import Tuple, Optional


class DQNNetwork:
    """Deep Q-Network architecture"""
    
    def __init__(self, state_size: int, action_size: int, config: dict):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
    def build_model(self) -> keras.Model:
        """Build DQN model"""
        model = keras.Sequential([
            layers.Dense(
                self.config['network']['hidden_layers'][0],
                input_shape=(self.state_size,),
                activation=self.config['network']['activation']
            ),
            layers.Dropout(self.config['network']['dropout_rate']),
            
            layers.Dense(
                self.config['network']['hidden_layers'][1],
                activation=self.config['network']['activation']
            ),
            layers.Dropout(self.config['network']['dropout_rate']),
            
            layers.Dense(
                self.config['network']['hidden_layers'][2],
                activation=self.config['network']['activation']
            ),
            layers.Dropout(self.config['network']['dropout_rate']),
            
            layers.Dense(self.action_size, activation='linear')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['model']['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model


class DuelingDQNNetwork:
    """Dueling DQN architecture with separate value and advantage streams"""
    
    def __init__(self, state_size: int, action_size: int, config: dict):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
    def build_model(self) -> keras.Model:
        """Build Dueling DQN model"""
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
        
        # Value stream
        value_stream = layers.Dense(
            self.config['network']['hidden_layers'][2] // 2,
            activation=self.config['network']['activation']
        )(x)
        value_stream = layers.Dense(1, activation='linear')(value_stream)
        
        # Advantage stream
        advantage_stream = layers.Dense(
            self.config['network']['hidden_layers'][2] // 2,
            activation=self.config['network']['activation']
        )(x)
        advantage_stream = layers.Dense(self.action_size, activation='linear')(advantage_stream)
        
        # Combine streams
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        advantage_mean = tf.reduce_mean(advantage_stream, axis=1, keepdims=True)
        q_values = value_stream + advantage_stream - advantage_mean
        
        model = keras.Model(inputs=inputs, outputs=q_values)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['model']['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model


class CNNDQNNetwork:
    """CNN-based DQN for processing sequential market data"""
    
    def __init__(self, state_size: int, action_size: int, config: dict, sequence_length: int = 60):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.sequence_length = sequence_length
        
    def build_model(self) -> keras.Model:
        """Build CNN-DQN model"""
        # Reshape input for CNN (assuming state contains sequential data)
        inputs = layers.Input(shape=(self.sequence_length, self.state_size // self.sequence_length))
        
        # CNN layers for pattern recognition
        x = layers.Conv1D(filters=32, kernel_size=3, activation='relu')(inputs)
        x = layers.Conv1D(filters=64, kernel_size=3, activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)
        
        x = layers.Conv1D(filters=128, kernel_size=3, activation='relu')(x)
        x = layers.GlobalMaxPooling1D()(x)
        
        # Dense layers
        x = layers.Dense(
            self.config['network']['hidden_layers'][0],
            activation=self.config['network']['activation']
        )(x)
        x = layers.Dropout(self.config['network']['dropout_rate'])(x)
        
        x = layers.Dense(
            self.config['network']['hidden_layers'][1],
            activation=self.config['network']['activation']
        )(x)
        x = layers.Dropout(self.config['network']['dropout_rate'])(x)
        
        # Output layer
        outputs = layers.Dense(self.action_size, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['model']['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model


class LSTMDQNNetwork:
    """LSTM-based DQN for temporal dependencies"""
    
    def __init__(self, state_size: int, action_size: int, config: dict, sequence_length: int = 60):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.sequence_length = sequence_length
        
    def build_model(self) -> keras.Model:
        """Build LSTM-DQN model"""
        inputs = layers.Input(shape=(self.sequence_length, self.state_size // self.sequence_length))
        
        # LSTM layers
        x = layers.LSTM(
            self.config['network']['lstm_units'],
            return_sequences=True,
            dropout=self.config['network']['dropout_rate']
        )(inputs)
        
        x = layers.LSTM(
            self.config['network']['lstm_units'] // 2,
            return_sequences=False,
            dropout=self.config['network']['dropout_rate']
        )(x)
        
        # Dense layers
        x = layers.Dense(
            self.config['network']['hidden_layers'][0],
            activation=self.config['network']['activation']
        )(x)
        x = layers.Dropout(self.config['network']['dropout_rate'])(x)
        
        x = layers.Dense(
            self.config['network']['hidden_layers'][1],
            activation=self.config['network']['activation']
        )(x)
        x = layers.Dropout(self.config['network']['dropout_rate'])(x)
        
        # Output layer
        outputs = layers.Dense(self.action_size, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['model']['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model


class AttentionDQNNetwork:
    """DQN with attention mechanism for feature importance"""
    
    def __init__(self, state_size: int, action_size: int, config: dict):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        
    def attention_layer(self, inputs, units: int):
        """Custom attention layer"""
        # Query, Key, Value projections
        query = layers.Dense(units)(inputs)
        key = layers.Dense(units)(inputs)
        value = layers.Dense(units)(inputs)
        
        # Attention scores
        scores = tf.matmul(query, key, transpose_b=True)
        scores = tf.nn.softmax(scores / tf.sqrt(tf.cast(units, tf.float32)))
        
        # Apply attention
        attended = tf.matmul(scores, value)
        
        return attended
        
    def build_model(self) -> keras.Model:
        """Build Attention-DQN model"""
        inputs = layers.Input(shape=(self.state_size,))
        
        # Reshape for attention (treat features as sequence)
        x = layers.Reshape((self.state_size, 1))(inputs)
        
        # Apply attention
        if self.config['network']['use_attention']:
            x = self.attention_layer(x, 64)
            x = layers.GlobalAveragePooling1D()(x)
        else:
            x = layers.Flatten()(x)
        
        # Dense layers
        x = layers.Dense(
            self.config['network']['hidden_layers'][0],
            activation=self.config['network']['activation']
        )(x)
        x = layers.Dropout(self.config['network']['dropout_rate'])(x)
        
        x = layers.Dense(
            self.config['network']['hidden_layers'][1],
            activation=self.config['network']['activation']
        )(x)
        x = layers.Dropout(self.config['network']['dropout_rate'])(x)
        
        x = layers.Dense(
            self.config['network']['hidden_layers'][2],
            activation=self.config['network']['activation']
        )(x)
        x = layers.Dropout(self.config['network']['dropout_rate'])(x)
        
        # Output layer
        outputs = layers.Dense(self.action_size, activation='linear')(x)
        
        model = keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['model']['learning_rate']),
            loss='mse',
            metrics=['mae']
        )
        
        return model


class EnsembleNetwork:
    """Ensemble of multiple networks for robust predictions"""
    
    def __init__(self, state_size: int, action_size: int, config: dict, n_models: int = 3):
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.n_models = n_models
        self.models = []
        
    def build_ensemble(self) -> list:
        """Build ensemble of models"""
        # Create different architectures
        architectures = [
            DQNNetwork(self.state_size, self.action_size, self.config),
            DuelingDQNNetwork(self.state_size, self.action_size, self.config),
            AttentionDQNNetwork(self.state_size, self.action_size, self.config)
        ]
        
        models = []
        for i in range(self.n_models):
            arch = architectures[i % len(architectures)]
            model = arch.build_model()
            models.append(model)
            
        self.models = models
        return models
    
    def predict_ensemble(self, state: np.ndarray) -> np.ndarray:
        """Make ensemble prediction"""
        if not self.models:
            raise ValueError("Ensemble not built yet. Call build_ensemble() first.")
        
        predictions = []
        for model in self.models:
            pred = model.predict(state, verbose=0)
            predictions.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(predictions, axis=0)
        return ensemble_pred
    
    def train_ensemble(self, states: np.ndarray, targets: np.ndarray, epochs: int = 1):
        """Train all models in ensemble"""
        for model in self.models:
            model.fit(states, targets, epochs=epochs, verbose=0)


def create_model(architecture: str, state_size: int, action_size: int, config: dict) -> keras.Model:
    """
    Factory function to create different model architectures
    
    Args:
        architecture: Model architecture name
        state_size: Input state size
        action_size: Number of actions
        config: Configuration dictionary
        
    Returns:
        Compiled Keras model
    """
    if architecture.lower() == 'dqn':
        network = DQNNetwork(state_size, action_size, config)
        return network.build_model()
    
    elif architecture.lower() == 'dueling_dqn':
        network = DuelingDQNNetwork(state_size, action_size, config)
        return network.build_model()
    
    elif architecture.lower() == 'cnn_dqn':
        network = CNNDQNNetwork(state_size, action_size, config)
        return network.build_model()
    
    elif architecture.lower() == 'lstm_dqn':
        network = LSTMDQNNetwork(state_size, action_size, config)
        return network.build_model()
    
    elif architecture.lower() == 'attention_dqn':
        network = AttentionDQNNetwork(state_size, action_size, config)
        return network.build_model()
    
    else:
        raise ValueError(f"Unknown architecture: {architecture}")


def model_summary(model: keras.Model) -> None:
    """Print detailed model summary"""
    print("=" * 60)
    print("MODEL ARCHITECTURE SUMMARY")
    print("=" * 60)
    
    model.summary()
    
    # Calculate total parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Non-trainable Parameters: {non_trainable_params:,}")
    print("=" * 60)


if __name__ == "__main__":
    # Example usage
    print("Testing neural network architectures...")
    
    # Sample configuration
    config = {
        'model': {'learning_rate': 0.001},
        'network': {
            'hidden_layers': [256, 128, 64],
            'activation': 'relu',
            'dropout_rate': 0.2,
            'lstm_units': 50,
            'use_attention': True
        }
    }
    
    state_size = 100
    action_size = 3
    
    # Test different architectures
    architectures = ['dqn', 'dueling_dqn', 'attention_dqn']
    
    for arch in architectures:
        print(f"\nTesting {arch.upper()} architecture:")
        model = create_model(arch, state_size, action_size, config)
        print(f"Model created successfully with {model.count_params():,} parameters")
