"""Configuration module for ParkWise-RL hyperparameters, reward constants, and lot defaults."""

import numpy as np

# Slot & Vehicle Type Definitions
SLOT_TYPES = ['regular', 'ev_charging', 'reserved', 'handicapped']
VEHICLE_TYPES = ['regular', 'ev', 'reserved', 'handicapped']

SLOT_TYPE_TO_IDX = {st: i for i, st in enumerate(SLOT_TYPES)}
VEHICLE_TYPE_TO_IDX = {vt: i for i, vt in enumerate(VEHICLE_TYPES)}

# Vehicle Distribution Probabilities (sums to 1.0)
VEHICLE_TYPE_PROBS = {
    'regular': 0.70,
    'ev': 0.15,
    'reserved': 0.10,
    'handicapped': 0.05,
}

# Reward Constants
REWARD_VALID_ALLOCATION = 10.0
REWARD_NEAR_DISTANCE_BONUS = 5.0
REWARD_FAR_DISTANCE_PENALTY = -3.0
REWARD_EXCESS_WAIT_PENALTY = -5.0
REWARD_INVALID_ALLOCATION = -10.0
REWARD_UNNECESSARY_REJECTION = -8.0
REWARD_SPECIALTY_MATCH_BONUS = 2.0

# Reward Distance & Wait Thresholds
NEAR_DISTANCE_THRESHOLD = 3.0
FAR_DISTANCE_THRESHOLD = 7.0
WAIT_TIME_THRESHOLD = 2.0

# Episode Settings
DEFAULT_EPISODE_STEPS = 500
BASE_ARRIVAL_RATE = 0.6  # Poisson lambda for baseline arrivals

# Q-Learning Hyperparameters
Q_LEARNING_ALPHA = 0.1
Q_LEARNING_GAMMA = 0.95
Q_LEARNING_EPSILON_START = 1.0
Q_LEARNING_EPSILON_END = 0.02
Q_LEARNING_EPSILON_DECAY = 0.997
Q_LEARNING_EPISODES = 1000

# DQN Hyperparameters
DQN_LR = 1e-3
DQN_GAMMA = 0.99
DQN_BATCH_SIZE = 64
DQN_BUFFER_SIZE = 20000
DQN_TARGET_UPDATE_FREQ = 100
DQN_EPISODES = 600
DQN_HIDDEN_DIM = 128
