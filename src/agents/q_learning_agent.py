"""Tabular Q-Learning Agent for ParkWise-RL with state space discretization."""

import pickle
import os
import numpy as np
from typing import Dict, Tuple, Any, Optional
from src.agents.baseline_agent import BaseAgent
from src.env.parking_env import ParkingEnv
from src.training.config import (
    Q_LEARNING_ALPHA,
    Q_LEARNING_GAMMA,
    Q_LEARNING_EPSILON_START,
    Q_LEARNING_EPSILON_END,
    Q_LEARNING_EPSILON_DECAY,
    VEHICLE_TYPE_TO_IDX
)

class QLearningAgent(BaseAgent):
    """Tabular Q-Learning Agent with discretized state feature mapping."""

    def __init__(
        self,
        num_slots: int = 6,
        alpha: float = Q_LEARNING_ALPHA,
        gamma: float = Q_LEARNING_GAMMA,
        epsilon_start: float = Q_LEARNING_EPSILON_START,
        epsilon_end: float = Q_LEARNING_EPSILON_END,
        epsilon_decay: float = Q_LEARNING_EPSILON_DECAY
    ):
        super().__init__(name="Q-Learning Agent")
        self.num_slots = num_slots
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.num_actions = num_slots + 1
        self.q_table: Dict[Tuple, np.ndarray] = {}

    def discretize_state(self, info: Dict[str, Any], env: ParkingEnv) -> Tuple:
        """Discretize high-dimensional state vector into discrete state tuple."""
        # 1. Congestion ratio bucketed into 5 levels [0..4]
        occ_count = sum(1 for s in env.slots if s.is_occupied)
        congestion_bucket = min(4, int((occ_count / float(env.num_slots)) * 5))

        # 2. Incoming vehicle type [0: None, 1: Regular, 2: EV, 3: Reserved, 4: Handicapped]
        incoming = info.get('incoming_vehicle')
        if incoming is None:
            veh_code = 0
        else:
            veh_code = VEHICLE_TYPE_TO_IDX.get(incoming.vehicle_type, 0) + 1

        # 3. Occupancy summary per slot type tuple
        occ_types = [0, 0, 0, 0]  # [regular, ev, reserved, handicapped]
        for slot in env.slots:
            if slot.is_occupied:
                t_idx = VEHICLE_TYPE_TO_IDX.get(slot.slot_type if slot.slot_type != 'ev_charging' else 'ev', 0)
                occ_types[t_idx] += 1

        return (congestion_bucket, veh_code, tuple(occ_types))

    def _get_q_values(self, state_key: Tuple) -> np.ndarray:
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions, dtype=np.float32)
        return self.q_table[state_key]

    def select_action(
        self,
        obs: np.ndarray,
        info: Dict[str, Any],
        env: ParkingEnv,
        eval_mode: bool = False
    ) -> int:
        state_key = self.discretize_state(info, env)
        q_vals = self._get_q_values(state_key)
        mask = info.get('action_mask', env.get_action_mask())

        # Exploration epsilon-greedy
        if (not eval_mode) and (np.random.rand() < self.epsilon):
            valid_actions = np.where(mask)[0]
            if len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))
            return self.num_slots

        # Greedy choice among valid masked actions
        masked_q = np.full_like(q_vals, -1e9)
        masked_q[mask] = q_vals[mask]
        
        # Break ties randomly
        max_val = np.max(masked_q)
        best_actions = np.where(masked_q == max_val)[0]
        return int(np.random.choice(best_actions))

    def update(
        self,
        state_key: Tuple,
        action: int,
        reward: float,
        next_state_key: Tuple,
        done: bool,
        next_mask: np.ndarray
    ):
        """Standard Q-Learning Bellman equation update."""
        current_q = self._get_q_values(state_key)[action]
        next_q_vals = self._get_q_values(next_state_key)

        masked_next_q = np.full_like(next_q_vals, -1e9)
        masked_next_q[next_mask] = next_q_vals[next_mask]
        max_next_q = 0.0 if done else np.max(masked_next_q)

        target = reward + self.gamma * max_next_q
        self.q_table[state_key][action] += self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump({'q_table': self.q_table, 'num_slots': self.num_slots}, f)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No Q-table found at {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.q_table = data['q_table']
            self.num_slots = data.get('num_slots', self.num_slots)
