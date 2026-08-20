"""Deep Q-Network (DQN) Agent implementation in PyTorch for ParkWise-RL."""

import os
import random
from collections import deque
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.baseline_agent import BaseAgent
from src.env.parking_env import ParkingEnv
from src.training.config import (
    DQN_LR,
    DQN_GAMMA,
    DQN_BATCH_SIZE,
    DQN_BUFFER_SIZE,
    DQN_HIDDEN_DIM
)

class QNetwork(nn.Module):
    """Deep Q-Network MLP Architecture."""

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = DQN_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class ReplayBuffer:
    """Experience Replay Buffer for Q-learning stability."""

    def __init__(self, capacity: int = DQN_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        next_mask: np.ndarray
    ):
        self.buffer.append((obs, action, reward, next_obs, done, next_mask))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        obs, action, reward, next_obs, done, next_mask = zip(*batch)
        return (
            np.array(obs, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(done, dtype=np.float32),
            np.array(next_mask, dtype=bool)
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent(BaseAgent):
    """PyTorch Deep Q-Network Agent supporting action masking."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = DQN_LR,
        gamma: float = DQN_GAMMA,
        hidden_dim: int = DQN_HIDDEN_DIM,
        device: str = "cpu"
    ):
        super().__init__(name="DQN Agent")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_slots = action_dim - 1
        self.gamma = gamma
        self.device = torch.device(device)

        self.policy_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer()

        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.997

    def select_action(
        self,
        obs: np.ndarray,
        info: Dict[str, Any],
        env: ParkingEnv,
        eval_mode: bool = False
    ) -> int:
        mask = info.get('action_mask', env.get_action_mask())

        # Exploration epsilon-greedy
        if (not eval_mode) and (random.random() < self.epsilon):
            valid_actions = np.where(mask)[0]
            if len(valid_actions) > 0:
                return int(np.random.choice(valid_actions))
            return self.action_dim - 1

        # Greedy choice using PyTorch Q-network with action masking
        self.policy_net.eval()
        with torch.no_grad():
            state_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t).squeeze(0).cpu().numpy()

        masked_q = np.full_like(q_values, -1e9)
        masked_q[mask] = q_values[mask]

        max_val = np.max(masked_q)
        best_actions = np.where(masked_q == max_val)[0]
        return int(np.random.choice(best_actions))

    def train_step(self, batch_size: int = DQN_BATCH_SIZE) -> Optional[float]:
        if len(self.replay_buffer) < batch_size:
            return None

        obs, action, reward, next_obs, done, next_mask = self.replay_buffer.sample(batch_size)

        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        action_t = torch.tensor(action, dtype=torch.int64, device=self.device).unsqueeze(1)
        reward_t = torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=self.device)
        done_t = torch.tensor(done, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_mask_t = torch.tensor(next_mask, dtype=torch.bool, device=self.device)

        self.policy_net.train()
        q_eval = self.policy_net(obs_t).gather(1, action_t)

        with torch.no_grad():
            q_next_target = self.target_net(next_obs_t)
            # Mask out invalid actions in target computation
            masked_q_next = q_next_target.masked_fill(~next_mask_t, -1e9)
            max_q_next, _ = masked_q_next.max(dim=1, keepdim=True)
            max_q_next[max_q_next < -1e8] = 0.0  # fallback if all actions masked
            q_target = reward_t + (1.0 - done_t) * self.gamma * max_q_next

        loss = nn.MSELoss()(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No checkpoint found at {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['policy_net'])
        self.policy_net.eval()
