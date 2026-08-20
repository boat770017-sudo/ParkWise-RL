"""Baseline Agent Implementations for ParkWise-RL.

Includes:
- NearestSlotAgent: Greedy heuristic picking closest compatible free slot.
- FCFSAgent: First-Come-First-Served picking first compatible free slot by index.
"""

from typing import Dict, Any, Optional
import numpy as np
from src.env.parking_env import ParkingEnv

class BaseAgent:
    """Base class interface for all ParkWise-RL agents."""
    def __init__(self, name: str = "BaseAgent"):
        self.name = name

    def select_action(self, obs: np.ndarray, info: Dict[str, Any], env: ParkingEnv) -> int:
        raise NotImplementedError

class NearestSlotAgent(BaseAgent):
    """Greedy heuristic: Selects the closest available compatible parking slot."""
    def __init__(self):
        super().__init__(name="Nearest-Slot Baseline")

    def select_action(self, obs: np.ndarray, info: Dict[str, Any], env: ParkingEnv) -> int:
        incoming_vehicle = info.get('incoming_vehicle')
        reject_action = env.num_slots

        if incoming_vehicle is None:
            return reject_action

        mask = info.get('action_mask', env.get_action_mask())
        best_slot = reject_action
        min_distance = float('inf')

        for i, slot in enumerate(env.slots):
            if mask[i]:  # Compatible and free
                if slot.distance_from_entrance < min_distance:
                    min_distance = slot.distance_from_entrance
                    best_slot = i

        return best_slot

class FCFSAgent(BaseAgent):
    """First-Come-First-Served heuristic: Selects the first available compatible slot by index order."""
    def __init__(self):
        super().__init__(name="FCFS Baseline")

    def select_action(self, obs: np.ndarray, info: Dict[str, Any], env: ParkingEnv) -> int:
        incoming_vehicle = info.get('incoming_vehicle')
        reject_action = env.num_slots

        if incoming_vehicle is None:
            return reject_action

        mask = info.get('action_mask', env.get_action_mask())
        for i in range(env.num_slots):
            if mask[i]:
                return i

        return reject_action
