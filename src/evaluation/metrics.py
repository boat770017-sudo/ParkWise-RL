"""Evaluation metrics and reward calculation logic for ParkWise-RL."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from src.training.config import (
    REWARD_VALID_ALLOCATION,
    REWARD_NEAR_DISTANCE_BONUS,
    REWARD_FAR_DISTANCE_PENALTY,
    REWARD_EXCESS_WAIT_PENALTY,
    REWARD_INVALID_ALLOCATION,
    REWARD_UNNECESSARY_REJECTION,
    REWARD_SPECIALTY_MATCH_BONUS,
    NEAR_DISTANCE_THRESHOLD,
    FAR_DISTANCE_THRESHOLD,
    WAIT_TIME_THRESHOLD,
)

def compute_allocation_reward(
    action: int,
    num_slots: int,
    is_valid_allocation: bool,
    is_rejection: bool,
    has_compatible_slot: bool,
    distance: float,
    wait_time: int,
    vehicle_type: str,
    slot_type: str
) -> float:
    """Compute step reward based on agent action and lot environment context."""
    reward = 0.0
    
    if is_rejection:
        if has_compatible_slot:
            # Rejected vehicle despite compatible free slot existing
            reward += REWARD_UNNECESSARY_REJECTION
        else:
            # Correct rejection when lot is full / no compatible slots exist
            reward += 0.0
        return reward

    if not is_valid_allocation:
        # Invalid action: slot occupied or incompatible type
        reward += REWARD_INVALID_ALLOCATION
        return reward

    # Valid allocation
    reward += REWARD_VALID_ALLOCATION

    # Distance rewards/penalties
    if distance <= NEAR_DISTANCE_THRESHOLD:
        reward += REWARD_NEAR_DISTANCE_BONUS
    elif distance > FAR_DISTANCE_THRESHOLD:
        reward += REWARD_FAR_DISTANCE_PENALTY

    # Excess wait time penalty
    if wait_time > WAIT_TIME_THRESHOLD:
        reward += REWARD_EXCESS_WAIT_PENALTY

    # Specialty bonus (EV in EV charging, Handicapped in Handicapped slot)
    if (vehicle_type == 'ev' and slot_type == 'ev_charging') or \
       (vehicle_type == 'handicapped' and slot_type == 'handicapped'):
        reward += REWARD_SPECIALTY_MATCH_BONUS

    return reward

@dataclass
class EpisodeMetricsTracker:
    """Accumulates episode metrics for logging and agent comparative analysis."""
    total_reward: float = 0.0
    total_vehicles_arrived: int = 0
    total_allocated: int = 0
    total_rejections: int = 0
    unnecessary_rejections: int = 0
    invalid_attempts: int = 0
    walking_distances: List[float] = field(default_factory=list)
    waiting_times: List[int] = field(default_factory=list)
    occupancy_history: List[float] = field(default_factory=list)

    def record_step(
        self,
        reward: float,
        arrived: bool,
        allocated: bool,
        rejection: bool,
        unnecessary_rej: bool,
        invalid_attempt: bool,
        distance: Optional[float] = None,
        wait_time: Optional[int] = None,
        occupancy_ratio: float = 0.0
    ):
        self.total_reward += reward
        self.occupancy_history.append(occupancy_ratio)

        if arrived:
            self.total_vehicles_arrived += 1

        if allocated:
            self.total_allocated += 1
            if distance is not None:
                self.walking_distances.append(distance)
            if wait_time is not None:
                self.waiting_times.append(wait_time)

        if rejection:
            self.total_rejections += 1
            if unnecessary_rej:
                self.unnecessary_rejections += 1

        if invalid_attempt:
            self.invalid_attempts += 1

    def get_summary(self) -> Dict[str, float]:
        avg_dist = float(np.mean(self.walking_distances)) if self.walking_distances else 0.0
        avg_wait = float(np.mean(self.waiting_times)) if self.waiting_times else 0.0
        avg_occ = float(np.mean(self.occupancy_history)) if self.occupancy_history else 0.0
        rej_rate = (self.unnecessary_rejections / max(1, self.total_vehicles_arrived)) * 100.0

        return {
            'total_reward': self.total_reward,
            'vehicles_arrived': self.total_vehicles_arrived,
            'allocated_count': self.total_allocated,
            'total_rejections': self.total_rejections,
            'unnecessary_rejections': self.unnecessary_rejections,
            'invalid_attempts': self.invalid_attempts,
            'avg_walking_distance': avg_dist,
            'avg_waiting_time': avg_wait,
            'avg_occupancy_rate': avg_occ,
            'unnecessary_rejection_rate_pct': rej_rate
        }
