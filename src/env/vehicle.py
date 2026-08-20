"""Vehicle model and Poisson arrival process generator for ParkWise-RL."""

from dataclasses import dataclass
import numpy as np
from typing import Optional, List
from src.training.config import VEHICLE_TYPES, VEHICLE_TYPE_PROBS, BASE_ARRIVAL_RATE

@dataclass
class Vehicle:
    id: str
    vehicle_type: str  # 'regular', 'ev', 'reserved', 'handicapped'
    arrival_time: int
    requires_ev: bool
    requires_handicapped: bool
    parking_duration: int
    wait_time: int = 0
    assigned_slot_id: Optional[int] = None

class PoissonArrivalGenerator:
    """Simulates dynamic Poisson vehicle arrivals and duration sampling."""
    
    def __init__(self, seed: Optional[int] = None, base_lambda: float = BASE_ARRIVAL_RATE):
        self.rng = np.random.default_rng(seed)
        self.base_lambda = base_lambda
        self.vehicle_counter = 0

    def seed(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)
        self.vehicle_counter = 0

    def _get_time_of_day_lambda(self, current_step: int, total_steps: int) -> float:
        """Modulate arrival rate based on time of day (morning & evening rush hour peaks)."""
        time_ratio = (current_step % total_steps) / float(total_steps)
        # Peak 1 at 25% of day (morning), Peak 2 at 70% of day (evening)
        peak1 = np.exp(-((time_ratio - 0.25) ** 2) / 0.01)
        peak2 = np.exp(-((time_ratio - 0.70) ** 2) / 0.01)
        multiplier = 1.0 + 1.2 * peak1 + 1.0 * peak2
        return self.base_lambda * multiplier

    def generate_arrivals(self, current_step: int, total_steps: int) -> List[Vehicle]:
        """Sample arrivals for the current discrete time step using Poisson process."""
        lam = self._get_time_of_day_lambda(current_step, total_steps)
        num_arrivals = self.rng.poisson(lam)
        
        arrivals = []
        types = list(VEHICLE_TYPE_PROBS.keys())
        probs = list(VEHICLE_TYPE_PROBS.values())
        
        for _ in range(num_arrivals):
            self.vehicle_counter += 1
            v_type = self.rng.choice(types, p=probs)
            
            requires_ev = (v_type == 'ev')
            requires_handicapped = (v_type == 'handicapped')
            
            # Duration sampled from log-normal distribution (min duration 15 steps, mean ~45 steps)
            duration = int(self.rng.lognormal(mean=3.8, sigma=0.5))
            duration = max(15, min(200, duration))
            
            arrivals.append(Vehicle(
                id=f"V_{self.vehicle_counter:04d}",
                vehicle_type=v_type,
                arrival_time=current_step,
                requires_ev=requires_ev,
                requires_handicapped=requires_handicapped,
                parking_duration=duration
            ))
            
        return arrivals
