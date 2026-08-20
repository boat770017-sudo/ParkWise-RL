"""Custom Gymnasium Environment for ParkWise-RL Parking Allocation.

State Representation Vector Specification:
------------------------------------------
For a parking lot with N slots:
- Slot occupied bitmap:                     N floats (0.0 or 1.0)
- Slot type one-hot encoding:               4 * N floats (4 types: regular, ev, reserved, handicapped)
- Incoming vehicle type one-hot:            4 floats (all 0.0 if no vehicle waiting)
- Distance to each compatible free slot:    N floats (0.0 if occupied/incompatible, normalized distance otherwise)
- Current congestion ratio:                 1 float (occupied / N)
- Normalized time of day:                   1 float (step / T)

Total State Vector Dimensionality = N + 4*N + 4 + N + 1 + 1 = 6*N + 6.
- For N=6 slots:  Dim = 42
- For N=12 slots: Dim = 78
- For N=20 slots: Dim = 126
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from collections import deque

from src.env.lot_config import create_lot_layout, ParkingSlot
from src.env.vehicle import Vehicle, PoissonArrivalGenerator
from src.training.config import (
    SLOT_TYPE_TO_IDX,
    VEHICLE_TYPE_TO_IDX,
    DEFAULT_EPISODE_STEPS,
    SLOT_TYPES,
    VEHICLE_TYPES
)
from src.evaluation.metrics import compute_allocation_reward, EpisodeMetricsTracker

class ParkingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_slots: int = 12,
        max_steps: int = DEFAULT_EPISODE_STEPS,
        base_arrival_rate: float = 0.6,
        seed: Optional[int] = None
    ):
        super().__init__()
        self.num_slots = num_slots
        self.max_steps = max_steps
        self.base_arrival_rate = base_arrival_rate

        # Initialize slots layout
        self.slots: List[ParkingSlot] = create_lot_layout(self.num_slots)
        self.max_distance = max([s.distance_from_entrance for s in self.slots] + [1.0])

        # State vector dimension: 6 * num_slots + 6
        self.state_dim = 6 * self.num_slots + 6

        # Gym spaces definition
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        # Action space: Discrete(num_slots + 1), where action N is Reject
        self.action_space = spaces.Discrete(self.num_slots + 1)

        # Arrival generator & metrics tracker
        self.arrival_generator = PoissonArrivalGenerator(seed=seed, base_lambda=self.base_arrival_rate)
        self.vehicle_queue: deque[Vehicle] = deque()
        self.current_step = 0
        self.metrics_tracker = EpisodeMetricsTracker()

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.arrival_generator.seed(seed)

        # Clear lot slots
        for s in self.slots:
            s.is_occupied = False
            s.current_vehicle_id = None
            s.departure_tick = -1

        self.vehicle_queue.clear()
        self.current_step = 0
        self.metrics_tracker = EpisodeMetricsTracker()

        # Generate initial arrivals for tick 0
        initial_arrivals = self.arrival_generator.generate_arrivals(self.current_step, self.max_steps)
        self.vehicle_queue.extend(initial_arrivals)

        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        reward = 0.0
        arrived_this_step = False
        allocated_this_step = False
        rejection_this_step = False
        unnecessary_rej = False
        invalid_attempt = False
        chosen_dist = None
        chosen_wait = None
        veh_type = 'regular'
        slot_type = 'regular'

        # 1. Process departures for current step
        self._process_departures()

        # 2. Check if an incoming vehicle is waiting for allocation
        incoming_vehicle = self.vehicle_queue[0] if len(self.vehicle_queue) > 0 else None

        if incoming_vehicle is not None:
            arrived_this_step = True
            veh_type = incoming_vehicle.vehicle_type
            chosen_wait = incoming_vehicle.wait_time
            has_compatible = self._has_compatible_free_slot(incoming_vehicle)

            if action == self.num_slots:
                # Reject Action
                rejection_this_step = True
                if has_compatible:
                    unnecessary_rej = True
                self.vehicle_queue.popleft()  # Remove rejected vehicle
                reward = compute_allocation_reward(
                    action=action,
                    num_slots=self.num_slots,
                    is_valid_allocation=False,
                    is_rejection=True,
                    has_compatible_slot=has_compatible,
                    distance=0.0,
                    wait_time=chosen_wait,
                    vehicle_type=veh_type,
                    slot_type='none'
                )
            elif 0 <= action < self.num_slots:
                target_slot = self.slots[action]
                slot_type = target_slot.slot_type
                chosen_dist = target_slot.distance_from_entrance

                is_compatible = target_slot.is_compatible_with(
                    veh_type,
                    incoming_vehicle.requires_ev,
                    incoming_vehicle.requires_handicapped
                )

                if (not target_slot.is_occupied) and is_compatible:
                    # Valid allocation
                    allocated_this_step = True
                    target_slot.is_occupied = True
                    target_slot.current_vehicle_id = incoming_vehicle.id
                    target_slot.departure_tick = self.current_step + incoming_vehicle.parking_duration
                    self.vehicle_queue.popleft()

                    reward = compute_allocation_reward(
                        action=action,
                        num_slots=self.num_slots,
                        is_valid_allocation=True,
                        is_rejection=False,
                        has_compatible_slot=has_compatible,
                        distance=chosen_dist,
                        wait_time=chosen_wait,
                        vehicle_type=veh_type,
                        slot_type=slot_type
                    )
                else:
                    # Invalid allocation attempt (slot occupied or incompatible)
                    invalid_attempt = True
                    reward = compute_allocation_reward(
                        action=action,
                        num_slots=self.num_slots,
                        is_valid_allocation=False,
                        is_rejection=False,
                        has_compatible_slot=has_compatible,
                        distance=chosen_dist,
                        wait_time=chosen_wait,
                        vehicle_type=veh_type,
                        slot_type=slot_type
                    )
        else:
            # No vehicle waiting; empty tick action
            reward = 0.0

        # Record metrics
        occ_ratio = sum(1 for s in self.slots if s.is_occupied) / float(self.num_slots)
        self.metrics_tracker.record_step(
            reward=reward,
            arrived=arrived_this_step,
            allocated=allocated_this_step,
            rejection=rejection_this_step,
            unnecessary_rej=unnecessary_rej,
            invalid_attempt=invalid_attempt,
            distance=chosen_dist,
            wait_time=chosen_wait,
            occupancy_ratio=occ_ratio
        )

        # 3. Advance step counter & generate new arrivals
        self.current_step += 1
        new_arrivals = self.arrival_generator.generate_arrivals(self.current_step, self.max_steps)
        self.vehicle_queue.extend(new_arrivals)

        # Increment wait time for queued vehicles
        for v in self.vehicle_queue:
            v.wait_time += 1

        # Check episode termination
        terminated = self.current_step >= self.max_steps
        truncated = False

        obs = self._get_observation()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def get_action_mask(self) -> np.ndarray:
        """Return boolean mask of valid actions for the current state (1 for valid, 0 for invalid)."""
        mask = np.zeros(self.num_slots + 1, dtype=bool)
        if len(self.vehicle_queue) == 0:
            # No vehicle waiting, action N (no-op reject) is valid
            mask[self.num_slots] = True
            return mask

        incoming = self.vehicle_queue[0]
        # Action N (Reject) is always valid when vehicle is present
        mask[self.num_slots] = True

        for i, slot in enumerate(self.slots):
            if (not slot.is_occupied) and slot.is_compatible_with(
                incoming.vehicle_type,
                incoming.requires_ev,
                incoming.requires_handicapped
            ):
                mask[i] = True

        return mask

    def _has_compatible_free_slot(self, vehicle: Vehicle) -> bool:
        """Check if any slot is currently unoccupied and compatible with vehicle."""
        for slot in self.slots:
            if (not slot.is_occupied) and slot.is_compatible_with(
                vehicle.vehicle_type, vehicle.requires_ev, vehicle.requires_handicapped
            ):
                return True
        return False

    def _process_departures(self):
        """Free slots where vehicle parking duration has expired."""
        for s in self.slots:
            if s.is_occupied and s.departure_tick >= 0 and self.current_step >= s.departure_tick:
                s.is_occupied = False
                s.current_vehicle_id = None
                s.departure_tick = -1

    def _get_observation(self) -> np.ndarray:
        """Build flat 1D state float32 vector of size (6 * N + 6)."""
        obs = []

        # 1. Occupied bitmap (N values)
        occupied_bitmap = [1.0 if s.is_occupied else 0.0 for s in self.slots]
        obs.extend(occupied_bitmap)

        # 2. Slot types one-hot (4 * N values)
        for s in self.slots:
            type_onehot = [0.0] * 4
            type_idx = SLOT_TYPE_TO_IDX.get(s.slot_type, 0)
            type_onehot[type_idx] = 1.0
            obs.extend(type_onehot)

        # 3. Incoming vehicle type one-hot (4 values)
        incoming = self.vehicle_queue[0] if len(self.vehicle_queue) > 0 else None
        veh_onehot = [0.0] * 4
        if incoming is not None:
            v_idx = VEHICLE_TYPE_TO_IDX.get(incoming.vehicle_type, 0)
            veh_onehot[v_idx] = 1.0
        obs.extend(veh_onehot)

        # 4. Distance to free compatible slots (N values, normalized to [0, 1])
        distances = []
        for s in self.slots:
            if (not s.is_occupied) and incoming is not None and s.is_compatible_with(
                incoming.vehicle_type, incoming.requires_ev, incoming.requires_handicapped
            ):
                distances.append(float(s.distance_from_entrance) / self.max_distance)
            else:
                distances.append(0.0)
        obs.extend(distances)

        # 5. Congestion ratio (1 value)
        congestion = sum(1 for s in self.slots if s.is_occupied) / float(self.num_slots)
        obs.append(float(congestion))

        # 6. Normalized time of day (1 value)
        time_norm = float(self.current_step) / float(max(1, self.max_steps))
        obs.append(float(time_norm))

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        incoming = self.vehicle_queue[0] if len(self.vehicle_queue) > 0 else None
        return {
            'step': self.current_step,
            'queue_length': len(self.vehicle_queue),
            'incoming_vehicle': incoming,
            'action_mask': self.get_action_mask(),
            'metrics': self.metrics_tracker.get_summary()
        }
