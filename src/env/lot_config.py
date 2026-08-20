"""Lot configuration module for ParkWise-RL layout grid, slot metadata, and precomputed distances."""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from src.training.config import SLOT_TYPES

@dataclass
class ParkingSlot:
    id: int
    slot_type: str  # 'regular', 'ev_charging', 'reserved', 'handicapped'
    grid_x: int
    grid_y: int
    distance_from_entrance: float
    is_occupied: bool = False
    current_vehicle_id: str = None
    departure_tick: int = -1

    def is_compatible_with(self, vehicle_type: str, requires_ev: bool, requires_handicapped: bool) -> bool:
        """Check if vehicle is allowed to park in this slot type."""
        # Handicapped slots reserved strictly for handicapped vehicles
        if self.slot_type == 'handicapped':
            return vehicle_type == 'handicapped' or requires_handicapped
        
        # Reserved slots only for reserved or handicapped vehicles
        if self.slot_type == 'reserved':
            return vehicle_type in ['reserved', 'handicapped']
        
        # EV charging slots only for EV vehicles
        if self.slot_type == 'ev_charging':
            return vehicle_type == 'ev' or requires_ev
        
        # Regular slot can accept regular or EV/reserved if allowed (handicapped can also park in regular)
        return True


def create_lot_layout(num_slots: int = 12, entrance_pos: Tuple[int, int] = (0, 0)) -> List[ParkingSlot]:
    """Generate a parking lot layout with precomputed Manhattan distances from entrance."""
    slots = []
    # Determine grid dimensions
    cols = min(6, num_slots)
    
    for i in range(num_slots):
        row = i // cols + 1  # 1-indexed row to avoid 0 distance for first slot
        col = i % cols + 1
        
        # Precompute Manhattan distance from entrance (0, 0)
        dist = float(abs(row - entrance_pos[0]) + abs(col - entrance_pos[1]))
        
        # Assign slot types based on index position pattern
        if i == 0 and num_slots >= 4:
            stype = 'handicapped'
        elif i in [1, 2] and num_slots >= 6:
            stype = 'ev_charging'
        elif i in [3, 4] and num_slots >= 8:
            stype = 'reserved'
        else:
            stype = 'regular'
            
        slots.append(ParkingSlot(
            id=i,
            slot_type=stype,
            grid_x=col,
            grid_y=row,
            distance_from_entrance=dist,
            is_occupied=False
        ))
        
    return slots
