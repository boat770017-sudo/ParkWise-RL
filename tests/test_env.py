"""Unit tests for ParkingEnv environment."""

import pytest
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from src.env.parking_env import ParkingEnv

def test_gym_env_checker():
    """Verify that ParkingEnv passes Gymnasium standard environment checks."""
    env = ParkingEnv(num_slots=6, max_steps=100)
    # Gymnasium check_env will validate action space, observation space, reset and step returns
    check_env(env.unwrapped, skip_render_check=True)

def test_state_vector_dimension():
    """Check exact state vector dimensionality rule (6 * N + 6)."""
    for n_slots in [6, 12, 20]:
        env = ParkingEnv(num_slots=n_slots, max_steps=100)
        obs, _ = env.reset()
        expected_dim = 6 * n_slots + 6
        assert obs.shape == (expected_dim,)
        assert obs.dtype == np.float32

def test_valid_allocation():
    """Test assigning an incoming vehicle to a free compatible slot."""
    env = ParkingEnv(num_slots=6, max_steps=100, seed=42)
    obs, info = env.reset(seed=42)
    
    # Ensure there is a vehicle
    while info['incoming_vehicle'] is None:
        obs, r, term, trunc, info = env.step(env.num_slots)
        
    mask = env.get_action_mask()
    valid_slot_action = -1
    for a in range(env.num_slots):
        if mask[a]:
            valid_slot_action = a
            break
            
    assert valid_slot_action != -1
    obs_next, reward, term, trunc, info_next = env.step(valid_slot_action)
    assert reward > 0.0  # Valid allocation yields positive reward

def test_invalid_allocation():
    """Test assigning vehicle to an occupied or invalid slot."""
    env = ParkingEnv(num_slots=6, max_steps=100, seed=42)
    obs, info = env.reset(seed=42)

    while info['incoming_vehicle'] is None:
        obs, r, term, trunc, info = env.step(env.num_slots)
    
    # Occupy slot 0 manually
    env.slots[0].is_occupied = True
    env.slots[0].departure_tick = 999
    
    # Attempt to assign to slot 0 when occupied
    obs_next, reward, term, trunc, info_next = env.step(0)
    assert reward <= -5.0  # Invalid attempt penalty

def test_rejection_action():
    """Test rejection action when compatible slot exists vs when full."""
    env = ParkingEnv(num_slots=6, max_steps=100, seed=42)
    obs, info = env.reset(seed=42)
    
    while info['incoming_vehicle'] is None:
        obs, r, term, trunc, info = env.step(env.num_slots)
        
    # Reject action is index N
    reject_action = env.num_slots
    obs_next, reward, term, trunc, info_next = env.step(reject_action)
    
    # Unnecessary rejection penalized
    assert reward < 0.0

def test_vehicle_departure():
    """Verify occupied slots free up when departure tick passes."""
    env = ParkingEnv(num_slots=6, max_steps=100, seed=42)
    env.reset(seed=42)
    
    # Manually occupy slot 0 with departure tick = 5
    env.slots[0].is_occupied = True
    env.slots[0].departure_tick = 5
    env.current_step = 5
    
    # Step environment to trigger departure processing
    env.step(env.num_slots)
    
    # Slot 0 should now be free
    assert not env.slots[0].is_occupied
