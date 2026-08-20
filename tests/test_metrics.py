"""Unit tests for reward calculations and evaluation metrics."""

import pytest
from src.evaluation.metrics import compute_allocation_reward, EpisodeMetricsTracker

def test_reward_logic_valid_and_invalid():
    """Verify reward calculation for valid, invalid, and rejection scenarios."""
    # Valid allocation with near distance and specialty match
    r1 = compute_allocation_reward(
        action=0,
        num_slots=6,
        is_valid_allocation=True,
        is_rejection=False,
        has_compatible_slot=True,
        distance=2.0,
        wait_time=0,
        vehicle_type='ev',
        slot_type='ev_charging'
    )
    # Valid allocation (+10) + Near distance (+5) + Specialty match (+2) = 17
    assert r1 == 17.0

    # Invalid allocation attempt
    r2 = compute_allocation_reward(
        action=0,
        num_slots=6,
        is_valid_allocation=False,
        is_rejection=False,
        has_compatible_slot=True,
        distance=2.0,
        wait_time=0,
        vehicle_type='regular',
        slot_type='regular'
    )
    assert r2 == -10.0

    # Unnecessary rejection when compatible slot exists
    r3 = compute_allocation_reward(
        action=6,
        num_slots=6,
        is_valid_allocation=False,
        is_rejection=True,
        has_compatible_slot=True,
        distance=0.0,
        wait_time=0,
        vehicle_type='regular',
        slot_type='none'
    )
    assert r3 == -8.0

def test_metrics_tracker():
    """Test accumulator metrics aggregation."""
    tracker = EpisodeMetricsTracker()
    tracker.record_step(
        reward=15.0,
        arrived=True,
        allocated=True,
        rejection=False,
        unnecessary_rej=False,
        invalid_attempt=False,
        distance=2.5,
        wait_time=1,
        occupancy_ratio=0.5
    )

    summary = tracker.get_summary()
    assert summary['total_reward'] == 15.0
    assert summary['vehicles_arrived'] == 1
    assert summary['allocated_count'] == 1
    assert summary['avg_walking_distance'] == 2.5
    assert summary['avg_waiting_time'] == 1.0
