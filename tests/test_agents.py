"""Unit tests for baseline, Q-learning, and DQN agents."""

import os
import pytest
import numpy as np
from src.env.parking_env import ParkingEnv
from src.agents.baseline_agent import NearestSlotAgent, FCFSAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent

def test_baseline_agents_action_selection():
    """Verify nearest slot and FCFS baseline action choices."""
    env = ParkingEnv(num_slots=6, max_steps=50)
    obs, info = env.reset(seed=123)
    
    nearest_agent = NearestSlotAgent()
    fcfs_agent = FCFSAgent()

    action_nearest = nearest_agent.select_action(obs, info, env)
    action_fcfs = fcfs_agent.select_action(obs, info, env)

    assert 0 <= action_nearest <= env.num_slots
    assert 0 <= action_fcfs <= env.num_slots

def test_q_learning_save_and_load(tmp_path):
    """Test tabular Q-learning save and restore."""
    env = ParkingEnv(num_slots=6, max_steps=50)
    obs, info = env.reset(seed=123)

    q_agent = QLearningAgent(num_slots=6)
    # Simulate an update
    state_key = q_agent.discretize_state(info, env)
    q_agent.update(state_key, 0, 10.0, state_key, False, env.get_action_mask())

    model_file = os.path.join(tmp_path, "q_table.pkl")
    q_agent.save(model_file)
    assert os.path.exists(model_file)

    restored_agent = QLearningAgent(num_slots=6)
    restored_agent.load(model_file)
    assert state_key in restored_agent.q_table
    assert np.allclose(restored_agent.q_table[state_key], q_agent.q_table[state_key])

def test_dqn_agent_step_and_save(tmp_path):
    """Test PyTorch DQN forward pass, replay buffer, and checkpoint persistence."""
    env = ParkingEnv(num_slots=6, max_steps=50)
    obs, info = env.reset(seed=123)

    dqn = DQNAgent(state_dim=env.state_dim, action_dim=env.action_space.n)
    action = dqn.select_action(obs, info, env, eval_mode=True)
    assert 0 <= action <= env.num_slots

    # Push to buffer and train step
    obs_next, r, term, trunc, info_next = env.step(action)
    dqn.replay_buffer.push(obs, action, r, obs_next, term, info_next['action_mask'])

    model_file = os.path.join(tmp_path, "dqn.pth")
    dqn.save(model_file)
    assert os.path.exists(model_file)

    restored_dqn = DQNAgent(state_dim=env.state_dim, action_dim=env.action_space.n)
    restored_dqn.load(model_file)
    action_restored = restored_dqn.select_action(obs, info, env, eval_mode=True)
    assert 0 <= action_restored <= env.num_slots
