"""Training script for Tabular Q-Learning Agent on ParkingEnv."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.env.parking_env import ParkingEnv
from src.agents.q_learning_agent import QLearningAgent
from src.training.config import (
    Q_LEARNING_EPISODES,
    Q_LEARNING_ALPHA,
    Q_LEARNING_GAMMA,
    Q_LEARNING_EPSILON_START,
    Q_LEARNING_EPSILON_END,
    Q_LEARNING_EPSILON_DECAY
)

def train_q_learning(
    num_slots: int = 6,
    num_episodes: int = Q_LEARNING_EPISODES,
    save_model_path: str = "models/q_table.pkl",
    log_csv_path: str = "logs/q_learning_training.csv"
) -> QLearningAgent:
    """Train tabular Q-Learning agent and save trained weights."""
    print(f"--- Starting Tabular Q-Learning Training ({num_episodes} episodes, {num_slots} slots) ---")
    env = ParkingEnv(num_slots=num_slots, max_steps=500)
    agent = QLearningAgent(
        num_slots=num_slots,
        alpha=Q_LEARNING_ALPHA,
        gamma=Q_LEARNING_GAMMA,
        epsilon_start=Q_LEARNING_EPSILON_START,
        epsilon_end=Q_LEARNING_EPSILON_END,
        epsilon_decay=Q_LEARNING_EPSILON_DECAY
    )

    history = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        state_key = agent.discretize_state(info, env)
        terminated = False
        truncated = False
        total_reward = 0.0

        while not (terminated or truncated):
            mask = info['action_mask']
            action = agent.select_action(obs, info, env, eval_mode=False)

            obs_next, reward, terminated, truncated, info_next = env.step(action)
            next_state_key = agent.discretize_state(info_next, env)
            next_mask = info_next['action_mask']

            agent.update(
                state_key=state_key,
                action=action,
                reward=reward,
                next_state_key=next_state_key,
                done=(terminated or truncated),
                next_mask=next_mask
            )

            state_key = next_state_key
            obs = obs_next
            info = info_next
            total_reward += reward

        agent.decay_epsilon()
        ep_summary = env.metrics_tracker.get_summary()
        history.append({
            'episode': ep,
            'reward': total_reward,
            'epsilon': agent.epsilon,
            'avg_wait_time': ep_summary['avg_waiting_time'],
            'avg_distance': ep_summary['avg_walking_distance'],
            'unnecessary_rejections': ep_summary['unnecessary_rejections']
        })

        if ep % 100 == 0:
            avg_rew = np.mean([h['reward'] for h in history[-100:]])
            print(f"Episode {ep}/{num_episodes} | Avg Reward (last 100): {avg_rew:.2f} | Epsilon: {agent.epsilon:.3f}")

    # Save trained model
    agent.save(save_model_path)
    print(f"Q-Table successfully saved to {save_model_path}")

    # Log training history to CSV
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(log_csv_path, index=False)
    print(f"Training history saved to {log_csv_path}")

    return agent

if __name__ == "__main__":
    train_q_learning()
