"""Training script for Deep Q-Network (DQN) Agent on ParkingEnv."""

import os
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.env.parking_env import ParkingEnv
from src.agents.dqn_agent import DQNAgent
from src.training.config import (
    DQN_EPISODES,
    DQN_LR,
    DQN_GAMMA,
    DQN_BATCH_SIZE,
    DQN_TARGET_UPDATE_FREQ
)

def train_dqn(
    num_slots: int = 12,
    num_episodes: int = DQN_EPISODES,
    save_model_path: str = "models/dqn_model.pth",
    log_csv_path: str = "logs/dqn_training.csv"
) -> DQNAgent:
    """Train Deep Q-Network agent on larger lot grid layout and save weights."""
    print(f"--- Starting PyTorch DQN Training ({num_episodes} episodes, {num_slots} slots) ---", flush=True)
    env = ParkingEnv(num_slots=num_slots, max_steps=500)
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_space.n,
        lr=DQN_LR,
        gamma=DQN_GAMMA
    )

    history = []
    global_step = 0

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        losses = []

        while not (terminated or truncated):
            global_step += 1
            mask = info['action_mask']
            action = agent.select_action(obs, info, env, eval_mode=False)

            obs_next, reward, terminated, truncated, info_next = env.step(action)
            next_mask = info_next['action_mask']

            # Store in replay buffer
            agent.replay_buffer.push(
                obs=obs,
                action=action,
                reward=reward,
                next_obs=obs_next,
                done=(terminated or truncated),
                next_mask=next_mask
            )

            # Perform training step
            loss = agent.train_step(batch_size=DQN_BATCH_SIZE)
            if loss is not None:
                losses.append(loss)

            # Periodically update target network
            if global_step % DQN_TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

            obs = obs_next
            info = info_next
            total_reward += reward

        agent.decay_epsilon()
        ep_summary = env.metrics_tracker.get_summary()
        avg_loss = float(np.mean(losses)) if losses else 0.0

        history.append({
            'episode': ep,
            'reward': total_reward,
            'epsilon': agent.epsilon,
            'loss': avg_loss,
            'avg_wait_time': ep_summary['avg_waiting_time'],
            'avg_distance': ep_summary['avg_walking_distance'],
            'unnecessary_rejections': ep_summary['unnecessary_rejections']
        })

        if ep % 20 == 0:
            avg_rew = np.mean([h['reward'] for h in history[-20:]])
            print(f"Episode {ep}/{num_episodes} | Avg Reward (last 20): {avg_rew:.2f} | Epsilon: {agent.epsilon:.3f} | Loss: {avg_loss:.4f}", flush=True)

    # Save trained network
    agent.save(save_model_path)
    print(f"DQN weights successfully saved to {save_model_path}", flush=True)

    # Export CSV history
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    df = pd.DataFrame(history)
    df.to_csv(log_csv_path, index=False)
    print(f"Training history saved to {log_csv_path}", flush=True)

    return agent

if __name__ == "__main__":
    train_dqn()
