"""Agent Comparison and Visual Evaluation Suite for ParkWise-RL."""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.env.parking_env import ParkingEnv
from src.agents.baseline_agent import NearestSlotAgent, FCFSAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent
from src.evaluation.logger import CSVLogger

def evaluate_agent(
    agent,
    num_slots: int = 12,
    num_episodes: int = 20,
    base_seed: int = 1000
) -> Tuple[Dict[str, float], np.ndarray]:
    """Evaluate a single agent over fixed evaluation seeds."""
    agent_slots = getattr(agent, 'num_slots', num_slots)
    env = ParkingEnv(num_slots=agent_slots, max_steps=500)
    episode_summaries = []
    occupancy_trajectories = []

    for ep in range(num_episodes):
        seed = base_seed + ep
        obs, info = env.reset(seed=seed)
        terminated = False
        truncated = False

        while not (terminated or truncated):
            if isinstance(agent, (QLearningAgent, DQNAgent)):
                action = agent.select_action(obs, info, env, eval_mode=True)
            else:
                action = agent.select_action(obs, info, env)
            obs, reward, terminated, truncated, info = env.step(action)

        summary = env.metrics_tracker.get_summary()
        episode_summaries.append(summary)
        occupancy_trajectories.append(np.array(env.metrics_tracker.occupancy_history))

    # Aggregate metrics across evaluation episodes
    agg_metrics = {
        'avg_total_reward': float(np.mean([s['total_reward'] for s in episode_summaries])),
        'avg_walking_distance': float(np.mean([s['avg_walking_distance'] for s in episode_summaries])),
        'avg_waiting_time': float(np.mean([s['avg_waiting_time'] for s in episode_summaries])),
        'avg_occupancy_rate': float(np.mean([s['avg_occupancy_rate'] for s in episode_summaries])),
        'avg_unnecessary_rejection_rate': float(np.mean([s['unnecessary_rejection_rate_pct'] for s in episode_summaries])),
        'total_unnecessary_rejections': int(np.sum([s['unnecessary_rejections'] for s in episode_summaries])),
    }

    mean_occ_trajectory = np.mean(occupancy_trajectories, axis=0)
    return agg_metrics, mean_occ_trajectory

def run_comparison(
    num_slots: int = 12,
    num_eval_episodes: int = 20,
    output_dir: str = "logs/comparison_charts",
    summary_csv: str = "logs/agent_comparison.csv"
):
    """Run comparative evaluation of Nearest-Slot, FCFS, Q-Learning, and DQN agents."""
    print("=== Starting ParkWise-RL Agent Comparison Evaluation ===")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)

    agents = []
    # 1. Baselines
    agents.append(NearestSlotAgent())
    agents.append(FCFSAgent())

    # 2. Q-Learning Agent (load if model exists, else initialize)
    q_agent = QLearningAgent(num_slots=num_slots)
    q_path = "models/q_table.pkl"
    if os.path.exists(q_path):
        q_agent.load(q_path)
        print(f"Loaded Q-Learning model from {q_path}")
    else:
        print(f"Warning: {q_path} not found; evaluating un-trained Q-Learning agent")
    agents.append(q_agent)

    # 3. DQN Agent (load if model exists, else initialize)
    env_temp = ParkingEnv(num_slots=num_slots)
    dqn_agent = DQNAgent(state_dim=env_temp.state_dim, action_dim=env_temp.action_space.n)
    dqn_path = "models/dqn_model.pth"
    if os.path.exists(dqn_path):
        dqn_agent.load(dqn_path)
        print(f"Loaded DQN model from {dqn_path}")
    else:
        print(f"Warning: {dqn_path} not found; evaluating un-trained DQN agent")
    agents.append(dqn_agent)

    # Collect statistics
    results = []
    occ_trajectories = {}

    for agent in agents:
        name = getattr(agent, 'name', agent.__class__.__name__)
        print(f"Evaluating {name}...")
        metrics, mean_occ = evaluate_agent(agent, num_slots=num_slots, num_episodes=num_eval_episodes)
        metrics['agent'] = name
        results.append(metrics)
        occ_trajectories[name] = mean_occ

    df_summary = pd.DataFrame(results)
    cols_order = ['agent', 'avg_total_reward', 'avg_walking_distance', 'avg_waiting_time', 'avg_occupancy_rate', 'avg_unnecessary_rejection_rate', 'total_unnecessary_rejections']
    df_summary = df_summary[cols_order]
    df_summary.to_csv(summary_csv, index=False)
    print(f"\nSummary metrics saved to {summary_csv}")
    print("\n--- Benchmark Summary Table ---")
    print(df_summary.to_string(index=False))

    # --- Plotting Visualizations ---
    palette = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71']

    # Chart 1: Avg Reward per Episode
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df_summary['agent'], df_summary['avg_total_reward'], color=palette)
    plt.title("Average Reward per Episode by Agent", fontsize=14, fontweight='bold')
    plt.ylabel("Avg Total Reward", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1.0, f"{yval:.1f}", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "01_avg_reward_comparison.png"), dpi=300)
    plt.close()

    # Chart 2: Avg Walking Distance
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df_summary['agent'], df_summary['avg_walking_distance'], color=palette)
    plt.title("Average Walking Distance to Assigned Slot", fontsize=14, fontweight='bold')
    plt.ylabel("Distance (Manhattan Units)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "02_avg_walking_distance.png"), dpi=300)
    plt.close()

    # Chart 3: Avg Waiting Time
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df_summary['agent'], df_summary['avg_waiting_time'], color=palette)
    plt.title("Average Vehicle Waiting Time Before Allocation", fontsize=14, fontweight='bold')
    plt.ylabel("Wait Time (Discrete Ticks)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "03_avg_waiting_time.png"), dpi=300)
    plt.close()

    # Chart 4: Occupancy Trajectory Over Time of Day
    plt.figure(figsize=(10, 5))
    for name, traj in occ_trajectories.items():
        plt.plot(traj, label=name, linewidth=2)
    plt.title("Parking Lot Occupancy Rate Over Simulated Day (Time of Day)", fontsize=14, fontweight='bold')
    plt.xlabel("Simulation Time Steps (Ticks)", fontsize=12)
    plt.ylabel("Occupancy Ratio", fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "04_occupancy_over_time.png"), dpi=300)
    plt.close()

    # Chart 5: Unnecessary Rejection Rate (%)
    plt.figure(figsize=(8, 5))
    bars = plt.bar(df_summary['agent'], df_summary['avg_unnecessary_rejection_rate'], color=palette)
    plt.title("Unnecessary Vehicle Rejection Rate (% turned away when free slot existed)", fontsize=13, fontweight='bold')
    plt.ylabel("Rejection Rate (%)", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "05_rejection_rate_comparison.png"), dpi=300)
    plt.close()

    # Chart 6: Joint Training Curves (if log CSVs exist)
    q_csv = "logs/q_learning_training.csv"
    dqn_csv = "logs/dqn_training.csv"
    if os.path.exists(q_csv) or os.path.exists(dqn_csv):
        plt.figure(figsize=(10, 5))
        if os.path.exists(q_csv):
            df_q = pd.read_csv(q_csv)
            # Smooth reward curve
            smooth_q = df_q['reward'].rolling(window=30, min_periods=1).mean()
            plt.plot(df_q['episode'], smooth_q, label='Q-Learning (Smooth)', color='#f1c40f', linewidth=2)
        if os.path.exists(dqn_csv):
            df_d = pd.read_csv(dqn_csv)
            smooth_d = df_d['reward'].rolling(window=30, min_periods=1).mean()
            plt.plot(df_d['episode'], smooth_d, label='DQN Agent (Smooth)', color='#2ecc71', linewidth=2)

        plt.title("Training Reward Progression: Q-Learning vs Deep Q-Network", fontsize=14, fontweight='bold')
        plt.xlabel("Training Episode", fontsize=12)
        plt.ylabel("Episode Total Reward", fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "06_joint_training_curves.png"), dpi=300)
        plt.close()

    print(f"All comparison visual charts generated successfully in {output_dir}")

if __name__ == "__main__":
    run_comparison()
