"""ParkWise-RL Streamlit Interactive Dashboard."""

import os
import sys
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add workspace root to python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.env.parking_env import ParkingEnv
from src.agents.baseline_agent import NearestSlotAgent, FCFSAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent
from src.evaluation.compare_agents import evaluate_agent

st.set_page_config(
    page_title="ParkWise-RL Dashboard",
    page_icon="🅿️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium UI aesthetics
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetricCard {
        background-color: #1f2937;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .grid-container {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
        gap: 12px;
        margin-top: 15px;
        margin-bottom: 20px;
    }
    .slot-card {
        border-radius: 10px;
        padding: 12px;
        text-align: center;
        color: white;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        transition: transform 0.2s ease-in-out;
    }
    .slot-card:hover {
        transform: scale(1.05);
    }
    .slot-regular-free { background: linear-gradient(135deg, #10b981, #059669); }
    .slot-ev-free { background: linear-gradient(135deg, #06b6d4, #0891b2); }
    .slot-reserved-free { background: linear-gradient(135deg, #8b5cf6, #6d28d9); }
    .slot-handicapped-free { background: linear-gradient(135deg, #f59e0b, #d97706); }
    .slot-occupied { background: linear-gradient(135deg, #ef4444, #dc2626); }
    .slot-selected { border: 4px solid #facc15; }
    .vehicle-banner {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #38bdf8;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_agents(num_slots: int):
    """Cached loader for RL models."""
    agents = {
        "Nearest-Slot Baseline": NearestSlotAgent(),
        "FCFS Baseline": FCFSAgent()
    }
    
    q_agent = QLearningAgent(num_slots=num_slots)
    q_path = "models/q_table.pkl"
    if os.path.exists(q_path):
        try:
            q_agent.load(q_path)
        except Exception:
            pass
    agents["Q-Learning Agent"] = q_agent

    env_temp = ParkingEnv(num_slots=num_slots)
    dqn_agent = DQNAgent(state_dim=env_temp.state_dim, action_dim=env_temp.action_space.n)
    dqn_path = "models/dqn_model.pth"
    if os.path.exists(dqn_path):
        try:
            dqn_agent.load(dqn_path)
        except Exception:
            pass
    agents["DQN Agent"] = dqn_agent

    return agents

# Header Banner
st.title("🅿️ ParkWise-RL")
st.markdown("### Reinforcement Learning Based Intelligent Parking Slot Allocation & Optimization")

# Sidebar Configuration
st.sidebar.header("⚙️ Simulation Controls")
num_slots = st.sidebar.selectbox("Parking Lot Slots (N)", [6, 12, 20], index=1)
agents_dict = load_agents(num_slots)

selected_agent_name = st.sidebar.selectbox(
    "Select Allocation Agent",
    list(agents_dict.keys()),
    index=3  # Default to DQN Agent
)
agent = agents_dict[selected_agent_name]

sim_speed = st.sidebar.slider("Simulation Speed (Delay sec)", 0.01, 0.5, 0.1)

# Initialize Session State Environment
if "env" not in st.session_state or st.session_state.env.num_slots != num_slots:
    st.session_state.env = ParkingEnv(num_slots=num_slots, max_steps=500)
    st.session_state.obs, st.session_state.info = st.session_state.env.reset(seed=42)
    st.session_state.last_action = None
    st.session_state.step_history = []

env: ParkingEnv = st.session_state.env

# Action Buttons
col_btn1, col_btn2, col_btn3 = st.sidebar.columns(3)
with col_btn1:
    if st.button("Reset 🔄"):
        st.session_state.obs, st.session_state.info = env.reset()
        st.session_state.last_action = None
        st.session_state.step_history = []

with col_btn2:
    step_clicked = st.button("Step ⏩")

with col_btn3:
    run_day_clicked = st.button("Run Day 🚀")

# Execute single step logic
def execute_step():
    obs = st.session_state.obs
    info = st.session_state.info
    if hasattr(agent, 'select_action'):
        action = agent.select_action(obs, info, env, eval_mode=True)
    else:
        action = agent.select_action(obs, info, env)

    st.session_state.last_action = action
    obs_next, reward, term, trunc, info_next = env.step(action)
    st.session_state.obs = obs_next
    st.session_state.info = info_next
    st.session_state.step_history.append((env.current_step, reward, sum(1 for s in env.slots if s.is_occupied)))

if step_clicked:
    execute_step()

if run_day_clicked:
    progress_bar = st.progress(0)
    status_text = st.empty()
    while env.current_step < env.max_steps:
        execute_step()
        progress_bar.progress(env.current_step / env.max_steps)
        status_text.text(f"Simulating tick {env.current_step} / {env.max_steps}")
        time.sleep(sim_speed)
    status_text.text("Day simulation completed! 🎉")

# KPI Summary Cards
metrics_summary = env.metrics_tracker.get_summary()
occupied_count = sum(1 for s in env.slots if s.is_occupied)
free_count = num_slots - occupied_count

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("🅿️ Total Slots", f"{num_slots}")
col2.metric("🚗 Occupied", f"{occupied_count}", delta=f"{(occupied_count/num_slots)*100:.0f}% Occ")
col3.metric("🟢 Available", f"{free_count}")
col4.metric("🚶‍♂️ Avg Walking Dist", f"{metrics_summary['avg_walking_distance']:.2f}")
col5.metric("📈 Episode Reward", f"{metrics_summary['total_reward']:.1f}")

st.markdown("---")

# Main Content Layout: Left Grid & Incoming Vehicle / Right Metrics Graphs
left_col, right_col = st.columns([3, 2])

with left_col:
    st.subheader("🚘 Incoming Vehicle Context")
    incoming_v = st.session_state.info.get('incoming_vehicle')
    if incoming_v:
        st.markdown(f"""
        <div class="vehicle-banner">
            🎯 <b>Incoming Vehicle ID:</b> {incoming_v.id} &nbsp;|&nbsp; 
            🚙 <b>Type:</b> {incoming_v.vehicle_type.upper()} &nbsp;|&nbsp; 
            ⚡ <b>Requires EV:</b> {'Yes' if incoming_v.requires_ev else 'No'} &nbsp;|&nbsp; 
            ♿ <b>Handicapped:</b> {'Yes' if incoming_v.requires_handicapped else 'No'} &nbsp;|&nbsp; 
            ⏱️ <b>Wait Time:</b> {incoming_v.wait_time} ticks
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No incoming vehicle waiting at entrance queue.")

    if st.session_state.last_action is not None:
        act = st.session_state.last_action
        if act == num_slots:
            st.warning("🎯 **Agent Action Choice:** REJECT / Turn Away Vehicle")
        else:
            st.success(f"🎯 **Agent Action Choice:** Allocate to Slot #{act} (Dist: {env.slots[act].distance_from_entrance:.1f})")

    st.subheader("🅿️ Live Parking Lot Grid State")
    grid_cols = st.columns(6)
    for i, slot in enumerate(env.slots):
        c_idx = i % 6
        with grid_cols[c_idx]:
            is_selected = (st.session_state.last_action == i)
            selected_class = "slot-selected" if is_selected else ""

            if slot.is_occupied:
                card_class = "slot-occupied"
                status_str = f"🔴 OCCUPIED ({slot.current_vehicle_id})"
            else:
                if slot.slot_type == 'ev_charging':
                    card_class = "slot-ev-free"
                elif slot.slot_type == 'reserved':
                    card_class = "slot-reserved-free"
                elif slot.slot_type == 'handicapped':
                    card_class = "slot-handicapped-free"
                else:
                    card_class = "slot-regular-free"
                status_str = f"🟢 FREE ({slot.slot_type.upper()})"

            st.markdown(f"""
            <div class="slot-card {card_class} {selected_class}">
                <div style="font-size:16px;">Slot #{slot.id}</div>
                <div style="font-size:11px;">Dist: {slot.distance_from_entrance:.1f}</div>
                <div style="font-size:10px; margin-top:4px;">{status_str}</div>
            </div>
            """, unsafe_allow_html=True)

with right_col:
    st.subheader("📊 Performance Analytics")
    if len(st.session_state.step_history) > 0:
        df_hist = pd.DataFrame(st.session_state.step_history, columns=['step', 'reward', 'occupied'])
        fig_occ = px.line(df_hist, x='step', y='occupied', title="Occupancy Level Over Step Ticks")
        fig_occ.update_layout(height=220, margin=dict(l=20, r=20, t=35, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_occ, use_container_width=True)

        df_hist['cum_reward'] = df_hist['reward'].cumsum()
        fig_rew = px.line(df_hist, x='step', y='cum_reward', title="Cumulative Reward Progression")
        fig_rew.update_layout(height=220, margin=dict(l=20, r=20, t=35, b=20), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_rew, use_container_width=True)
    else:
        st.info("Step simulation to view live metric trajectories.")

# Full-Day Comparison Tab
st.markdown("---")
st.subheader("⚖️ Agent Benchmark Comparative Evaluation")
if st.button("Run Full Day Agent Comparison 📊"):
    with st.spinner("Simulating all agents over evaluation day..."):
        summary_rows = []
        for name, ag in agents_dict.items():
            metrics, _ = evaluate_agent(ag, num_slots=num_slots, num_episodes=5)
            metrics['agent'] = name
            summary_rows.append(metrics)
        
        df_comp = pd.DataFrame(summary_rows)

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_bar_rew = px.bar(df_comp, x='agent', y='avg_total_reward', title="Avg Total Reward (RL vs Baseline)", color='agent')
            st.plotly_chart(fig_bar_rew, use_container_width=True)

        with col_c2:
            fig_bar_rej = px.bar(df_comp, x='agent', y='avg_unnecessary_rejection_rate', title="Unnecessary Rejection Rate (%)", color='agent')
            st.plotly_chart(fig_bar_rej, use_container_width=True)

        st.dataframe(df_comp[['agent', 'avg_total_reward', 'avg_walking_distance', 'avg_waiting_time', 'avg_unnecessary_rejection_rate']], use_container_width=True)
