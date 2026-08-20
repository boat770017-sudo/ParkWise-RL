# ParkWise-RL
### Reinforcement Learning Based Intelligent Parking Slot Allocation and Optimization System

ParkWise-RL is a Reinforcement Learning application developed for an MCA semester project. It models dynamic parking slot allocation with multiple vehicle types (Regular, EV, Reserved, Handicapped), dynamic arrival/departure processes, dynamic congestion levels, and custom reward metrics.

## Features
- **Custom Gymnasium Environment**: `ParkingEnv` modeling grid layouts, slot types, Manhattan walking distance, Poisson arrival rates, and vehicle departure durations.
- **Multiple Agents**:
  - `NearestSlotAgent`: Greedily assigns closest free compatible slot.
  - `FCFSAgent`: First-Come-First-Served slot selection.
  - `QLearningAgent`: Tabular Q-learning with discrete state representation.
  - `DQNAgent`: Deep Q-Network utilizing PyTorch with action masking for larger parking grids.
- **Evaluation & Visualizations**: Automated comparison script producing comparative bar charts, line plots, and CSV summaries.
- **Interactive Streamlit Dashboard**: Live simulation visualization, metric KPIs, parking lot grid color-coding, and step-by-step playback.
- **Cloud Deployment Ready**: Ready for Render deployment with `render.yaml` and headless `.streamlit/config.toml`.

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Run Tests
```bash
pytest tests/
```

### 3. Train Agents
```bash
python src/training/train_q_learning.py
python src/training/train_dqn.py
```

### 4. Run Agent Comparison
```bash
python src/evaluation/compare_agents.py
```

### 5. Launch Dashboard
```bash
streamlit run src/dashboard/app.py
```
