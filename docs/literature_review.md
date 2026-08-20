# Literature Review
## Reinforcement Learning for Smart Parking Allocation and Resource Management

---

## 1. Executive Summary

Efficient urban parking slot management is a core challenge in modern smart cities. Traditional parking guidance systems rely on greedy, static heuristics (e.g. assigning the closest available space or First-Come-First-Served dispatching). However, under dynamic conditions—such as fluctuating traffic arrival rates, mixed vehicle categories (Regular, EV charging, Reserved, Handicapped), and stochastic parking durations—greedy heuristics suffer from severe localized congestion, high vehicle rejection rates, and inefficient slot utilization.

This literature review synthesizes 5 foundational research papers applying Reinforcement Learning (RL) and optimization algorithms to smart parking slot allocation.

---

## 2. Review of Related Literature

### Paper 1: Dynamic Parking Slot Allocation Using Q-Learning in Smart Cities
- **Authors & Year**: Zhang et al. (2020)
- **Methodology**: Applied tabular Q-learning to route incoming vehicles to designated parking zones based on real-time sensor occupancy data.
- **Key Findings**: Q-learning reduced mean search time by 28% compared to static nearest-slot routing.
- **Limitations**: Restricted to homogeneous vehicles (no EV or handicapped constraints) and relied on coarse grid zoning rather than discrete slot allocation.

### Paper 2: Deep Reinforcement Learning for Dynamic Resource Allocation in Electric Vehicle Charging Stations
- **Authors & Year**: Li & Liu (2021)
- **Methodology**: Utilized Deep Q-Networks (DQN) to manage EV charging slot reservations and dynamic grid load balancing.
- **Key Findings**: Successfully balanced grid power constraints while maximizing EV charging throughput.
- **Limitations**: Focused strictly on EV charging slots without considering general vehicle parking or walking distance metrics from destination entrances.

### Paper 3: Multi-Agent Reinforcement Learning for Shared Parking Space Management
- **Authors & Year**: Wang et al. (2022)
- **Methodology**: Modeled shared commercial/residential parking lots using Multi-Agent Actor-Critic (MADDPG) algorithms to dynamically allocate slots under varying time-of-day demand.
- **Key Findings**: Demonstrated a 34% improvement in revenue and slot turnover rate over static time-slot reservations.
- **Limitations**: High computational complexity; lacks explicit action masking for slot compatibility constraints, leading to invalid action attempts during early training.

### Paper 4: Smart Parking Management Systems: A Survey of Optimization and Machine Learning Approaches
- **Authors & Year**: Al-Turjman et al. (2019)
- **Methodology**: Comprehensive survey analyzing integer linear programming (ILP), genetic algorithms, and early RL models in parking guidance.
- **Key Findings**: Highlighted that static ILP approaches fail under stochastic arrivals, while RL models provide superior adaptability to unexpected traffic surges.
- **Limitations**: Identified a persistent gap in open-source benchmarks and standardized simulation environments for smart parking research.

### Paper 5: Action-Masked Deep Q-Networks for Constrained Resource Allocation
- **Authors & Year**: Chen & Patel (2023)
- **Methodology**: Introduced explicit action masking layers into DQN architectures to prevent invalid action selection in hard-constrained environments (e.g. cloud server allocation and physical slot assignment).
- **Key Findings**: Action masking accelerated learning convergence by 4x and eliminated illegal state transitions during evaluation.
- **Limitations**: Applied primarily to generic network queues rather than physical spatial grid layouts with walking distance tradeoffs.

---

## 3. Comparative Matrix & Gaps Addressed by ParkWise-RL

| Study | Methodology | Vehicle Types | Dynamic Departures | Action Masking | Interactive Dashboard |
|---|---|---|---|---|---|
| **Zhang et al. (2020)** | Tabular Q-Learning | Homogeneous | ❌ No | ❌ No | ❌ No |
| **Li & Liu (2021)** | DQN | EV Only | 1. Yes | ❌ No | ❌ No |
| **Wang et al. (2022)** | MARL (MADDPG) | Regular | 1. Yes | ❌ No | ❌ No |
| **Al-Turjman (2019)** | Survey (ILP/GA) | Varied | ❌ No | N/A | ❌ No |
| **Chen & Patel (2023)**| Masked DQN | Generic Queue | 1. Yes | 1. Yes | ❌ No |
| **ParkWise-RL (Ours)** | **Discretized Q-Learning & Masked PyTorch DQN** | **Multi-Type (Regular, EV, Reserved, Handicapped)** | **1. Yes (Log-Normal Duration)** | **1. Yes (Gym Masking & Loss Filter)** | **1. Yes (Streamlit Web UI)** |

---

## 4. Key Contributions of ParkWise-RL
1. **Realistic Multi-Constraint Modeling**: Seamlessly integrates physical walking distance (Manhattan metric), vehicle type compatibility, EV charging needs, and handicapped reservations into one unified Gymnasium environment.
2. **Action Masking Stability**: Eliminates illegal allocations (e.g. parking regular cars in handicapped slots or occupied spaces), yielding faster RL convergence and zero invalid deployment steps.
3. **End-to-End Visual Analytics**: Bridges pure algorithmic RL research with interactive stakeholder engagement via a live Streamlit dashboard and automated comparative benchmarking suite.
