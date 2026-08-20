# Software Requirements Specification (SRS)
## ParkWise-RL: Reinforcement Learning Based Intelligent Parking Slot Allocation System

---

## 1. Introduction

### 1.1 Purpose
This document specifies the software requirements for **ParkWise-RL**, an intelligent parking slot allocation and optimization system developed as an MCA semester project. ParkWise-RL leverages Reinforcement Learning (`gymnasium`, `numpy`, PyTorch / tabular Q-Learning) to dynamically allocate incoming vehicles to optimal parking slots in real time.

### 1.2 Scope
The system handles multi-category parking lot environments containing regular slots, EV charging slots, reserved slots, and handicapped-accessible slots. It models dynamic vehicle arrival processes (Poisson distribution with rush-hour peaks) and variable parking duration releases. The system outperforms conventional greedy heuristics (e.g. Nearest-Slot, First-Come-First-Served) across walking distance, queue wait times, slot compatibility, and rejection rates.

---

## 2. System Overview & Actors

### 2.1 Actors
- **Parking System Operator / Admin**: Monitors lot state, selects agent policies, triggers evaluation benchmarks, and views visual analytics via the dashboard.
- **Incoming Vehicle Driver (Simulated User)**: Enters the system with specific vehicle traits (Regular, EV, Reserved, Handicapped) requesting an optimal slot allocation.
- **RL Agent (Decision Maker)**: Observes environment state vector and selects allocation action or rejection.

---

## 3. Functional Requirements

### 3.1 Environment & Simulation Engine
- **FR-1.1 Layout Customization**: System shall support configurable grid sizes ($N = 6, 12, 20$) with precomputed Manhattan walking distances from entrance $(0,0)$.
- **FR-1.2 Multi-Type Slots**: Slots must support categorical types (`regular`, `ev_charging`, `reserved`, `handicapped`).
- **FR-1.3 Dynamic Arrival Generator**: Vehicles must arrive according to a Poisson process with time-dependent arrival rates $\lambda(t)$ simulating morning/evening rush hour peaks.
- **FR-1.4 Departure Engine**: System must track vehicle parking duration and automatically free slots when parking time expires.

### 3.2 State, Action, & Reward Engine
- **FR-2.1 State Vector Encoding**: The environment state vector must be a flat $1\text{D}$ float32 array of shape $(6N + 6)$ encoding slot occupancy, slot type one-hot vectors, incoming vehicle type, distances to free compatible slots, overall congestion ratio, and normalized time of day.
- **FR-2.2 Action Space & Masking**: Action space shall be `Discrete(N + 1)` (slots $0 \dots N-1$ plus Reject action $N$). Action masking must invalidate occupied or type-incompatible slots.
- **FR-2.3 Multi-Objective Reward Function**: Reward function must incorporate valid allocation bonuses (+10), walking distance thresholds (+5 near, -3 far), excess wait penalties (-5), invalid attempt penalties (-10), unnecessary rejection penalties (-8), and specialty slot match bonuses (+2).

### 3.3 Agents & Training
- **FR-3.1 Baseline Heuristics**: Nearest-Slot (greedy distance) and FCFS baseline agents for benchmarking.
- **FR-3.2 Tabular Q-Learning**: Q-learning agent with compact state feature discretization.
- **FR-3.3 Deep Q-Network (DQN)**: PyTorch MLP DQN agent with target network and experience replay buffer for scaling to larger lot grids.

### 3.4 Evaluation & Visualization
- **FR-4.1 Benchmark Evaluation**: Script to run all agents on identical random evaluation seeds.
- **FR-4.2 Chart Generation**: Automated export of comparison plots (Reward, Distance, Wait Time, Occupancy Trajectory, Rejections, Training Curves).
- **FR-4.3 Interactive Streamlit Dashboard**: Web-based UI providing metric KPI cards, color-coded parking grid, live step simulation, and agent benchmark tab.

---

## 4. Non-Functional Requirements

- **NFR-1 Response Time**: Agent action selection must execute within $< 20\text{ ms}$ per step.
- **NFR-2 Reliability & Reproducibility**: Simulation environment must support deterministic random seeding for repeatable evaluations.
- **NFR-3 Deployability**: Cloud deployable on Render using `render.yaml` and headless `.streamlit/config.toml`.
- **NFR-4 Scalability**: Architecture must support expansion from small (6 slots) to standard (12-20 slots) layouts without breaking environment contracts.
