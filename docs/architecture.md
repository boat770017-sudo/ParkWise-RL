# System Architecture Specification
## ParkWise-RL: Intelligent Parking Slot Allocation Engine

---

## 1. High-Level Architecture Diagram

```
+-------------------------------------------------------------------------+
|                        Streamlit Dashboard (app.py)                     |
|     [KPI Cards] [Interactive Grid] [Performance Charts] [Agent Choice]   |
+------------------------------------+------------------------------------+
                                     |
                                     v
+------------------------------------+------------------------------------+
|                         Agent Policy Interface                          |
|    +------------------+------------------+-----------------+-----------+|
|    | Nearest-Slot     | FCFS Baseline    | Q-Learning      | PyTorch   ||
|    | Heuristic        | Heuristic        | (Discretized)   | DQN Agent ||
|    +------------------+------------------+-----------------+-----------+|
+------------------------------------+------------------------------------+
                                     | Selects Action a in Discrete(N+1)
                                     v
+------------------------------------+------------------------------------+
|                  Gymnasium Environment (ParkingEnv)                     |
|  - Layout & Slot Metadata (lot_config.py)                               |
|  - Poisson Arrival Generator (vehicle.py)                               |
|  - Vehicle Departure Queue Manager                                      |
|  - Action Masking Filter                                                |
+------------------------------------+------------------------------------+
                                     | Emits (Obs Vector, Reward, Info)
                                     v
+------------------------------------+------------------------------------+
|                   Metrics & Logging Engine                              |
|  - Reward Calculator & Tracker (metrics.py)                             |
|  - CSV Logger & Plotting Pipeline (compare_agents.py, logger.py)        |
+-------------------------------------------------------------------------+
```

---

## 2. Environment Specifications

### 2.1 State Vector Representation
For a parking lot with $N$ slots, the state observation is a flat $1\text{D}$ array of length $6N + 6$:

$$\text{State Vector} = \begin{bmatrix}
\mathbf{O}_{N} \\
\mathbf{T}_{4N} \\
\mathbf{V}_{4} \\
\mathbf{D}_{N} \\
c \\
\tau
\end{bmatrix}$$

Where:
- $\mathbf{O}_N \in \{0, 1\}^N$: Binary occupancy state for slots $1 \dots N$.
- $\mathbf{T}_{4N} \in \{0, 1\}^{4N}$: One-hot type encoding for each slot (Regular, EV, Reserved, Handicapped).
- $\mathbf{V}_4 \in \{0, 1\}^4$: One-hot type encoding of currently arriving vehicle (all zeros if queue empty).
- $\mathbf{D}_N \in \mathbb{R}^N$: Precomputed Manhattan distance from entrance to free compatible slots ($0$ if occupied or incompatible).
- $c = \frac{\text{Occupied Slots}}{N} \in [0, 1]$: Current congestion ratio.
- $\tau = \frac{t}{T_{\text{max}}} \in [0, 1]$: Normalized step time of day.

### 2.2 Action Space & Masking
Action space is discrete with $N + 1$ actions:
- Actions $0 \le a \le N-1$: Allocate vehicle to slot $a$.
- Action $a = N$: Reject / No Allocation action.

Valid action mask $M \in \{0, 1\}^{N+1}$ enforces physical and policy constraints:
$$M(a) = \begin{cases} 
1 & \text{if } a = N \text{ (Reject is always valid if vehicle present)} \\
1 & \text{if } a < N \land \text{Slot } a \text{ is un-occupied } \land \text{Slot } a \text{ is type-compatible with vehicle} \\
0 & \text{otherwise}
\end{cases}$$

### 2.3 Reward Structure
$$\mathcal{R}(s, a) = R_{\text{alloc}} + R_{\text{dist}} + R_{\text{wait}} + R_{\text{invalid}} + R_{\text{unnecessary\_rej}} + R_{\text{bonus}}$$

- $R_{\text{alloc}} = +10.0$ for valid allocation.
- $R_{\text{dist}} = +5.0$ if $d \le 3.0$, else $-3.0$ if $d > 7.0$.
- $R_{\text{wait}} = -5.0$ if vehicle wait time $> 2$ ticks.
- $R_{\text{invalid}} = -10.0$ for occupied or type-incompatible allocation attempt.
- $R_{\text{unnecessary\_rej}} = -8.0$ if agent rejects vehicle despite a compatible slot existing.
- $R_{\text{bonus}} = +2.0$ for EV in EV slot or Handicapped in Handicapped slot.

---

## 3. Agent Architectures

### 3.1 Tabular Q-Learning (Small Lots)
Discretizes state into tuple: `(congestion_bucket [0..4], vehicle_type [0..4], slot_type_occupancy_counts [4])`. Updates Q-table via:
$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a' \in M(s')} Q(s', a') - Q(s, a) \right]$$

### 3.2 PyTorch Deep Q-Network (Standard / Large Lots)
- Multi-Layer Perceptron: `Input(6N+6)` $\rightarrow$ `Linear(128)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(128)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(N+1)`.
- Uses Target Network updated every 100 steps and Experience Replay Buffer (capacity 20,000) for training stability.
