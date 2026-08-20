# ParkWise-RL
### Reinforcement Learning Based Intelligent Parking Slot Allocation and Optimization System

MCA semester project. Goal: build an RL agent that learns to allocate parking
slots to arriving vehicles better than a traditional nearest-slot heuristic,
under dynamic conditions (multiple vehicle types, reserved slots, EV charging
slots, congestion, releases over time), with a Streamlit dashboard and a
final comparison report.

This file is a build roadmap for an AI coding agent (Antigravity). Follow the
phases in order. Each phase has a definition of done. Do not skip ahead —
later phases depend on interfaces defined earlier.

---

## 0. Tech Stack

- Python 3.10+
- `gymnasium` — custom environment (`gymnasium.Env` subclass)
- `numpy` — state vectors, math
- Q-Learning: plain numpy (no library needed)
- DQN: `torch` / `stable-baselines3`
- `matplotlib` / `plotly` — training curves, comparison charts
- `streamlit` — dashboard
- `pytest` — unit tests
- `pandas` — logging episode metrics to CSV for analysis

---

## 1. Repo Structure

```
parkwise-rl/
├── README.md
├── requirements.txt
├── roadmap.md                     (this file)
├── docs/
│   ├── srs.md                     (requirements spec)
│   ├── architecture.md            (system diagram + component descriptions)
│   └── literature_review.md
├── src/
│   ├── env/
│   │   ├── parking_env.py         (custom Gymnasium environment)
│   │   ├── vehicle.py             (Vehicle class + arrival generator)
│   │   └── lot_config.py          (slot layout, types, reserved/EV slots)
│   ├── agents/
│   │   ├── q_learning_agent.py
│   │   ├── dqn_agent.py
│   │   └── baseline_agent.py      (nearest-free-slot heuristic, FCFS)
│   ├── training/
│   │   ├── train_q_learning.py
│   │   ├── train_dqn.py
│   │   └── config.py              (hyperparameters)
│   ├── evaluation/
│   │   ├── metrics.py             (distance, wait time, occupancy, reward)
│   │   ├── compare_agents.py      (RL vs baseline, produces charts)
│   │   └── logger.py              (CSV/episode logging)
│   └── dashboard/
│       └── app.py                 (Streamlit dashboard)
├── models/                        (saved Q-tables / DQN weights)
├── logs/                          (training logs, CSVs)
├── tests/
│   ├── test_env.py
│   ├── test_agents.py
│   └── test_metrics.py
├── notebooks/
│   └── exploration.ipynb          (optional, ad-hoc analysis)
├── render.yaml                    (Render deploy config — Phase 8)
└── .streamlit/
    └── config.toml                (headless server config for Render)
```
