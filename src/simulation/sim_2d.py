"""2D Animated Parking Lot Simulation for ParkWise-RL.

Renders a top-down view of the parking lot with animated vehicles
arriving, parking, and departing. Uses the existing ParkingEnv and
agent interfaces.

Usage:
    python -m src.simulation.sim_2d [--slots N] [--agent NAME] [--episodes E] [--fps F]
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.collections import PatchCollection
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.env.parking_env import ParkingEnv
from src.env.lot_config import create_lot_layout
from src.agents.baseline_agent import NearestSlotAgent, FCFSAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent

SLOT_COLORS = {
    "regular": "#d1d5db",
    "ev_charging": "#67e8f9",
    "reserved": "#c4b5fd",
    "handicapped": "#fcd34d",
}

VEHICLE_COLORS = {
    "regular": "#2563eb",
    "ev": "#059669",
    "reserved": "#7c3aed",
    "handicapped": "#d97706",
}

VEHICLE_TYPE_PROBS_ORDER = ["regular", "ev", "reserved", "handicapped"]


@dataclass
class AnimatedVehicle:
    vid: str
    vtype: str
    slot_id: int
    x: float
    y: float
    target_x: float
    target_y: float
    phase: str = "entering"
    progress: float = 0.0
    start_x: float = 0.0
    start_y: float = 0.0


@dataclass
class SimState:
    vehicles: List[AnimatedVehicle] = field(default_factory=list)
    rejected_flash: List[Tuple[float, float, float]] = field(default_factory=list)
    total_reward: float = 0.0
    allocated: int = 0
    rejected: int = 0
    departed: int = 0
    queue_len: int = 0
    occupancy: float = 0.0


def build_grid_layout(num_slots: int) -> Tuple[int, int, Dict[int, Tuple[float, float]], float, float]:
    cols = min(6, num_slots)
    rows = (num_slots + cols - 1) // cols
    slot_positions = {}
    for i in range(num_slots):
        r = i // cols
        c = i % cols
        slot_positions[i] = (float(c), float(rows - 1 - r))
    lot_width = cols
    lot_height = rows
    entrance_x = lot_width / 2.0 - 0.5
    entrance_y = -1.5
    return cols, rows, slot_positions, entrance_x, entrance_y


class ParkingSim2D:
    def __init__(self, num_slots: int = 12, agent_name: str = "Nearest-Slot", max_steps: int = 200, seed: int = 42):
        self.num_slots = num_slots
        self.agent_name = agent_name
        self.max_steps = max_steps
        self.seed = seed

        self.env = ParkingEnv(num_slots=num_slots, max_steps=max_steps)
        self.agent = self._create_agent(agent_name)

        self.cols, self.rows, self.slot_pos, self.entrance_x, self.entrance_y = build_grid_layout(num_slots)
        self.slots = self.env.slots

        self.state = SimState()
        self.frame_count = 0
        self.tick_done = False
        self.anim_active = True

        self._slot_patches = []
        self._vehicle_artists = []
        self._queue_artists = []
        self._flash_artists = []
        self._text_artists = {}
        self._arrow_patches = []
        self._legend_built = False

    def _create_agent(self, name: str):
        agents = {
            "Nearest-Slot": lambda: NearestSlotAgent(),
            "FCFS": lambda: FCFSAgent(),
            "Q-Learning": lambda: QLearningAgent(num_slots=self.num_slots),
            "DQN": lambda: DQNAgent(
                state_dim=self.env.state_dim,
                action_dim=self.env.action_space.n,
            ),
        }
        factory = agents.get(name, agents["Nearest-Slot"])
        agent = factory()
        if name == "Q-Learning" and os.path.exists("models/q_table.pkl"):
            agent.load("models/q_table.pkl")
        if name == "DQN" and os.path.exists("models/dqn_model.pth"):
            agent.load("models/dqn_model.pth")
        return agent

    def _slot_center(self, slot_id: int) -> Tuple[float, float]:
        return self.slot_pos[slot_id]

    def _entrance_pos(self) -> Tuple[float, float]:
        return self.entrance_x, self.entrance_y

    def _queue_position(self, idx: int) -> Tuple[float, float]:
        ex, ey = self._entrance_pos()
        return ex - 1.5, ey - 0.5 - idx * 0.6

    def reset(self):
        obs, info = self.env.reset(seed=self.seed)
        self.state = SimState()
        self.frame_count = 0
        self._update_state_from_env(info)
        return obs, info

    def _update_state_from_env(self, info: dict):
        self.state.queue_len = len(self.env.vehicle_queue)
        occ = sum(1 for s in self.env.slots if s.is_occupied)
        self.state.occupancy = occ / max(1, self.num_slots)
        self.state.total_reward = self.env.metrics_tracker.total_reward
        self.state.allocated = self.env.metrics_tracker.total_allocated
        self.state.rejected = self.env.metrics_tracker.total_rejections

        env_vids = set()
        for s in self.env.slots:
            if s.is_occupied and s.current_vehicle_id:
                env_vids.add(s.current_vehicle_id)

        parked = {v.vid for v in self.state.vehicles if v.phase == "parked"}
        for vid in env_vids - parked:
            slot = next((s for s in self.env.slots if s.current_vehicle_id == vid), None)
            if slot is None:
                continue
            cx, cy = self._slot_center(slot.id)
            vx = (ord(vid[-1]) % 10 - 5) * 0.05
            vy = (ord(vid[-2]) % 10 - 5) * 0.05
            self.state.vehicles.append(AnimatedVehicle(
                vid=vid,
                vtype=VEHICLE_TYPE_PROBS_ORDER[hash(vid) % 4],
                slot_id=slot.id,
                x=cx + vx,
                y=cy + vy,
                target_x=cx + vx,
                target_y=cy + vy,
                phase="parked",
            ))

        departing = [v for v in self.state.vehicles if v.phase == "departing" and v.progress >= 1.0]
        self.state.vehicles = [v for v in self.state.vehicles if v not in departing]
        self.state.departed += len(departing)

        arriving = [v for v in self.state.vehicles if v.phase == "entering" and v.progress >= 1.0]
        for v in arriving:
            v.phase = "parked"

    def step_once(self) -> bool:
        obs, info = self.env.reset(seed=self.seed + self.frame_count) if self.frame_count == 0 else (None, None)

        if self.frame_count == 0:
            obs, info = self.env.reset(seed=self.seed)
        else:
            if len(self.env.vehicle_queue) > 0:
                incoming = self.env.vehicle_queue[0]
                _, _, term, trunc, info = self.env.step(self.env.num_slots)
            else:
                _, _, term, trunc, info = self.env.step(self.env.num_slots)

        if self.frame_count >= self.max_steps:
            self.anim_active = False
            return False

        incoming = self.env.vehicle_queue[0] if len(self.env.vehicle_queue) > 0 else None
        if incoming is not None:
            if hasattr(self.agent, "eval_mode"):
                action = self.agent.select_action(
                    self.env._get_observation(), info, self.env, eval_mode=True
                )
            else:
                action = self.agent.select_action(
                    self.env._get_observation(), info, self.env
                )
            obs, reward, term, trunc, info = self.env.step(action)
        else:
            action = self.env.num_slots
            obs, reward, term, trunc, info = self.env.step(self.env.num_slots)

        self.state.total_reward = self.env.metrics_tracker.total_reward
        self.state.allocated = self.env.metrics_tracker.total_allocated
        self.state.rejected = self.env.metrics_tracker.total_rejections
        self.state.queue_len = len(self.env.vehicle_queue)
        occ = sum(1 for s in self.env.slots if s.is_occupied)
        self.state.occupancy = occ / max(1, self.num_slots)

        self._sync_vehicles(action, incoming, info)

        self.frame_count += 1
        if term or trunc:
            self.anim_active = False
            return False
        return True

    def _sync_vehicles(self, action: int, incoming, info: dict):
        env_vids = {}
        for s in self.env.slots:
            if s.is_occupied and s.current_vehicle_id:
                env_vids[s.current_vehicle_id] = s

        parked_vids = {v.vid: v for v in self.state.vehicles if v.phase == "parked"}

        for vid, slot in env_vids.items():
            if vid not in parked_vids:
                cx, cy = self._slot_center(slot.id)
                ex, ey = self._entrance_pos()
                vx = (ord(vid[-1]) % 10 - 5) * 0.04
                vy = (ord(vid[-2]) % 10 - 5) * 0.04
                vtype = incoming.vehicle_type if (incoming and incoming.id == vid) else (
                    VEHICLE_TYPE_PROBS_ORDER[hash(vid) % 4]
                )
                self.state.vehicles.append(AnimatedVehicle(
                    vid=vid,
                    vtype=vtype,
                    slot_id=slot.id,
                    x=ex,
                    y=ey,
                    target_x=cx + vx,
                    target_y=cy + vy,
                    phase="parked",
                ))
            else:
                v = parked_vids[vid]
                v.slot_id = slot.id
                cx, cy = self._slot_center(slot.id)
                vx = (ord(vid[-1]) % 10 - 5) * 0.04
                vy = (ord(vid[-2]) % 10 - 5) * 0.04
                v.target_x = cx + vx
                v.target_y = cy + vy
                v.x = v.target_x
                v.y = v.target_y

        for v in list(self.state.vehicles):
            if v.phase == "parked" and v.vid not in env_vids:
                ex, ey = self._entrance_pos()
                v.phase = "departing"
                v.start_x = v.x
                v.start_y = v.y
                v.target_x = ex
                v.target_y = ey
                v.progress = 0.0

        for v in self.state.vehicles:
            if v.phase == "departing":
                v.progress = min(1.0, v.progress + 0.15)
                v.x = v.start_x + (v.target_x - v.start_x) * v.progress
                v.y = v.start_y + (v.target_y - v.start_y) * v.progress

        if action is not None and action == self.num_slots and incoming is not None:
            ex, ey = self._entrance_pos()
            self.state.rejected_flash.append((ex, ey, 1.0))


def setup_figure(sim: ParkingSim2D):
    lot_w = sim.cols + 2
    lot_h = sim.rows + 3
    fig_w = max(12, lot_w * 1.4 + 5)
    fig_h = max(7, lot_h * 1.4 + 2)

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="#0f172a")
    ax_lot = fig.add_axes([0.02, 0.08, 0.55, 0.82])
    ax_stats = fig.add_axes([0.60, 0.50, 0.38, 0.40])
    ax_metrics = fig.add_axes([0.60, 0.08, 0.38, 0.38])

    ax_lot.set_facecolor("#0f172a")
    ax_stats.set_facecolor("#1e293b")
    ax_metrics.set_facecolor("#1e293b")

    for ax in [ax_stats, ax_metrics]:
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.set_title(ax.get_title(), color="#e2e8f0", fontsize=10, fontweight="bold", pad=8)

    return fig, ax_lot, ax_stats, ax_metrics


def draw_lot(sim: ParkingSim2D, ax):
    ax.clear()
    ax.set_facecolor("#0f172a")
    ax.set_xlim(-2.5, sim.cols + 1)
    ax.set_ylim(-2.5, sim.rows + 1)
    ax.set_aspect("equal")
    ax.axis("off")

    ax.text(
        sim.entrance_x, sim.entrance_y + 0.7,
        "ENTRANCE", ha="center", va="center",
        fontsize=9, fontweight="bold", color="#38bdf8",
    )
    ax.plot(
        [sim.entrance_x - 1, sim.entrance_x + 1],
        [sim.entrance_y + 0.2, sim.entrance_y + 0.2],
        color="#38bdf8", linewidth=2, alpha=0.6,
    )

    for slot in sim.slots:
        cx, cy = sim.slot_pos[slot.id]
        color = SLOT_COLORS.get(slot.slot_type, "#d1d5db")
        edge_color = "#facc15" if slot.is_occupied else "#475569"
        lw = 2.5 if slot.is_occupied else 1.0

        rect = mpatches.FancyBboxPatch(
            (cx - 0.4, cy - 0.35), 0.8, 0.7,
            boxstyle="round,pad=0.05",
            facecolor=color if not slot.is_occupied else "#991b1b",
            edgecolor=edge_color,
            linewidth=lw,
            alpha=0.85,
        )
        ax.add_patch(rect)

        label = f"#{slot.id}"
        type_label = slot.slot_type.replace("_", "\n")[:6]
        ax.text(cx, cy + 0.05, label, ha="center", va="center",
                fontsize=7, fontweight="bold", color="#0f172a")
        ax.text(cx, cy - 0.18, type_label, ha="center", va="center",
                fontsize=5, color="#334155")

    for v in sim.state.vehicles:
        if v.phase == "departing" and v.progress >= 1.0:
            continue
        color = VEHICLE_COLORS.get(v.vtype, "#2563eb")
        alpha = 0.3 if v.phase == "departing" else 0.95
        size = 0.25 if v.phase == "departing" else 0.30

        marker = mpatches.FancyBboxPatch(
            (v.x - size, v.y - size * 0.7), size * 2, size * 1.4,
            boxstyle="round,pad=0.03",
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            alpha=alpha,
            zorder=10,
        )
        ax.add_patch(marker)
        ax.text(v.x, v.y, v.vid[-4:], ha="center", va="center",
                fontsize=4.5, color="white", fontweight="bold", zorder=11)

    for flash in sim.state.rejected_flash:
        fx, fy, falpha = flash
        if falpha > 0:
            circle = plt.Circle((fx, fy), 0.6, color="#ef4444", alpha=falpha * 0.4, zorder=5)
            ax.add_patch(circle)
            ax.text(fx, fy, "X", ha="center", va="center",
                    fontsize=14, color="#ef4444", fontweight="bold", alpha=falpha, zorder=6)

    sim.state.rejected_flash = [
        (x, y, a - 0.15) for x, y, a in sim.state.rejected_flash if a > 0.15
    ]

    ax.text(
        sim.entrance_x, sim.rows + 0.7,
        f"Parking Lot  |  {sim.num_slots} Slots  |  Agent: {sim.agent_name}",
        ha="center", va="center", fontsize=11, fontweight="bold",
        color="#e2e8f0",
    )


def draw_stats(sim: ParkingSim2D, ax):
    ax.clear()
    ax.set_facecolor("#1e293b")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    ax.text(5, 9.5, "LIVE STATISTICS", ha="center", va="top",
            fontsize=13, fontweight="bold", color="#f1f5f9")

    stats = [
        ("Step", f"{sim.frame_count} / {sim.max_steps}"),
        ("Reward", f"{sim.state.total_reward:.1f}"),
        ("Allocated", f"{sim.state.allocated}"),
        ("Rejected", f"{sim.state.rejected}"),
        ("Departed", f"{sim.state.departed}"),
        ("Queue", f"{sim.state.queue_len}"),
        ("Occupancy", f"{sim.state.occupancy:.0%}"),
    ]

    for i, (label, value) in enumerate(stats):
        y = 8.5 - i * 1.15
        ax.text(1, y, label, ha="left", va="center",
                fontsize=10, color="#94a3b8")
        ax.text(9, y, value, ha="right", va="center",
                fontsize=12, fontweight="bold", color="#e2e8f0")

    bar_y = 0.3
    ax.barh(bar_y, sim.state.occupancy * 8, left=1, height=0.35,
            color="#3b82f6", alpha=0.8, zorder=2)
    ax.barh(bar_y, 8, left=1, height=0.35, color="#1e293b", zorder=1)
    ax.barh(bar_y, sim.state.occupancy * 8, left=1, height=0.35,
            color="#3b82f6", alpha=0.8, zorder=2)
    ax.text(5, 0.7, "Lot Fill Level", ha="center", va="center",
            fontsize=8, color="#64748b")


def draw_metrics(sim: ParkingSim2D, ax, history: list):
    ax.clear()
    ax.set_facecolor("#1e293b")
    ax.set_title("Reward & Occupancy Over Time", color="#e2e8f0",
                  fontsize=10, fontweight="bold", pad=8)

    for spine in ax.spines.values():
        spine.set_color("#334155")
    ax.tick_params(colors="#94a3b8", labelsize=7)

    if len(history) < 2:
        ax.text(0.5, 0.5, "Collecting data...", ha="center", va="center",
                transform=ax.transAxes, fontsize=10, color="#64748b")
        return

    steps = [h[0] for h in history]
    rewards = [h[1] for h in history]
    occs = [h[2] for h in history]

    ax2 = ax.twinx()
    ax.plot(steps, rewards, color="#3b82f6", linewidth=1.5, label="Reward", alpha=0.9)
    ax2.plot(steps, occs, color="#f59e0b", linewidth=1.5, label="Occupancy", alpha=0.7, linestyle="--")

    ax.set_xlabel("Step", color="#94a3b8", fontsize=8)
    ax.set_ylabel("Cumulative Reward", color="#3b82f6", fontsize=8)
    ax2.set_ylabel("Occupancy %", color="#f59e0b", fontsize=8)
    ax2.set_ylim(0, 1.1)
    ax2.tick_params(colors="#94a3b8", labelsize=7)
    for spine in ax2.spines.values():
        spine.set_color("#334155")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left",
              fontsize=7, facecolor="#1e293b", edgecolor="#334155",
              labelcolor="#e2e8f0")


def build_legend(fig):
    handles = [
        mpatches.Patch(color=SLOT_COLORS["regular"], label="Regular Slot"),
        mpatches.Patch(color=SLOT_COLORS["ev_charging"], label="EV Charging"),
        mpatches.Patch(color=SLOT_COLORS["reserved"], label="Reserved"),
        mpatches.Patch(color=SLOT_COLORS["handicapped"], label="Handicapped"),
        mpatches.Patch(color="#991b1b", label="Occupied"),
        mpatches.Patch(color=VEHICLE_COLORS["regular"], label="Regular Vehicle"),
        mpatches.Patch(color=VEHICLE_COLORS["ev"], label="EV Vehicle"),
        mpatches.Patch(color=VEHICLE_COLORS["reserved"], label="Reserved Vehicle"),
        mpatches.Patch(color=VEHICLE_COLORS["handicapped"], label="Handicapped Vehicle"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=5,
        fontsize=7, facecolor="#1e293b", edgecolor="#334155",
        labelcolor="#e2e8f0", framealpha=0.9,
    )


def run_simulation(num_slots: int = 12, agent_name: str = "Nearest-Slot",
                   max_steps: int = 200, seed: int = 42, interval: int = 200):
    sim = ParkingSim2D(
        num_slots=num_slots,
        agent_name=agent_name,
        max_steps=max_steps,
        seed=seed,
    )

    fig, ax_lot, ax_stats, ax_metrics = setup_figure(sim)
    build_legend(fig)
    fig.text(0.5, 0.97, "ParkWise-RL  |  2D Parking Lot Simulation",
             ha="center", va="top", fontsize=15, fontweight="bold",
             color="#f1f5f9")

    sim.reset()
    history = [(0, 0.0, 0.0)]

    def animate(frame):
        nonlocal history
        if sim.anim_active:
            sim.step_once()
            history.append((sim.frame_count, sim.state.total_reward, sim.state.occupancy))

        draw_lot(sim, ax_lot)
        draw_stats(sim, ax_stats)
        draw_metrics(sim, ax_metrics, history)
        fig.canvas.draw_idle()
        return []

    ani = animation.FuncAnimation(
        fig, animate, frames=max_steps + 10,
        interval=interval, repeat=False, blit=False,
    )

    plt.show()
    return ani


def main():
    parser = argparse.ArgumentParser(description="ParkWise-RL 2D Parking Simulation")
    parser.add_argument("--slots", type=int, default=12, help="Number of parking slots")
    parser.add_argument("--agent", type=str, default="Nearest-Slot",
                        choices=["Nearest-Slot", "FCFS", "Q-Learning", "DQN"],
                        help="Allocation agent")
    parser.add_argument("--steps", type=int, default=200, help="Max simulation steps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fps", type=int, default=5, help="Frames per second")
    args = parser.parse_args()

    interval = max(1, 1000 // args.fps)
    run_simulation(
        num_slots=args.slots,
        agent_name=args.agent,
        max_steps=args.steps,
        seed=args.seed,
        interval=interval,
    )


if __name__ == "__main__":
    main()
