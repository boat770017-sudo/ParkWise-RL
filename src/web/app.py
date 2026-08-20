"""FastAPI backend for ParkWise-RL web interface.

Provides REST endpoints for simulation control and a WebSocket
for streaming real-time 2D simulation state to the browser.

Usage:
    uvicorn src.web.app:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import asyncio
import json
import inspect
from typing import Optional
from dataclasses import dataclass, asdict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.env.parking_env import ParkingEnv
from src.agents.baseline_agent import NearestSlotAgent, FCFSAgent
from src.agents.q_learning_agent import QLearningAgent
from src.agents.dqn_agent import DQNAgent

app = FastAPI(title="ParkWise-RL")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

SLOT_COLORS = {
    "regular": "#94a3b8",
    "ev_charging": "#22d3ee",
    "reserved": "#a78bfa",
    "handicapped": "#fbbf24",
}
VEHICLE_COLORS = {
    "regular": "#3b82f6",
    "ev": "#10b981",
    "reserved": "#8b5cf6",
    "handicapped": "#f59e0b",
}
AGENT_FACTORIES = {
    "Nearest-Slot": lambda n: NearestSlotAgent(),
    "FCFS": lambda n: FCFSAgent(),
    "Q-Learning": lambda n: QLearningAgent(num_slots=n),
    "DQN": lambda n: DQNAgent(state_dim=ParkingEnv(num_slots=n).state_dim, action_dim=n + 1),
}


def _load_agent(name: str, num_slots: int):
    agent = AGENT_FACTORIES[name](num_slots)
    if name == "Q-Learning" and os.path.exists("models/q_table.pkl"):
        agent.load("models/q_table.pkl")
    if name == "DQN" and os.path.exists("models/dqn_model.pth"):
        agent.load("models/dqn_model.pth")
    return agent


def _select_action(agent, obs, info, env):
    if "eval_mode" in inspect.signature(agent.select_action).parameters:
        return agent.select_action(obs, info, env, eval_mode=True)
    return agent.select_action(obs, info, env)


def _build_grid(num_slots: int):
    cols = min(6, num_slots)
    rows = (num_slots + cols - 1) // cols
    positions = {}
    for i in range(num_slots):
        r, c = divmod(i, cols)
        positions[i] = [c, rows - 1 - r]
    return {
        "cols": cols,
        "rows": rows,
        "slots": positions,
        "entrance": [cols / 2.0 - 0.5, -1.5],
    }


def _env_state(env: ParkingEnv, grid: dict, action=None, reward=0.0, last_action=None):
    slots = []
    for s in env.slots:
        slots.append({
            "id": s.id,
            "type": s.slot_type,
            "occupied": s.is_occupied,
            "vehicle_id": s.current_vehicle_id,
            "departure_tick": s.departure_tick,
            "distance": s.distance_from_entrance,
            "pos": grid["slots"][s.id],
        })

    queue = []
    for v in list(env.vehicle_queue):
        queue.append({
            "id": v.id,
            "type": v.vehicle_type,
            "wait_time": v.wait_time,
            "requires_ev": v.requires_ev,
            "requires_handicapped": v.requires_handicapped,
        })

    incoming = None
    if len(env.vehicle_queue) > 0:
        v = env.vehicle_queue[0]
        incoming = {
            "id": v.id,
            "type": v.vehicle_type,
            "wait_time": v.wait_time,
            "requires_ev": v.requires_ev,
            "requires_handicapped": v.requires_handicapped,
        }

    m = env.metrics_tracker.get_summary()
    return {
        "step": env.current_step,
        "max_steps": env.max_steps,
        "slots": slots,
        "queue": queue,
        "incoming": incoming,
        "grid": grid,
        "stats": {
            "reward": round(m["total_reward"], 1),
            "allocated": m["allocated_count"],
            "rejected": m["total_rejections"],
            "unnecessary_rejections": m["unnecessary_rejections"],
            "queue_length": len(env.vehicle_queue),
            "occupancy": round(sum(1 for s in env.slots if s.is_occupied) / max(1, env.num_slots), 3),
            "avg_walk_dist": round(m["avg_walking_distance"], 2),
            "avg_wait_time": round(m["avg_waiting_time"], 2),
        },
        "action": action,
        "last_action": last_action,
        "reward": round(reward, 1),
    }


@app.get("/")
async def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/agents")
async def list_agents():
    return JSONResponse({"agents": list(AGENT_FACTORIES.keys())})


@app.get("/api/config")
async def get_config():
    return JSONResponse({"slot_colors": SLOT_COLORS, "vehicle_colors": VEHICLE_COLORS})


@app.websocket("/ws/sim")
async def sim_websocket(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            cmd = msg.get("cmd")

            if cmd == "init":
                num_slots = msg.get("slots", 12)
                agent_name = msg.get("agent", "Nearest-Slot")
                max_steps = msg.get("steps", 200)
                seed = msg.get("seed", 42)

                env = ParkingEnv(num_slots=num_slots, max_steps=max_steps)
                agent = _load_agent(agent_name, num_slots)
                grid = _build_grid(num_slots)

                obs, info = env.reset(seed=seed)
                state = _env_state(env, grid)
                state["config"] = {"slot_colors": SLOT_COLORS, "vehicle_colors": VEHICLE_COLORS}
                await ws.send_text(json.dumps({"type": "state", "data": state}))

                session = {"env": env, "agent": agent, "grid": grid, "seed": seed, "last_action": None}

            elif cmd == "step":
                env = session["env"]
                agent = session["agent"]
                grid = session["grid"]

                obs = env._get_observation()
                info = env._get_info()
                incoming = info.get("incoming_vehicle")

                if incoming is not None:
                    action = _select_action(agent, obs, info, env)
                else:
                    action = env.num_slots

                obs, reward, term, trunc, info = env.step(action)
                session["last_action"] = action

                state = _env_state(env, grid, action=action, reward=reward, last_action=action)
                state["terminated"] = term or trunc
                await ws.send_text(json.dumps({"type": "state", "data": state}))

            elif cmd == "reset":
                num_slots = msg.get("slots", 12)
                agent_name = msg.get("agent", "Nearest-Slot")
                max_steps = msg.get("steps", 200)
                seed = msg.get("seed", 42)

                env = ParkingEnv(num_slots=num_slots, max_steps=max_steps)
                agent = _load_agent(agent_name, num_slots)
                grid = _build_grid(num_slots)

                obs, info = env.reset(seed=seed)
                session["env"] = env
                session["agent"] = agent
                session["grid"] = grid
                session["seed"] = seed
                session["last_action"] = None

                state = _env_state(env, grid)
                state["config"] = {"slot_colors": SLOT_COLORS, "vehicle_colors": VEHICLE_COLORS}
                await ws.send_text(json.dumps({"type": "state", "data": state}))

            elif cmd == "run_day":
                env = session["env"]
                agent = session["agent"]
                grid = session["grid"]

                while env.current_step < env.max_steps:
                    obs = env._get_observation()
                    info = env._get_info()
                    incoming = info.get("incoming_vehicle")

                    if incoming is not None:
                        action = _select_action(agent, obs, info, env)
                    else:
                        action = env.num_slots

                    obs, reward, term, trunc, info = env.step(action)
                    state = _env_state(env, grid, action=action, reward=reward, last_action=action)
                    state["terminated"] = term or trunc
                    await ws.send_text(json.dumps({"type": "state", "data": state}))
                    await asyncio.sleep(0.05)

                    if term or trunc:
                        break

                await ws.send_text(json.dumps({"type": "done"}))

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(e)}))
        except Exception:
            pass
