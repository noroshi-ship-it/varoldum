
import os
import json
import pickle
import numpy as np


def save_checkpoint(path: str, tick: int, agents: list, grid_cells: np.ndarray,
                    physics_state: dict, structures_state: dict,
                    config_dict: dict, rng_state: dict = None):
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    state = {
        "tick": tick,
        "config": config_dict,
        "grid_cells": grid_cells,
        "physics": physics_state,
        "structures": structures_state,
        "n_agents": len(agents),
        "agents": [],
    }

    for agent in agents:
        agent_state = {
            "id": agent.id,
            "parent_id": agent.parent_id,
            "genome": agent.genome.tolist(),
            "position": agent.body.position.tolist(),
            "energy": agent.body.energy,
            "health": agent.body.health,
            "age": agent.body.age,
            "heading": agent.body.heading,
            "vx": agent.body.vx,
            "vy": agent.body.vy,
            "mineral_carried": agent.body.mineral_carried,
            "generation": agent.generation,
            "total_reward": agent.total_reward,
            "ticks_alive": agent.ticks_alive,
            "children_count": agent.children_count,
            "internal_state": agent.internal.as_vector().tolist(),
            "hypothesis_data": agent.get_hypothesis_data().tolist(),
            "brain_hidden": agent.brain.hidden_state.tolist(),
            "self_model_accuracy": agent.self_model.cumulative_accuracy,
        }
        state["agents"].append(agent_state)

    if rng_state:
        state["rng"] = rng_state

    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary_path = path.replace(".pkl", "_summary.json")
    summary = {
        "tick": tick,
        "n_agents": len(agents),
        "max_generation": max((a.generation for a in agents), default=0),
        "mean_energy": float(np.mean([a.body.energy for a in agents])) if agents else 0,
        "top_agent_reward": float(max((a.total_reward for a in agents), default=0)),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


def load_checkpoint(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def get_latest_checkpoint(output_dir: str) -> str | None:
    if not os.path.exists(output_dir):
        return None
    checkpoints = [f for f in os.listdir(output_dir) if f.startswith("checkpoint_") and f.endswith(".pkl")]
    if not checkpoints:
        return None
    checkpoints.sort()
    return os.path.join(output_dir, checkpoints[-1])
