
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.agent import Agent


def population_stats(agents: list) -> dict:
    if not agents:
        return {
            "count": 0, "mean_energy": 0, "mean_health": 0,
            "mean_age": 0, "max_generation": 0,
        }
    energies = [a.body.energy for a in agents]
    healths = [a.body.health for a in agents]
    ages = [a.body.age for a in agents]
    gens = [a.generation for a in agents]
    return {
        "count": len(agents),
        "mean_energy": float(np.mean(energies)),
        "mean_health": float(np.mean(healths)),
        "mean_age": float(np.mean(ages)),
        "max_age": int(max(ages)),
        "max_generation": int(max(gens)),
        "mean_children": float(np.mean([a.children_count for a in agents])),
    }


def genome_diversity(agents: list) -> float:
    if len(agents) < 2:
        return 0.0
    from agents.genome import NUM_TRAIT_GENES
    traits = np.array([a.genome[:NUM_TRAIT_GENES] for a in agents])
    n = min(50, len(agents))
    indices = np.random.choice(len(agents), size=(n, 2), replace=True)
    dists = np.linalg.norm(traits[indices[:, 0]] - traits[indices[:, 1]], axis=1)
    return float(np.mean(dists))


def behavioral_complexity(agents: list) -> dict:
    if not agents:
        return {"action_diversity": 0, "mean_self_model_accuracy": 0, "mean_surprise": 0}

    accuracies = [a.self_model.cumulative_accuracy for a in agents]
    surprises = [a.self_model.surprise for a in agents]

    return {
        "mean_self_model_accuracy": float(np.mean(accuracies)),
        "mean_surprise": float(np.mean(surprises)),
        "max_self_model_accuracy": float(np.max(accuracies)) if accuracies else 0,
    }


def trait_distribution(agents: list) -> dict:
    if not agents:
        return {}
    from agents.genome import LOCI, get_trait
    result = {}
    for name in LOCI:
        values = [get_trait(a.genome, name) for a in agents]
        result[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return result


def spatial_distribution(agents: list, w: int, h: int) -> dict:
    if not agents:
        return {"clustering": 0}
    positions = np.array([[a.x, a.y] for a in agents], dtype=np.float64)
    center = positions.mean(axis=0)
    dists = np.linalg.norm(positions - center, axis=1)
    return {
        "mean_dist_from_center": float(np.mean(dists)),
        "spatial_std": float(np.std(dists)),
    }
