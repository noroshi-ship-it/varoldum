
import numpy as np
from config import Config
from agents.agent_v4 import Agent
from evolution.mutation import mutate, crossover


def reproduce_asexual(
    parent: Agent, cfg: Config, rng: np.random.Generator
) -> Agent | None:
    if not parent.can_reproduce():
        return None

    child_genome = mutate(parent.genome, rng)

    offset = rng.integers(-2, 3, size=2)
    child_pos = np.array([
        (parent.x + offset[0]) % cfg.world_width,
        (parent.y + offset[1]) % cfg.world_height,
    ], dtype=np.float64)

    child = Agent(child_genome, cfg, child_pos, parent_id=parent.id)
    child.generation = parent.generation + 1

    parent_hyp_data = parent.get_hypothesis_data()
    child.inherit_hypotheses(parent_hyp_data, rng)

    child.inherit_composable_rules(parent, rng)

    child.inherit_concept_hypotheses(parent, rng)

    # Phase 6-8: Inherit meta-cognition systems
    child.inherit_grammar(parent, rng)
    child.inherit_meta_concepts(parent, rng)
    child.inherit_tom(parent, rng)

    # Phase 9-12: Inherit cognitive systems
    child.inherit_episodic(parent, rng)
    child.inherit_naming(parent, rng)

    # Phase 13: Inherit emergence infrastructure
    child.inherit_workspace(parent, rng)
    child.inherit_norms(parent, rng)

    child.lineage_id = parent.lineage_id
    child.body.init_genetic_frailty(rng)

    parent.body.energy -= cfg.reproduction_cost
    child.body.energy = cfg.reproduction_cost * 0.7

    parent.children_count += 1
    return child


def reproduce_sexual(
    parent_a: Agent, parent_b: Agent, cfg: Config, rng: np.random.Generator
) -> Agent | None:
    if not parent_a.can_reproduce() or not parent_b.can_reproduce():
        return None

    child_genome = crossover(parent_a.genome, parent_b.genome, rng)
    child_genome = mutate(child_genome, rng)

    mid_x = ((parent_a.x + parent_b.x) // 2) % cfg.world_width
    mid_y = ((parent_a.y + parent_b.y) // 2) % cfg.world_height
    child_pos = np.array([mid_x, mid_y], dtype=np.float64)

    child = Agent(child_genome, cfg, child_pos, parent_id=parent_a.id)
    child.generation = max(parent_a.generation, parent_b.generation) + 1

    stats_a = parent_a.hypotheses.stats
    stats_b = parent_b.hypotheses.stats
    if stats_a["mean_accuracy"] >= stats_b["mean_accuracy"]:
        better = parent_a
    else:
        better = parent_b
    child.inherit_hypotheses(better.get_hypothesis_data(), rng)
    child.inherit_composable_rules(better, rng)
    child.inherit_concept_hypotheses(better, rng)
    # Phase 6-8: Inherit from better parent
    child.inherit_grammar(better, rng)
    child.inherit_meta_concepts(better, rng)
    child.inherit_tom(better, rng)

    # Phase 9-12: Inherit cognitive systems from better parent
    child.inherit_episodic(better, rng)
    child.inherit_naming(better, rng)

    # Phase 13: Inherit emergence infrastructure from better parent
    child.inherit_workspace(better, rng)
    child.inherit_norms(better, rng)

    child.lineage_id = parent_a.lineage_id if parent_a.generation >= parent_b.generation else parent_b.lineage_id
    child.body.init_genetic_frailty(rng)

    cost = cfg.reproduction_cost * 0.6
    parent_a.body.energy -= cost
    parent_b.body.energy -= cost
    child.body.energy = cost * 1.2

    parent_a.children_count += 1
    parent_b.children_count += 1
    return child
