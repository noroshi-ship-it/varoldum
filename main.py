
import sys
import os
import time
import argparse
import numpy as np
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from utils import rng as rng_module
from world.grid import Grid
from world.environment import Environment
from world.physics import Physics
from world.ecology import Ecology
from world.social import SocialSystem
from world.structures import StructureManager, StructureType, BUILD_COST
from world.culture import CultureSystem
from agents.agent_v4 import Agent
from agents.language import ProtoLanguage
from agents.discrete_language import DiscreteLanguageSystem
from world.lineage_memory import LineageMemorySystem
from evolution.reproduction import reproduce_asexual, reproduce_sexual
from analytics.metrics import population_stats, genome_diversity, behavioral_complexity, trait_distribution
from analytics.consciousness_probes import probe_all
from analytics.logger import Logger
from analytics.events import EventDetector
from analytics.hall_of_fame import HallOfFame
from utils.geometry import toroidal_distance
from checkpoint import save_checkpoint, load_checkpoint, get_latest_checkpoint
from neural.gpu_batch import batch_encode, batch_think, get_device


def parse_args():
    p = argparse.ArgumentParser(description="Varoldum v4 ALife Simulation")
    p.add_argument("--ticks", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pop", type=int, default=None)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--output", type=str, default="output")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-gpu", action="store_true", help="Disable GPU batch, use CPU numpy only")
    p.add_argument("--resume", nargs="?", const="auto", default=None)
    p.add_argument("--checkpoint-interval", type=int, default=5000)
    return p.parse_args()


class SpatialHash:
    def __init__(self, cell_size=8):
        self.cell_size = cell_size
        self.grid = {}

    def build(self, agents):
        self.grid.clear()
        for a in agents:
            if not a.is_alive:
                continue
            cx = int(a.body.position[0]) // self.cell_size
            cy = int(a.body.position[1]) // self.cell_size
            key = (cx, cy)
            if key not in self.grid:
                self.grid[key] = []
            self.grid[key].append(a)

    def query(self, x, y, radius):
        cs = self.cell_size
        min_cx = int(x - radius) // cs
        max_cx = int(x + radius) // cs
        min_cy = int(y - radius) // cs
        max_cy = int(y + radius) // cs
        result = []
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                result.extend(self.grid.get((cx, cy), []))
        return result


_spatial = SpatialHash(cell_size=8)


def find_mate(agent, agents, cfg):
    nearby = _spatial.query(agent.body.position[0], agent.body.position[1], 3.0)
    for other in nearby:
        if other.id == agent.id or not other.can_reproduce():
            continue
        dist = toroidal_distance(agent.body.position, other.body.position,
                                 cfg.world_width, cfg.world_height)
        if dist < 3.0:
            return other
    return None


def count_nearby(agent, agents, radius=4):
    nearby = _spatial.query(agent.body.position[0], agent.body.position[1], radius)
    count = 0
    for other in nearby:
        if other.id == agent.id:
            continue
        dx = abs(agent.body.position[0] - other.body.position[0])
        dy = abs(agent.body.position[1] - other.body.position[1])
        if dx <= radius and dy <= radius:
            count += 1
    return count


def resolve_actions(grid, physics, structures, agents, actions, cfg, evo_rng, ecology=None, tick=0, disasters=None):
    """
    Context-based physics resolution.
    6 generic force channels — physics determines meaning from context.
    mouth > 0 on food = eat. mouth > 0 on mineral = collect. mouth < 0 = emit/deposit.
    manipulate > 0 with material = craft/build.
    social is handled separately in social.py.
    signal > 0 = broadcast.
    """
    new_agents = []
    agent_data = {}

    # Count absorbers per cell for cooperative foraging bonus
    mouth_cells = {}
    for agent, action in zip(agents, actions):
        if agent.is_alive and action.get("mouth", 0) > 0:
            cell_key = (agent.x, agent.y)
            mouth_cells[cell_key] = mouth_cells.get(cell_key, 0) + 1

    # Parental care map
    parent_positions = {}
    for agent in agents:
        if agent.is_alive and agent.children_count > 0:
            parent_positions[agent.id] = agent.body.position.copy()

    for agent, action in zip(agents, actions):
        if not agent.is_alive:
            continue

        data = {"energy_gained": 0.0, "damage": 0.0, "mineral": 0.0,
                "toxin": 0.0, "medicine": 0.0, "temperature": 0.5,
                "structure_built": False, "teaching_done": False}

        # === MOVEMENT (a[0], a[1]) ===
        new_dx = action["move_dx"]
        new_dy = action["move_dy"]
        new_x = int(round(agent.body.position[0] + new_dx)) % cfg.world_width
        new_y = int(round(agent.body.position[1] + new_dy)) % cfg.world_height
        if structures.is_blocked(new_x, new_y):
            new_dx *= 0.1
            new_dy *= 0.1
        agent.body.move(new_dx, new_dy, cfg.world_width, cfg.world_height)

        x, y = agent.x, agent.y
        data["temperature"] = physics.get_temperature(x, y)

        # Nest bonus (physics: shelter provides warmth)
        if structures.get_nest_bonus(x, y):
            agent.body.energy += 0.001

        # === MOUTH: intake/output with environment (a[2]) ===
        mouth = action.get("mouth", 0)

        if mouth > 0:
            # Positive mouth = absorb from environment
            # CONTEXT: what's here determines what gets absorbed
            intensity = mouth  # [0, 1] after tanh mapping... actually [-1,1]
            # mouth is raw [-1,1], positive half = absorb intensity
            absorb = mouth  # 0 to 1

            # Food absorption
            consumed = grid.consume_resource(x, y, amount=absorb * 0.3)
            farm_bonus = structures.get_farm_bonus(x, y)
            consumed *= (1.0 + farm_bonus)
            if ecology is not None:
                plant_nut, plant_tox = ecology.consume_plants(x, y, amount=absorb * 0.1)
                consumed += plant_nut
                agent.body.take_damage(plant_tox * 0.3)
                data["damage"] += plant_tox * 0.3
            # Cooperative foraging
            n_eaters = mouth_cells.get((x, y), 1)
            consumed *= 1.0 + 0.15 * max(0, n_eaters - 1)
            agent.body.eat(consumed)
            data["energy_gained"] = consumed

            # Mineral absorption (same mouth force, different material)
            mineral_available = physics.consume_mineral(x, y, absorb * 0.2)
            collected = agent.body.collect_mineral(mineral_available)
            data["mineral"] = collected

            # Substance collection into inventory
            if hasattr(agent, 'inventory'):
                top_ids, top_concs = physics.chemistry.get_top_substances(x, y, n=1)
                if len(top_concs) > 0 and top_concs[0] > 0:
                    top_sid = int(top_ids[0])
                    sub_consumed = physics.chemistry.consume_substance(
                        x, y, top_sid, absorb * 0.1)
                    agent.inventory.add_item(top_sid, sub_consumed)

            # Withdraw from storage structure (mouth absorb at structure)
            withdrawn = structures.withdraw_resource(x, y, absorb * 0.1)
            if withdrawn > 0:
                agent.body.energy += withdrawn

        elif mouth < 0:
            # Negative mouth = emit/deposit to environment
            emit = -mouth  # 0 to 1
            deposited = structures.deposit_resource(x, y, emit * 0.1)
            if deposited > 0:
                agent.body.energy -= deposited

        # === MANIPULATE: transform materials (a[4]) ===
        manipulate = action.get("manipulate", 0)

        if manipulate > 0:
            # Positive manipulate = craft/build
            # CONTEXT: if carrying minerals → combine into medicine
            if agent.body.mineral_carried > 0:
                use_amount = manipulate * min(0.1, agent.body.mineral_carried)
                heal = physics.apply_medicine(use_amount, agent.body.mineral_carried)
                if heal > 0:
                    agent.body.heal_medicine(heal)
                    data["medicine"] = heal

            # CONTEXT: enough energy → build structure
            # manipulate intensity as probability (physics: effort → chance)
            if evo_rng.random() < manipulate * 0.05:
                # Structure type from movement pattern (emergent)
                pattern = action["move_dx"] + action["move_dy"] * 2
                stype = int(abs(pattern * 3) % 6) + 1
                cost = BUILD_COST.get(stype, 0.15)
                if agent.body.energy > cost:
                    if structures.build(stype, x, y, agent.lineage_id):
                        agent.body.energy -= cost
                        data["structure_built"] = True

        # === SIGNAL: broadcast (a[5]) ===
        signal = action.get("signal", 0)
        if signal > 0:
            grid.stamp_agent(x, y, signal)
            # Inscription: signal at structure = write
            if action.get("token_utterance") is not None:
                concepts = agent.brain.get_concepts()
                structures.inscribe(x, y, action["token_utterance"],
                                    concepts, agent.lineage_id, tick)
        physics.stamp_scent(x, y, 0.3)

        # === REPRODUCTION: emergent from social + mouth + energy ===
        # When social force is positive AND mouth is positive AND enough energy
        # Physics: two compatible organisms exchanging resources = offspring
        social = action.get("social", 0)
        if social > 0 and mouth > 0 and agent.can_reproduce():
            mate = find_mate(agent, agents, cfg)
            if mate is not None:
                child = reproduce_sexual(agent, mate, cfg, evo_rng)
            else:
                child = reproduce_asexual(agent, cfg, evo_rng)
            if child is not None and len(agents) + len(new_agents) < cfg.max_population:
                # Disaster-zone mutagenesis: offspring born in danger get extra mutations
                # Creates diversity bursts after catastrophes — like radiation evolution
                danger = env.disasters.get_total_damage(agent.x, agent.y)
                if danger > 0.1:
                    from agents.genome import LOCI, ARCH_OFFSET
                    mr_idx = LOCI["mutation_rate"][0]
                    child.genome[mr_idx] += danger * 0.5  # temporary mutation boost
                    # Also nudge architecture genes toward exploration
                    for ai in range(1, 5):
                        if evo_rng.random() < danger:
                            child.genome[ARCH_OFFSET + ai] += evo_rng.normal(0, danger * 30)
                new_agents.append(child)

        # === PARENTAL CARE: physics — proximity + youth ===
        if agent.parent_id > 0:
            parent_pos = parent_positions.get(agent.parent_id)
            if parent_pos is not None:
                pdist = toroidal_distance(agent.body.position, parent_pos,
                                          cfg.world_width, cfg.world_height)
                care_range = 5.0
                proximity = max(0.0, 1.0 - pdist / care_range)
                youth = max(0.0, 1.0 - agent.body.age / 200.0)
                care = proximity * youth
                if care > 0.0:
                    agent.body.energy += 0.003 * care
                    agent.body.health = min(1.0, agent.body.health + 0.0015 * care)

        # === NATURAL MORTALITY: infection, genetic defects, accidents, aging ===
        natural_dmg = agent.body.update_natural_mortality(evo_rng)
        data["damage"] += natural_dmg

        # === ENVIRONMENTAL DAMAGE (pure physics) ===
        hazard = grid.get_cell(x, y)[1]
        hazard_dmg = hazard * 0.1
        agent.body.take_damage(hazard_dmg)

        trap_dmg = structures.get_trap_damage(x, y, agent.lineage_id)
        if trap_dmg > 0:
            agent.body.take_damage(trap_dmg)

        toxin = physics.get_toxin(x, y)
        toxin_dmg = toxin * 0.05
        agent.body.take_damage(toxin_dmg)
        if ecology is not None:
            pred_danger = ecology.get_predator_danger(x, y)
            if pred_danger > 0 and evo_rng.random() < pred_danger:
                pred_dmg = pred_danger * 0.3
                agent.body.take_damage(pred_dmg)
                data["damage"] += pred_dmg
                if agent.body.health <= 0:
                    agent.body.cause_of_death = "predator"

        rad_dmg = physics.hidden.get_radiation_damage(x, y)
        if rad_dmg > 0:
            agent.body.take_damage(rad_dmg)
            data["damage"] += rad_dmg

        # === DISASTER DAMAGE ===
        disaster_dmg = 0.0
        if disasters is not None:
            dd = disasters.get_disaster_damage(x, y)
            # Earthquake: direct structural damage — LETHAL near epicenter
            if dd["earthquake"] > 0.05:
                eq_dmg = dd["earthquake"] * 0.5
                agent.body.take_damage(eq_dmg)
                disaster_dmg += eq_dmg
            # Flood: drowning damage — severe
            if dd["flood"] > 0.03:
                fl_dmg = dd["flood"] * 0.4
                agent.body.take_damage(fl_dmg)
                agent.body.energy -= dd["flood"] * 0.1
                disaster_dmg += fl_dmg
            # Drought: slow starvation (energy drain, not health)
            if dd["drought"] > 0.1:
                agent.body.energy -= dd["drought"] * 0.03
            # Plague: sickness — energy + health drain
            if dd["plague"] > 0.05:
                pl_dmg = dd["plague"] * 0.15
                agent.body.take_damage(pl_dmg)
                agent.body.energy -= dd["plague"] * 0.05
                disaster_dmg += pl_dmg
            data["disaster_damage"] = disaster_dmg
            if agent.body.health <= 0 and disaster_dmg > 0:
                if dd["earthquake"] > dd["flood"] and dd["earthquake"] > dd["plague"]:
                    agent.body.cause_of_death = "earthquake"
                elif dd["flood"] > dd["plague"]:
                    agent.body.cause_of_death = "flood"
                elif dd["plague"] > 0:
                    agent.body.cause_of_death = "plague"

        data["toxin"] = toxin
        data["damage"] += hazard_dmg + toxin_dmg + trap_dmg + disaster_dmg

        agent_data[agent.id] = data

    return new_agents, agent_data


def log_discovered_rules(agents, tick, logger):
    all_rules = []
    for agent in agents:
        for hyp in agent.concept_hyp.get_best(min_tests=10, min_acc=0.6):
            all_rules.append({
                "agent_id": agent.id, "generation": agent.generation,
                "rule": hyp.describe(agent.bottleneck_size),
                "accuracy": hyp.accuracy, "tests": hyp.tests,
            })
    all_rules.sort(key=lambda r: r["accuracy"], reverse=True)
    for i, rule in enumerate(all_rules[:15]):
        logger.log_dict("discovered_rules", tick, {"rank": i + 1, **rule})


def log_composable_rules(agents, tick, logger):
    all_rules = []
    for agent in agents:
        for rule in agent.composable.get_best_rules(min_tests=8, min_accuracy=0.55):
            all_rules.append({
                "agent_id": agent.id, "generation": agent.generation,
                "rule": rule.describe(), "accuracy": rule.accuracy,
                "complexity": rule.complexity, "tests": rule.tests,
            })
    all_rules.sort(key=lambda r: r["accuracy"] * r["complexity"], reverse=True)
    for i, rule in enumerate(all_rules[:10]):
        logger.log_dict("composable_rules", tick, {"rank": i + 1, **rule})


def main():
    args = parse_args()
    cfg = Config()
    if args.ticks: cfg.max_ticks = args.ticks
    if args.seed: cfg.seed = args.seed
    if args.pop: cfg.initial_population = args.pop
    if args.width: cfg.world_width = args.width
    if args.height: cfg.world_height = args.height

    checkpoint_interval = args.checkpoint_interval
    start_tick = 0

    if args.resume:
        if args.resume == "auto":
            ckpt_path = get_latest_checkpoint(args.output)
        else:
            ckpt_path = args.resume
        if ckpt_path:
            print(f"Resuming from checkpoint: {ckpt_path}")
            ckpt = load_checkpoint(ckpt_path)
            if ckpt:
                start_tick = ckpt["tick"]
                print(f"  Resumed at tick {start_tick} with {ckpt['n_agents']} agents")
            else:
                print(f"  Failed to load, starting fresh")
                args.resume = None
        else:
            print(f"  No checkpoint found, starting fresh")
            args.resume = None

    rng_module.init(cfg.seed)
    world_rng = rng_module.get("world")
    agent_rng = rng_module.get("agents")
    evo_rng = rng_module.get("evolution")

    grid = Grid(cfg, world_rng)
    env = Environment(cfg, grid, world_rng)
    physics = Physics(cfg, world_rng)
    social = SocialSystem(cfg.world_width, cfg.world_height)
    structures = StructureManager(cfg.world_width, cfg.world_height)
    culture = CultureSystem()
    ecology = Ecology(cfg.world_width, cfg.world_height, rng=world_rng)
    language = ProtoLanguage(cfg.world_width, cfg.world_height, cfg.hear_radius)
    discrete_lang = DiscreteLanguageSystem(cfg.world_width, cfg.world_height, cfg.hear_radius)
    lineage_memory = LineageMemorySystem(max_rules_per_lineage=8)

    agents: list[Agent] = []
    if args.resume and ckpt:
        for agent_state in ckpt.get("agents", []):
            genome = np.array(agent_state["genome"])
            pos = np.array(agent_state["position"])
            agent = Agent(genome, cfg, pos, parent_id=agent_state.get("parent_id", -1))
            agent.generation = agent_state.get("generation", 0)
            agent.total_reward = agent_state.get("total_reward", 0)
            agent.ticks_alive = agent_state.get("ticks_alive", 0)
            agent.children_count = agent_state.get("children_count", 0)
            agent.body.energy = agent_state.get("energy", 0.5)
            agent.body.health = agent_state.get("health", 1.0)
            agent.body.age = agent_state.get("age", 0)
            agents.append(agent)
        if "grid_cells" in ckpt:
            gc = np.array(ckpt["grid_cells"])
            if gc.shape == grid.cells.shape:
                grid.cells = gc
    else:
        for _ in range(cfg.initial_population):
            pos = np.array([agent_rng.integers(0, cfg.world_width),
                            agent_rng.integers(0, cfg.world_height)], dtype=np.float64)
            agents.append(Agent.create_random(cfg, pos, agent_rng))

    logger = Logger(args.output)
    event_detector = EventDetector(args.output)
    hall_of_fame = HallOfFame(args.output)

    if not args.quiet:
        print(f"=== VAROLDUM v4 ===")
        print(f"World: {cfg.world_width}x{cfg.world_height} | Pop: {len(agents)} | Ticks: {cfg.max_ticks} | Seed: {cfg.seed}")
        if start_tick > 0:
            print(f"Resumed from tick {start_tick}")
        print()

    start_time = time.time()
    best_generation_ever = 0
    max_complexity_ever = 0
    births_this_period = 0
    deaths_this_period = 0
    use_gpu = not args.no_gpu
    profile_accum = {"world": 0, "perceive": 0, "think": 0, "actions": 0, "update": 0, "ticks": 0}

    for tick in range(start_tick, cfg.max_ticks):
        t0 = time.time()
        env.update(tick)
        physics.update(tick, env.season_phase, grid.cells)
        ecology.update(tick, physics.temperature, physics.terrain.water, env.light_level,
                       soil_ph=physics.hidden.soil_ph)
        structures.update(tick)
        grid.clear_agent_layer()
        for agent in agents:
            grid.stamp_agent(agent.x, agent.y)

        _spatial.build(agents)
        t1 = time.time()

        for agent in agents:
            nearby = count_nearby(agent, agents, radius=4)
            agent._nearby_agent_count = nearby  # Phase 5A: feed social emotions
            heard = language.get_strongest_signal(agent, tick)
            # Phase 3: Pass heard discrete tokens
            heard_tokens = discrete_lang.get_nearest_utterance(
                agent.id, agent.body.position, tick)
            agent._heard_tokens = heard_tokens
            agent.perceive(grid, env.light_level, agent_rng,
                          season=env.season_phase, physics=physics,
                          nearby_agents=nearby, heard_signal=heard,
                          ecology=ecology, structures=structures,
                          disasters=env.disasters)
        t2 = time.time()

        if use_gpu:
            contexts = {}
            for agent in agents:
                contexts[agent.id] = agent.prepare_context()
            gpu_results = batch_think(agents, contexts)
        else:
            gpu_results = None

        actions = []
        if gpu_results:
            for agent in agents:
                r = gpu_results.get(agent.id)
                if r:
                    agent.apply_gpu_result(r[0], r[1], r[2], r[3])
                else:
                    agent.think()
                actions.append(agent.act())
        else:
            for agent in agents:
                agent.think()
                actions.append(agent.act())
        t3 = time.time()

        new_agents, agent_data = resolve_actions(
            grid, physics, structures, agents, actions, cfg, evo_rng, ecology=ecology, tick=tick,
            disasters=env.disasters
        )
        t3b = time.time()
        profile_accum["actions"] += t3b - t3

        for agent, action in zip(agents, actions):
            if agent.is_alive:
                utterance = action.get("utterance", None)
                if utterance is not None:
                    language.broadcast(agent, utterance, tick)
                # Phase 3: Discrete token broadcast
                token_utt = action.get("token_utterance", None)
                if token_utt is not None:
                    discrete_lang.broadcast(
                        agent.id, agent.body.position, agent.lineage_id,
                        token_utt, tick)
        language.cleanup(tick)
        discrete_lang.cleanup(tick)

        substance_props = physics.chemistry.substance_props if hasattr(physics.chemistry, 'substance_props') else None
        social_rewards, positive_ints, negative_ints = social.resolve_social(
            agents, actions, tick, substance_props=substance_props)

        # Phase 5A: Feed social interaction signals to agents for emotion update
        for agent in agents:
            if not agent.is_alive:
                continue
            agent._pending_positive_interaction += positive_ints.get(agent.id, 0)
            agent._pending_negative_interaction += negative_ints.get(agent.id, 0)
            agent._pending_social_reward += social_rewards.get(agent.id, 0) * 0.1

        # Phase 5B: Communication reward - listener benefited from speaker's info
        agent_by_id = {a.id: a for a in agents if a.is_alive}
        for agent in agents:
            if not agent.is_alive or agent._heard_tokens is None:
                continue
            data = agent_data.get(agent.id, {})
            eg = data.get("energy_gained", 0)
            dmg = data.get("damage", 0)
            # Continuous outcome score instead of binary judgment
            outcome_score = eg - dmg  # positive = net benefit
            speaker = agent_by_id.get(agent._heard_tokens.sender_id)
            if speaker is not None and outcome_score > 0:
                # Both parties benefit from successful communication
                speaker._pending_positive_interaction += 0.3
                speaker._pending_social_reward += 0.1
                agent._pending_positive_interaction += 0.2
                agent._pending_social_reward += 0.05

        # Disaster communication reward: sharing warnings saves lives
        # Agents who hear tokens in disaster zones get survival bonus,
        # speakers who warned get large reward — makes communication evolutionary useful
        for agent in agents:
            if not agent.is_alive or agent._heard_tokens is None:
                continue
            x, y = agent.x, agent.y
            warnings = env.disasters.get_disaster_warnings(x, y)
            danger_level = warnings["tremor"] + warnings["flood"] + warnings["drought"] + warnings["plague"]
            if danger_level > 0.3:
                # Listener heard warning in danger zone → survival bonus
                agent.body.energy += min(0.02, danger_level * 0.03)
                agent._pending_positive_interaction += 0.5
                # Speaker gets major reward for warning others
                speaker = agent_by_id.get(agent._heard_tokens.sender_id)
                if speaker is not None:
                    speaker._pending_social_reward += danger_level * 0.5
                    speaker._pending_positive_interaction += 1.0

        if tick % 5 == 0:
            culture.process_teaching(agents, tick, agent_rng, lineage_memory=lineage_memory, actions=actions)
            culture.process_lineage_study(agents, lineage_memory, tick, agent_rng)
            culture.process_imitation(agents, tick, agent_rng)
            culture.cleanup()
            # Cleanup extinct lineage pools
            active_lineages = {a.lineage_id for a in agents if a.is_alive}
            lineage_memory.cleanup(active_lineages)

        for agent in agents:
            if not agent.is_alive:
                continue
            data = agent_data.get(agent.id, {})
            eg = data.get("energy_gained", 0)
            dmg = data.get("damage", 0)
            mineral = data.get("mineral", 0)
            medicine = data.get("medicine", 0)
            temp = data.get("temperature", 0.5)
            toxin = data.get("toxin", 0)
            struct_built = data.get("structure_built", False)
            teaching = data.get("teaching_done", False)

            reward = agent.compute_intrinsic_reward(
                eg, dmg, mineral, medicine,
                structure_built=struct_built, teaching_done=teaching
            )
            reward += social_rewards.get(agent.id, 0)

            x, y = agent.x, agent.y
            agent.update(reward, energy_gained=eg, damage_taken=dmg,
                        temperature=temp, mineral_found=mineral, toxin_exposure=toxin,
                        plant_found=ecology.get_plant_density(x, y),
                        predator_near=ecology.get_predator_density(x, y),
                        substance_conc=float(physics.chemistry.substances[
                            x % physics.w, y % physics.h].max()),
                        water_here=physics.terrain.get_water(x, y),
                        height_here=physics.terrain.get_height(x, y),
                        radiation_dmg=physics.hidden.get_radiation_damage(x, y),
                        medicine_result=medicine,
                        fertility_here=float(physics.fertility[
                            x % physics.w, y % physics.h]))
        t4 = time.time()

        profile_accum["world"] += t1 - t0
        profile_accum["perceive"] += t2 - t1
        profile_accum["think"] += t3 - t2
        profile_accum["update"] += t4 - t3
        profile_accum["ticks"] += 1

        dead = [a for a in agents if not a.is_alive]
        for a in dead:
            hall_of_fame.check_dead_agent(a, tick)
            ecology.add_dead_matter(a.x, a.y, amount=0.15)
            # Phase 4: Nearby agents observe death through concept similarity
            dead_concepts = a.brain.get_concepts()
            nearby_alive = _spatial.query(a.x, a.y, 6)
            for observer in nearby_alive:
                if observer.id != a.id and observer.is_alive:
                    observer.mortality.observe_nearby_death(
                        observer.brain.get_concepts(), dead_concepts, tick)
                    # Grief: if observer was bonded to the dead agent
                    if hasattr(observer, 'observe_partner_death'):
                        observer.observe_partner_death(a.id)
                    # Negative interaction proportional to bond — no threshold
                    if hasattr(observer, '_pending_negative_interaction'):
                        bond = observer._bonds.get(a.id, 0) if hasattr(observer, '_bonds') else 0
                        observer._pending_negative_interaction += bond
        deaths_this_period += len(dead)
        births_this_period += len(new_agents)
        agents = [a for a in agents if a.is_alive]
        agents.extend(new_agents)

        if len(agents) < cfg.min_population:
            deficit = cfg.extinction_recovery_count - len(agents)
            for _ in range(max(5, deficit)):
                pos = np.array([agent_rng.integers(0, cfg.world_width),
                                agent_rng.integers(0, cfg.world_height)], dtype=np.float64)
                agents.append(Agent.create_random(cfg, pos, agent_rng))

        if tick % 10 == 0:
            # Decomposer activity boosts soil fertility
            decomp_boost = ecology.fauna[:, :, 2] * 0.02
            physics.fertility += decomp_boost
            np.clip(physics.fertility, 0, 2, out=physics.fertility)

            fertility_boost = physics.fertility.copy()
            for pos_key, s in structures.structures.items():
                if s.stype == StructureType.FARM:
                    fx, fy = pos_key
                    for dx in range(-1, 2):
                        for dy in range(-1, 2):
                            nx = (fx + dx) % cfg.world_width
                            ny = (fy + dy) % cfg.world_height
                            fertility_boost[nx, ny] = min(1.0, fertility_boost[nx, ny] + 0.1)
            grid.cells[:, :, 0] += 0.005 * fertility_boost * (1.0 - grid.cells[:, :, 0])
            np.clip(grid.cells[:, :, 0], 0, 1, out=grid.cells[:, :, 0])

        if tick % cfg.log_interval == 0:
            stats = population_stats(agents)
            diversity = genome_diversity(agents)
            behavior = behavioral_complexity(agents)
            social_stats = social.recent_stats
            culture_stats = culture.recent_stats
            struct_stats = structures.stats
            lang_stats = language.recent_stats

            event_detector.check(tick, agents, structures=structures, lang_stats=lang_stats)

            lineage_counts = Counter(a.lineage_id for a in agents)
            for lid, count in lineage_counts.items():
                founder_gen = min((a.generation for a in agents if a.lineage_id == lid), default=0)
                hall_of_fame.check_dynasty(lid, count, founder_gen, tick)

            best_generation_ever = max(best_generation_ever, int(stats.get("max_generation", 0)))

            n_with_concepts = 0
            mean_wm_acc = 0.0
            mean_think_steps = 0.0
            mean_bottleneck = 0.0
            n_concept_rules = 0
            mean_concept_acc = 0.0
            n_complex_rules = 0
            max_complexity = 0

            mean_symbol_entropy = 0.0
            mean_survival_prob = 0.0
            mean_death_awareness = 0.0
            mean_thought_depth = 0.0
            total_vocab_used = 0
            mean_social_need = 0.0
            mean_trust_state = 0.0
            mean_social_sat = 0.0
            mean_inventory_fullness = 0.0
            mean_trust_score = 0.0
            mean_grief = 0.0
            total_bonds = 0
            total_debts = 0

            for a in agents:
                mean_wm_acc += a.brain.cumulative_wm_accuracy
                mean_think_steps += a.think_steps
                mean_bottleneck += a.bottleneck_size
                cs = a.concept_hyp.stats
                if cs["n_rules"] > 0:
                    n_with_concepts += 1
                    mean_concept_acc += cs["mean_accuracy"]
                    n_concept_rules += cs["n_rules"]
                comp_s = a.composable.stats
                n_complex_rules += comp_s["n_complex_rules"]
                if comp_s["max_complexity"] > max_complexity:
                    max_complexity = comp_s["max_complexity"]
                # Abstract thinking stats
                mean_symbol_entropy += a.symbols.stats["symbol_entropy"]
                mean_survival_prob += a.mortality.survival_prob
                mean_death_awareness += a.mortality.get_death_awareness()
                mean_thought_depth += a.brain._thought_depth
                total_vocab_used += a.discrete_vocab.stats["vocab_used"]
                # Society stats
                mean_social_need += a.internal.social_need
                mean_trust_state += a.internal.trust_state
                mean_social_sat += a.internal.social_satisfaction
                mean_inventory_fullness += a.inventory.fullness
                mean_trust_score += a.trust_memory.mean_trust
                mean_grief += a._grief_level
                total_bonds += len(a._bonds)
                total_debts += len(a._debts)

            n = max(1, len(agents))
            mean_wm_acc /= n
            mean_think_steps /= n
            mean_bottleneck /= n
            if n_with_concepts > 0:
                mean_concept_acc /= n_with_concepts
            max_complexity_ever = max(max_complexity_ever, max_complexity)
            mean_symbol_entropy /= n
            mean_survival_prob /= n
            mean_death_awareness /= n
            mean_thought_depth /= n
            mean_social_need /= n
            mean_trust_state /= n
            mean_social_sat /= n
            mean_inventory_fullness /= n
            mean_trust_score /= n
            mean_grief /= n
            lineage_stats = lineage_memory.get_stats()

            log_data = {
                **stats, "diversity": diversity, **behavior,
                "season": env.season_phase, "light": env.light_level,
                "mean_wm_accuracy": mean_wm_acc,
                "mean_think_steps": mean_think_steps,
                "mean_bottleneck_size": mean_bottleneck,
                "agents_with_concept_rules": n_with_concepts,
                "mean_concept_rule_accuracy": mean_concept_acc,
                "n_concept_rules": n_concept_rules,
                "n_complex_rules": n_complex_rules,
                "max_complexity": max_complexity,
                "trades": social_stats["trades"],
                "combats": social_stats["combats"],
                "cooperations": social_stats["cooperations"],
                "teachings": culture_stats["teachings"],
                "imitations": culture_stats["imitations"],
                "total_teachings": culture_stats["total_teachings"],
                "total_imitations": culture_stats["total_imitations"],
                "struct_total": struct_stats.get("total_structures", 0),
                "struct_built_ever": struct_stats.get("total_built_ever", 0),
                "signals_sent": lang_stats["signals_sent"],
                "signals_heard": lang_stats["signals_heard"],
                **{f"chem_{k}": v for k, v in physics.chemistry.get_stats().items()},
                **{f"terrain_{k}": v for k, v in physics.terrain.get_stats().items()},
                **{f"eco_{k}": v for k, v in ecology.get_stats().items()},
                **{f"hidden_{k}": v for k, v in physics.hidden.get_stats().items()},
                "unique_signalers": lang_stats["unique_senders"],
                "mean_symbol_entropy": mean_symbol_entropy,
                "mean_survival_prob": mean_survival_prob,
                "mean_death_awareness": mean_death_awareness,
                "mean_thought_depth": mean_thought_depth,
                "total_vocab_used": total_vocab_used,
                **{f"dlang_{k}": v for k, v in discrete_lang.recent_stats.items()},
                "best_generation_ever": best_generation_ever,
                "max_complexity_ever": max_complexity_ever,
                "births": births_this_period,
                "deaths": deaths_this_period,
                "total_resource": float(np.sum(grid.cells[:, :, 0])),
                # Society engine stats
                "mean_social_need": mean_social_need,
                "mean_trust_state": mean_trust_state,
                "mean_social_satisfaction": mean_social_sat,
                "mean_inventory_fullness": mean_inventory_fullness,
                "mean_trust_score": mean_trust_score,
                "inventory_trades": social_stats.get("inventory_trades", 0),
                "lineage_pools": lineage_stats["pools"],
                "lineage_rules_total": lineage_stats["rules"],
                "lineage_mean_accuracy": lineage_stats["mean_accuracy"],
                "lineage_studies": culture_stats.get("total_lineage_studies", 0),
                "mean_grief": mean_grief,
                "total_bonds": total_bonds,
                "total_debts": total_debts,
                **{f"disaster_{k}": v for k, v in env.disasters.get_stats().items()},
            }
            logger.log_dict("population", tick, log_data)
            births_this_period = 0
            deaths_this_period = 0

            if not args.quiet:
                elapsed = time.time() - start_time
                tps = (tick - start_tick + 1) / max(0.001, elapsed)
                st = struct_stats.get("total_structures", 0)
                pt = profile_accum["ticks"] or 1
                prof_str = (f"world={profile_accum['world']/pt*1000:.0f}ms "
                           f"perc={profile_accum['perceive']/pt*1000:.0f}ms "
                           f"think={profile_accum['think']/pt*1000:.0f}ms "
                           f"upd={profile_accum['update']/pt*1000:.0f}ms")
                print(
                    f"[t={tick:>6d}] pop={stats['count']:>3d} "
                    f"gen={stats['max_generation']:>3d} "
                    f"E={stats['mean_energy']:.2f} "
                    f"BN={mean_bottleneck:.1f} "
                    f"WM={mean_wm_acc:.2f} "
                    f"think={mean_think_steps:.1f} "
                    f"concepts={n_with_concepts}({mean_concept_acc:.2f}) "
                    f"struct={st} "
                    f"({tps:.1f}t/s) [{prof_str}]"
                )
                profile_accum = {"world": 0, "perceive": 0, "think": 0, "actions": 0, "update": 0, "ticks": 0}

        if tick % cfg.snapshot_interval == 0 and tick > 0:
            logger.log_snapshot(tick, agents)
            log_discovered_rules(agents, tick, logger)
            log_composable_rules(agents, tick, logger)

            traits = trait_distribution(agents)
            flat = {}
            for name, dist in traits.items():
                flat[f"{name}_mean"] = dist["mean"]
                flat[f"{name}_std"] = dist["std"]
            logger.log_dict("traits", tick, flat)

            if agents:
                top = max(agents, key=lambda a: a.total_reward)
                probes = probe_all(top)
                probes["agent_id"] = top.id
                probes["generation"] = top.generation
                logger.log_dict("consciousness", tick, probes)

                if not args.quiet:
                    print(f"  top=#{top.id} gen={top.generation} wm={top.brain.cumulative_wm_accuracy:.3f}")

        if tick > 0 and tick % checkpoint_interval == 0:
            ckpt_path = os.path.join(args.output, f"checkpoint_{tick:08d}.pkl")
            struct_list = []
            for pos_key, s in structures.structures.items():
                struct_list.append({
                    "stype": int(s.stype), "x": s.x, "y": s.y,
                    "lineage": s.builder_lineage,
                    "stored": s.stored_resource,
                })
            structures_save = {"structures_list": struct_list, "total_built": structures.total_built}
            physics_save = {
                "temperature": physics.temperature.tolist(),
                "mineral": physics.mineral.tolist(),
                "toxin": physics.toxin.tolist(),
                "fertility": physics.fertility.tolist(),
            }
            config_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}
            save_checkpoint(
                ckpt_path, tick, agents, grid.cells,
                physics_save, structures_save, config_dict
            )
            if not args.quiet:
                print(f"  [CHECKPOINT saved: {ckpt_path}]")

    logger.flush()
    logger.close()
    elapsed = time.time() - start_time

    final_ckpt = os.path.join(args.output, f"checkpoint_{cfg.max_ticks:08d}_final.pkl")
    struct_list = []
    for pos_key, s in structures.structures.items():
        struct_list.append({
            "stype": int(s.stype), "x": s.x, "y": s.y,
            "lineage": s.builder_lineage, "stored": s.stored_resource,
        })
    structures_save = {"structures_list": struct_list, "total_built": structures.total_built}
    physics_save = {
        "temperature": physics.temperature.tolist(),
        "mineral": physics.mineral.tolist(),
        "toxin": physics.toxin.tolist(),
        "fertility": physics.fertility.tolist(),
    }
    config_dict = {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")}
    save_checkpoint(
        final_ckpt, cfg.max_ticks, agents, grid.cells,
        physics_save, structures_save, config_dict
    )

    if not args.quiet:
        print(f"\nDone. {elapsed:.1f}s | pop={len(agents)} | gen={best_generation_ever} | {final_ckpt}")

if __name__ == "__main__":
    main()
