
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
from world.social import SocialSystem
from world.structures import StructureManager, StructureType, BUILD_COST
from world.culture import CultureSystem
from agents.agent_v4 import Agent
from agents.language import ProtoLanguage
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


def resolve_actions(grid, physics, structures, agents, actions, cfg, evo_rng):
    new_agents = []
    agent_data = {}

    for agent, action in zip(agents, actions):
        if not agent.is_alive:
            continue

        data = {"energy_gained": 0.0, "damage": 0.0, "mineral": 0.0,
                "toxin": 0.0, "medicine": 0.0, "temperature": 0.5,
                "structure_built": False, "teaching_done": False}

        new_dx = action["move_dx"]
        new_dy = action["move_dy"]
        new_x = int(round(agent.body.position[0] + new_dx)) % cfg.world_width
        new_y = int(round(agent.body.position[1] + new_dy)) % cfg.world_height
        if structures.is_blocked(new_x, new_y):
            new_dx *= 0.1
            new_dy *= 0.1
        agent.body.move(new_dx, new_dy, cfg.world_width, cfg.world_height)
        agent.body.heading += action["turn"]

        x, y = agent.x, agent.y
        data["temperature"] = physics.get_temperature(x, y)

        if structures.get_nest_bonus(x, y):
            agent.body.energy += 0.001

        if action["eat"]:
            consumed = grid.consume_resource(x, y, amount=0.3)
            farm_bonus = structures.get_farm_bonus(x, y)
            if farm_bonus > 0:
                consumed *= (1.0 + farm_bonus)
            agent.body.eat(consumed)
            data["energy_gained"] = consumed

        if action["collect_mineral"]:
            available = physics.consume_mineral(x, y, 0.2)
            collected = agent.body.collect_mineral(available)
            data["mineral"] = collected

        if action["combine"] and agent.body.mineral_carried > 0.1 and agent.body.energy > 0.2:
            heal = physics.apply_medicine(0.1, agent.body.mineral_carried)
            if heal > 0:
                agent.body.heal_medicine(heal)
                data["medicine"] = heal

        build_type = action["build"]
        if build_type > 0:
            cost = BUILD_COST.get(build_type, 0.15)
            if agent.body.energy > cost:
                if structures.build(build_type, x, y, agent.lineage_id):
                    agent.body.energy -= cost
                    data["structure_built"] = True

        if action["deposit"]:
            deposited = structures.deposit_resource(x, y, 0.1)
            if deposited > 0:
                agent.body.energy -= deposited

        if action["withdraw"]:
            withdrawn = structures.withdraw_resource(x, y, 0.1)
            if withdrawn > 0:
                agent.body.energy += withdrawn

        hazard = grid.get_cell(x, y)[1]
        hazard_dmg = hazard * 0.1
        agent.body.take_damage(hazard_dmg)

        trap_dmg = structures.get_trap_damage(x, y, agent.lineage_id)
        if trap_dmg > 0:
            agent.body.take_damage(trap_dmg)

        toxin = physics.get_toxin(x, y)
        toxin_dmg = toxin * 0.05
        agent.body.take_damage(toxin_dmg)
        data["toxin"] = toxin
        data["damage"] = hazard_dmg + toxin_dmg + trap_dmg

        if action["signal"] > 0.1:
            grid.stamp_agent(x, y, action["signal"])
        physics.stamp_scent(x, y, 0.3)

        if action["reproduce"] and agent.can_reproduce():
            mate = find_mate(agent, agents, cfg)
            if mate is not None:
                child = reproduce_sexual(agent, mate, cfg, evo_rng)
            else:
                child = reproduce_asexual(agent, cfg, evo_rng)
            if child is not None and len(agents) + len(new_agents) < cfg.max_population:
                new_agents.append(child)

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
    language = ProtoLanguage(cfg.world_width, cfg.world_height, cfg.hear_radius)

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
        structures.update(tick)
        grid.clear_agent_layer()
        for agent in agents:
            grid.stamp_agent(agent.x, agent.y)

        _spatial.build(agents)
        t1 = time.time()

        for agent in agents:
            nearby = count_nearby(agent, agents, radius=4)
            heard = language.get_strongest_signal(agent, tick)
            agent.perceive(grid, env.light_level, agent_rng,
                          season=env.season_phase, physics=physics,
                          nearby_agents=nearby, heard_signal=heard)
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
            grid, physics, structures, agents, actions, cfg, evo_rng
        )
        t3b = time.time()
        profile_accum["actions"] += t3b - t3

        for agent, action in zip(agents, actions):
            if agent.is_alive:
                utterance = action.get("utterance", None)
                if utterance is not None:
                    language.broadcast(agent, utterance, tick)
        language.cleanup(tick)

        social_rewards = social.resolve_social(agents, actions, tick)

        if tick % 5 == 0:
            culture.process_teaching(agents, tick, agent_rng)
            culture.process_imitation(agents, tick, agent_rng)
            culture.cleanup()

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

            agent.update(reward, energy_gained=eg, damage_taken=dmg,
                        temperature=temp, mineral_found=mineral, toxin_exposure=toxin)
        t4 = time.time()

        profile_accum["world"] += t1 - t0
        profile_accum["perceive"] += t2 - t1
        profile_accum["think"] += t3 - t2
        profile_accum["update"] += t4 - t3
        profile_accum["ticks"] += 1

        dead = [a for a in agents if not a.is_alive]
        for a in dead:
            hall_of_fame.check_dead_agent(a, tick)
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

            n = max(1, len(agents))
            mean_wm_acc /= n
            mean_think_steps /= n
            mean_bottleneck /= n
            if n_with_concepts > 0:
                mean_concept_acc /= n_with_concepts
            max_complexity_ever = max(max_complexity_ever, max_complexity)

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
                "unique_signalers": lang_stats["unique_senders"],
                "best_generation_ever": best_generation_ever,
                "max_complexity_ever": max_complexity_ever,
                "births": births_this_period,
                "deaths": deaths_this_period,
                "total_resource": float(np.sum(grid.cells[:, :, 0])),
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
