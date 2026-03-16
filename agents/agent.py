
import numpy as np
from config import Config
from agents.body import Body
from agents.sensors import Sensors
from agents.internal_state import InternalState
from agents.memory import MemorySystem
from agents.self_model import SelfModel
from agents.hypothesis import HypothesisSystem
from agents.meta_hypothesis import MetaHypothesisSystem
from agents.morphology import Morphology
from agents.composable_rules import ComposableRuleSystem
from agents.genome import (
    get_trait, get_nn_weights, get_arch_genes, get_morph_genes,
    random_genome, FIXED_GENE_COUNT
)
from neural.evolvable import EvolvableBrain
from world.grid import Grid


_next_id = 0


def _gen_id() -> int:
    global _next_id
    _next_id += 1
    return _next_id


class Agent:
    def __init__(self, genome: np.ndarray, cfg: Config, position: np.ndarray,
                 parent_id: int = -1):
        self.id = _gen_id()
        self.parent_id = parent_id
        self.genome = genome.copy()
        self._cfg = cfg
        self.generation = 0
        self.total_reward = 0.0
        self.ticks_alive = 0
        self.children_count = 0
        self.lineage_id = self.id

        self.morphology = Morphology(get_morph_genes(genome))

        self.body = Body(genome, cfg, position)
        self.sensors = Sensors(genome, cfg)
        self.internal = InternalState(genome)

        stm_cap = int(round(get_trait(genome, "stm_capacity")))
        ltm_cap = int(round(get_trait(genome, "ltm_capacity")))
        entry_dim = self.sensors.output_dim + cfg.proprioception_dim
        self.memory = MemorySystem(stm_cap, ltm_cap, entry_dim, cfg.memory_summary_dim)

        extra_sensor = self.morphology.extra_sensor_dim
        total_input = cfg.total_brain_input_dim + extra_sensor
        self.brain = EvolvableBrain(total_input, cfg.action_dim, get_arch_genes(genome))

        sm_input = self.brain.gru_size + cfg.internal_state_dim + cfg.proprioception_dim
        self.self_model = SelfModel(sm_input, cfg.self_model_hidden, cfg.internal_state_dim)

        self.hypotheses = HypothesisSystem(max_hypotheses=8, action_dim=cfg.action_dim)
        self.meta = MetaHypothesisSystem(max_meta_rules=6, max_experiments=2)
        self.composable = ComposableRuleSystem(max_rules=6, action_dim=cfg.action_dim)

        nn_weights = get_nn_weights(genome)
        n_policy = self.brain.policy_param_count
        if len(nn_weights) >= n_policy:
            self.brain.set_policy_params(nn_weights[:n_policy])

        self.learning_rate = get_trait(genome, "learning_rate")

        self._sensor_vec = np.zeros(self.sensors.output_dim)
        self._extra_sensor = np.array([])
        self._action = np.zeros(cfg.action_dim)
        self._value = 0.0
        self._prev_value = 0.0
        self._feature_vec = None
        self._hyp_rng = None

    def _ensure_systems(self, rng):
        if len(self.hypotheses.hypotheses) == 0:
            self.hypotheses.init_random(rng)
        if len(self.composable.rules) == 0:
            self.composable.init_random(rng)

    def perceive(self, grid, light_level, rng, season=0.0, physics=None,
                 structures=None, nearby_agents=0):
        self._hyp_rng = rng
        self._ensure_systems(rng)

        self._sensor_vec = self.sensors.perceive(
            grid, self.body.position, self.body.heading, light_level, rng,
            physics=physics,
        )

        if physics is not None and self.morphology.extra_sensor_dim > 0:
            x, y = self.body.x, self.body.y
            self._extra_sensor = self.morphology.get_extra_sensor_input(
                temperature=physics.get_temperature(x, y),
                toxin=physics.get_toxin(x, y),
                mineral=physics.get_mineral(x, y),
                scent=physics.scent[x % physics.w, y % physics.h],
                abs_x=x / max(1, self._cfg.world_width),
                abs_y=y / max(1, self._cfg.world_height),
                nearby_movement=min(1.0, nearby_agents * 0.2),
            )
        else:
            self._extra_sensor = np.array([])

        self._feature_vec = self.hypotheses.build_feature_vector(
            self._sensor_vec,
            self.body.energy, self.body.health,
            self.internal.hunger, self.internal.fear, self.internal.curiosity,
            season, light_level,
            mineral_carried=self.body.mineral_carried,
            speed=self.body.speed,
            temp_comfort=self.internal.temperature_comfort,
        )

    def think(self):
        sm_input = np.concatenate([
            self.brain.hidden_state,
            self.internal.as_vector(),
            self.body.get_proprioception(),
        ])
        expected_sm = self.brain.gru_size + self._cfg.internal_state_dim + self._cfg.proprioception_dim
        sm_padded = np.zeros(expected_sm)
        sm_padded[:min(len(sm_input), expected_sm)] = sm_input[:min(len(sm_input), expected_sm)]
        self_prediction = self.self_model.predict(sm_padded)

        experience = np.concatenate([self._sensor_vec, self.body.get_proprioception()])
        mem_summary = self.memory.get_summary(experience)

        brain_input = np.concatenate([
            self._sensor_vec,
            self.body.get_proprioception(),
            self.internal.as_vector(),
            mem_summary,
            self_prediction,
            self._extra_sensor,
        ])

        self._prev_value = self._value
        self._action, self._value = self.brain.forward(brain_input)

        if self._feature_vec is not None:
            hyp_bias = self.hypotheses.get_action_bias(self._feature_vec)
            comp_bias = self.composable.record_and_evaluate(self._feature_vec)
            exp_bias = self.meta.get_experiment_bias(len(self._action))
            n = min(len(self._action), len(hyp_bias), len(comp_bias))
            self._action[:n] = np.tanh(
                self._action[:n]
                + hyp_bias[:n] * 0.4
                + comp_bias[:n] * 0.3
                + exp_bias[:n] * 0.2
            )

        return self._action

    def act(self):
        a = self._action
        speed = self.morphology.speed_multiplier
        return {
            "move_dx": float(a[0]) * speed,
            "move_dy": float(a[1]) * speed,
            "eat": float(a[2]) > 0,
            "reproduce": float(a[3]) > 0.5,
            "signal": float((a[4] + 1) / 2) * self.morphology.visibility,
            "turn": float(a[5]) * 0.3,
            "collect_mineral": float(a[6]) > 0 if len(a) > 6 else False,
            "combine": float(a[7]) > 0.3 if len(a) > 7 else False,
            "build": self._get_build_action(a),
            "deposit": float(a[2]) > 0.8 and self.body.energy > 0.5,
            "withdraw": float(a[2]) > 0 and self.body.energy < 0.3,
        }

    def _get_build_action(self, a) -> int:
        if len(a) < 8:
            return 0
        build_intent = float(a[4]) > 0.7 and float(a[7]) > 0.5 and self.body.energy > 0.4
        if not build_intent:
            return 0
        pattern = float(a[0]) + float(a[1]) * 2 + float(a[5]) * 3
        stype = int(abs(pattern * 3) % 6) + 1
        return stype

    def update(self, reward, energy_gained=0.0, damage_taken=0.0,
               temperature=0.5, mineral_found=0.0, toxin_exposure=0.0):
        self.ticks_alive += 1
        self.total_reward += reward

        effective_damage = damage_taken * (1.0 - self.morphology.damage_reduction)

        self.internal.update(
            self.body.energy, self.body.health,
            self._sensor_vec, self.self_model.surprise,
            temperature=temperature,
        )

        actual_state = self.internal.as_vector()
        self.self_model.observe_actual(actual_state, self.learning_rate)

        experience = np.concatenate([self._sensor_vec, self.body.get_proprioception()])
        self.memory.store_experience(experience, reward)

        if self._feature_vec is not None:
            outcome_vec = np.array([
                energy_gained, -effective_damage,
                self._sensor_vec[0] if len(self._sensor_vec) > 0 else 0,
                self._sensor_vec[1] if len(self._sensor_vec) > 1 else 0,
                1.0 if effective_damage < 0.001 else 0.0,
                mineral_found, toxin_exposure,
                abs(temperature - 0.5) * 2,
            ])
            self.hypotheses.test_hypotheses(self._feature_vec, outcome_vec)
            self.composable.test_rules(self._feature_vec, outcome_vec)
            if self._hyp_rng is not None:
                self.meta.update(
                    self.hypotheses.hypotheses, self._feature_vec,
                    outcome_vec, self._hyp_rng,
                )

        if self.ticks_alive % 50 == 0 and self._hyp_rng is not None:
            self.hypotheses.evolve_hypotheses(self._hyp_rng)
            self.composable.evolve(self._hyp_rng)

        td_error = reward + self._cfg.discount_gamma * self._value - self._prev_value
        self._apply_td_update(td_error)

        brain_cost = self._cfg.brain_metabolic_cost * self.brain.param_count
        morph_cost = self.morphology.total_metabolic_cost
        temp_modifier = 0.7 + 0.6 * temperature
        self.body.metabolize(brain_cost + morph_cost, temp_modifier)
        self.body.heal()

    def compute_intrinsic_reward(self, energy_gained, damage_taken,
                                  mineral_collected=0.0, medicine_healed=0.0,
                                  structure_built=False, teaching_done=False):
        reward = energy_gained * 2.0
        reward -= damage_taken * 3.0
        reward += mineral_collected * 0.5
        reward += medicine_healed * 3.0
        if structure_built:
            reward += 0.5
        if teaching_done:
            reward += 0.3
        reward += self.self_model.surprise * self._cfg.exploration_bonus_scale * self.internal.curiosity

        hyp_stats = self.hypotheses.stats
        if hyp_stats["n_rules"] > 0:
            reward += hyp_stats["mean_accuracy"] * 0.01

        comp_stats = self.composable.stats
        if comp_stats["n_complex_rules"] > 0:
            reward += comp_stats["mean_accuracy"] * 0.02

        reward += 0.001
        return reward

    def _apply_td_update(self, td_error):
        lr = self.learning_rate
        value_params = self.brain.get_value_params()
        hidden = self.brain.hidden_state
        v_grad = np.zeros_like(value_params)
        n_weights = min(hidden.shape[0], len(v_grad) - 1)
        if n_weights > 0:
            v_grad[:n_weights] = -td_error * hidden[:n_weights]
            v_grad[n_weights] = -td_error
        v_grad = np.clip(v_grad, -1.0, 1.0)
        value_params += lr * v_grad
        self.brain.set_value_params(value_params)

    def can_reproduce(self):
        threshold = get_trait(self.genome, "reproduction_threshold")
        return (
            self.body.energy > threshold
            and self.body.age > self._cfg.maturity_age
            and self.body.is_alive()
        )

    def get_hypothesis_data(self):
        return self.hypotheses.encode_all()

    def inherit_hypotheses(self, parent_data, rng):
        self.hypotheses.decode_all(parent_data)
        self.hypotheses.evolve_hypotheses(rng)

    def inherit_composable_rules(self, parent_agent, rng):
        for i, rule in enumerate(parent_agent.composable.rules):
            if i < len(self.composable.rules) and rule.accuracy > 0.5:
                self.composable.rules[i] = parent_agent.composable._mutate(rule, rng)

    @property
    def is_alive(self):
        return self.body.is_alive()

    @property
    def x(self):
        return self.body.x

    @property
    def y(self):
        return self.body.y

    @classmethod
    def estimate_max_nn_params(cls, cfg):
        from neural.evolvable import EvolvableBrain
        max_arch = np.array([4, 128, 128, 64, 32, 64])
        extra = 9
        total_input = cfg.total_brain_input_dim + extra
        brain = EvolvableBrain(total_input, cfg.action_dim, max_arch)
        return brain.policy_param_count

    @classmethod
    def create_random(cls, cfg, position, rng):
        max_nn = cls.estimate_max_nn_params(cfg)
        genome = random_genome(cfg, max_nn, rng)
        return cls(genome, cfg, position)
