
import numpy as np
from config import Config
from agents.body import Body
from agents.raw_sensors import RawSensors, RAW_INPUT_DIM
from agents.internal_state import InternalState
from agents.memory import MemorySystem
from agents.self_model import SelfModel, MortalitySelfModel
from agents.concept_hypothesis import ConceptHypothesisSystem, NUM_BODY_FEATURES
from agents.meta_hypothesis import MetaHypothesisSystem
from agents.morphology import Morphology
from agents.composable_rules import ComposableRuleSystem
from agents.symbol_system import SymbolCodebook
from agents.discrete_language import DiscreteVocab
from agents.inventory import Inventory, TrustMemory
from agents.genome import (
    get_trait, get_nn_weights, get_arch_genes, get_morph_genes,
    get_concept_genes, random_genome, FIXED_GENE_COUNT
)
from agents.language import ListenerModel, SIGNAL_DIM
from neural.bottleneck_brain import BottleneckBrain
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
        self._reward_ema_slow = 0.0  # long-term reward average (nostalgia)
        self._reward_ema_fast = 0.0  # recent reward average (nostalgia)
        self.ticks_alive = 0
        self.children_count = 0
        self.lineage_id = self.id

        self.morphology = Morphology(get_morph_genes(genome))

        self.body = Body(genome, cfg, position)

        self.sensors = RawSensors(genome, cfg)

        self.internal = InternalState(genome)

        stm_cap = int(round(get_trait(genome, "stm_capacity")))
        ltm_cap = int(round(get_trait(genome, "ltm_capacity")))
        entry_dim = RAW_INPUT_DIM + cfg.proprioception_dim
        self.memory = MemorySystem(stm_cap, ltm_cap, entry_dim, cfg.memory_summary_dim)

        concept_genes = get_concept_genes(genome)
        self.bottleneck_size = int(np.clip(round(concept_genes[0]), 2, 16))
        self.think_steps = int(np.clip(round(concept_genes[1]), 0, 8))

        extra_sensor = self.morphology.extra_sensor_dim
        context_dim = cfg.context_dim + cfg.internal_state_dim + extra_sensor

        self.brain = BottleneckBrain(
            raw_input_dim=RAW_INPUT_DIM,
            context_dim=context_dim,
            action_dim=cfg.action_dim,
            arch_genes=get_arch_genes(genome),
            concept_genes=concept_genes,
        )

        sm_input = self.brain.gru_size + cfg.internal_state_dim + cfg.proprioception_dim
        self.self_model = SelfModel(sm_input, cfg.self_model_hidden, cfg.internal_state_dim)

        self.listener = ListenerModel(cfg.signal_dim)

        self.concept_hyp = ConceptHypothesisSystem(
            self.bottleneck_size, max_hyp=8, action_dim=cfg.action_dim
        )

        self.composable = ComposableRuleSystem(max_rules=6, action_dim=cfg.action_dim)
        self.meta = MetaHypothesisSystem(max_meta_rules=6, max_experiments=2)

        from agents.hypothesis import HypothesisSystem
        self.hypotheses = HypothesisSystem(max_hypotheses=8, action_dim=cfg.action_dim)

        # Phase 1: Symbol system (VQ codebook)
        n_symbols = int(round(get_trait(genome, "n_symbols")))
        n_symbol_slots = int(round(get_trait(genome, "n_symbol_slots")))
        symbol_lr = get_trait(genome, "symbol_lr")
        self.symbols = SymbolCodebook(
            self.bottleneck_size, n_symbols=n_symbols,
            n_slots=n_symbol_slots, lr=symbol_lr,
        )

        # Phase 2: Recursive thought depth gate
        self.depth_gate_threshold = get_trait(genome, "depth_gate_threshold")

        # Phase 3: Discrete language
        vocab_active = int(round(get_trait(genome, "vocab_active")))
        utterance_length = int(round(get_trait(genome, "utterance_length")))
        listen_weight = get_trait(genome, "listen_weight")
        self.discrete_vocab = DiscreteVocab(
            self.bottleneck_size, vocab_active=vocab_active,
            utterance_length=utterance_length, listen_weight=listen_weight,
        )

        # Phase 4: Mortality awareness
        mortality_sensitivity = get_trait(genome, "mortality_sensitivity")
        self.mortality = MortalitySelfModel(
            self.bottleneck_size, mortality_sensitivity=mortality_sensitivity,
        )

        # Phase 5: Society engine
        inv_capacity = int(round(get_trait(genome, "inventory_capacity")))
        self.inventory = Inventory(
            inv_capacity,
            get_trait(genome, "utility_pref_0"),
            get_trait(genome, "utility_pref_1"),
        )
        self.trust_memory = TrustMemory()
        self._trade_willingness = get_trait(genome, "trade_willingness")
        self._teaching_receptivity = get_trait(genome, "trust_sensitivity")  # reuse as receptivity
        self._pending_social_reward = 0.0
        self._pending_positive_interaction = 0.0
        self._pending_negative_interaction = 0.0
        self._nearby_agent_count = 0
        self._lineage_study_cooldown = 0
        self._social_sensitivity_k = get_trait(genome, "social_sensitivity")

        # Attachment: track specific partners (agent_id -> interaction_count)
        self._bonds: dict[int, float] = {}  # agent_id -> bond_strength [0-1]
        self._max_bonds = 4

        # Reciprocity: track who helped me (agent_id -> debt)
        self._debts: dict[int, float] = {}  # positive = I owe them

        # Grief: decaying pain from bonded partner death
        self._grief_level = 0.0

        # Parental care tracking
        self._children_ids: list[int] = []  # my children still alive

        nn_weights = get_nn_weights(genome)
        n_policy = self.brain.policy_param_count
        if len(nn_weights) >= n_policy:
            self.brain.set_policy_params(nn_weights[:n_policy])

        self.learning_rate = get_trait(genome, "learning_rate")

        self._raw_input = np.zeros(RAW_INPUT_DIM)
        self._extra_sensor = np.array([])
        self._action = np.zeros(cfg.action_dim)
        self._value = 0.0
        self._prev_value = 0.0
        self._body_features = np.zeros(NUM_BODY_FEATURES)
        self._think_steps_used = 0
        self._hyp_rng = None
        self._feature_vec = None
        self._heard_signal = np.zeros(cfg.signal_dim)
        self._heard_tokens = None  # TokenUtterance or None
        self._last_spoken_tokens = None  # np.ndarray or None

    def _ensure_systems(self, rng):
        if len(self.concept_hyp.hypotheses) == 0:
            self.concept_hyp.init_random(rng)
        if len(self.composable.rules) == 0:
            self.composable.init_random(rng)
        if len(self.hypotheses.hypotheses) == 0:
            self.hypotheses.init_random(rng)


    def perceive(self, grid, light_level, rng, season=0.0, physics=None,
                 nearby_agents=0, heard_signal=None, ecology=None,
                 structures=None, disasters=None):
        self._hyp_rng = rng
        self._ensure_systems(rng)

        self._raw_input = self.sensors.perceive(
            grid, self.body.position, self.body.heading, light_level, rng,
            physics=physics, ecology=ecology,
        )

        if heard_signal is not None:
            self._heard_signal = heard_signal
        else:
            self._heard_signal = np.zeros(self._cfg.signal_dim)

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

        # Disaster warning signals — subtle precursors that big brains can learn
        self._disaster_warnings = np.zeros(4)
        if disasters is not None:
            x, y = self.body.x, self.body.y
            warnings = disasters.get_disaster_warnings(x, y)
            self._disaster_warnings = np.array([
                warnings["tremor"],
                warnings["flood"],
                warnings["drought"],
                warnings["plague"],
            ])

        self._body_features = np.array([
            self.body.energy,
            self.body.health,
            self.internal.hunger,
            self.internal.fear,
            self.internal.curiosity,
            self.internal.temperature_comfort,
        ])

        # Gather v2 features from physics/ecology for hypothesis system
        v2 = {}
        x, y = self.body.x, self.body.y
        if physics is not None:
            ts = physics.get_terrain_sensor(x, y)
            v2["water"] = ts[1]
            v2["height"] = ts[0]
            v2["slope"] = ts[2]
            v2["earthquake"] = ts[3]
            v2["fertility"] = float(physics.fertility[x % physics.w, y % physics.h])
            v2["radiation_effect"] = physics.hidden.get_radiation_damage(x, y)
            v2["soil_ph_effect"] = physics.hidden.get_plant_growth_modifier(x, y)
            v2["magnetic_strength"] = physics.hidden.get_magnetic_strength(x, y)
            ss = physics.get_substance_sensor(x, y, n=1)
            v2["substance_top_conc"] = ss[1] if len(ss) > 1 else 0.0
        if ecology is not None:
            v2["plant_density"] = ecology.get_plant_density(x, y)
            v2["herbivore_density"] = ecology.get_herbivore_density(x, y)
            v2["predator_density"] = ecology.get_predator_density(x, y)
            v2["dead_matter"] = float(ecology.dead_matter[x % ecology.w, y % ecology.h])

        self._feature_vec = self.hypotheses.build_feature_vector(
            self._raw_input[:40],
            self.body.energy, self.body.health,
            self.internal.hunger, self.internal.fear, self.internal.curiosity,
            season, light_level,
            mineral_carried=self.body.mineral_carried,
            speed=self.body.speed,
            temp_comfort=self.internal.temperature_comfort,
            age_ratio=self.body.age / max(1, self._cfg.max_lifespan),
            **v2,
        )

        # Read inscriptions from nearby structures
        self._read_inscriptions = []
        if structures is not None:
            inscriptions = structures.read_inscriptions(x, y)
            for insc in inscriptions:
                decoded = self.discrete_vocab.decode_utterance(insc.tokens)
                self._read_inscriptions.append(decoded)


    def prepare_context(self):
        sm_input = np.concatenate([
            self.brain.hidden_state,
            self.internal.as_vector(),
            self.body.get_proprioception(),
        ])
        expected_sm = self.brain.gru_size + self._cfg.internal_state_dim + self._cfg.proprioception_dim
        sm_padded = np.zeros(expected_sm)
        sm_padded[:min(len(sm_input), expected_sm)] = sm_input[:min(len(sm_input), expected_sm)]
        self_prediction = self.self_model.predict(sm_padded)

        experience = np.concatenate([self._raw_input, self.body.get_proprioception()])
        mem_summary = self.memory.get_summary(experience)

        # Phase 4: Mortality awareness feeds into context
        mortality_vec = np.array([
            self.mortality.survival_prob,
            self.mortality.time_to_death,
            self.mortality.get_death_awareness(),
        ])

        # Phase 3: Decoded token meaning from heard utterance
        heard_meaning = np.zeros(self.bottleneck_size)
        if self._heard_tokens is not None:
            heard_meaning = self.discrete_vocab.decode_utterance(self._heard_tokens.tokens)

        # Inscription reading: average decoded meanings from structures
        if hasattr(self, '_read_inscriptions') and self._read_inscriptions:
            insc_sum = np.zeros(self.bottleneck_size)
            for decoded in self._read_inscriptions:
                n = min(len(decoded), self.bottleneck_size)
                insc_sum[:n] += decoded[:n]
            insc_sum /= len(self._read_inscriptions)
            # Blend inscription knowledge into heard meaning
            heard_meaning = heard_meaning + insc_sum * 0.5

        self._context = np.concatenate([
            self.body.get_proprioception(),
            self.internal.as_vector(),
            mem_summary,
            self_prediction,
            self._heard_signal,
            self._extra_sensor,
            mortality_vec,
            heard_meaning[:min(4, len(heard_meaning))],  # first 4 dims of decoded meaning
            self._disaster_warnings,  # 4 dims: tremor, flood, drought, plague
        ])
        return self._context

    def apply_gpu_result(self, concepts, actions, value, hidden):
        self.brain._last_bottleneck = concepts
        self.brain._last_hidden = hidden.copy()
        self.brain.gru.h = hidden.copy()
        self.brain._last_value = value
        self.brain._last_action = actions.copy()
        self._prev_value = self._value
        self._value = value
        self._action = actions.copy()
        self._think_steps_used = 0

        # Phase 1: Quantize concepts into discrete symbols
        self.symbols.quantize(concepts)
        self.symbols.learn(concepts)

        concept_bias = self.concept_hyp.get_action_bias(concepts, self._body_features)
        comp_bias = np.zeros(len(self._action))
        if self._feature_vec is not None:
            comp_bias = self.composable.record_and_evaluate(self._feature_vec)

        n = len(self._action)
        cb = concept_bias[:n] if len(concept_bias) >= n else np.pad(concept_bias, (0, n - len(concept_bias)))
        cpb = comp_bias[:n] if len(comp_bias) >= n else np.pad(comp_bias, (0, n - len(comp_bias)))
        self._action = np.tanh(self._action + cb * 0.4 + cpb * 0.3)
        return self._action

    def think(self):
        self.prepare_context()

        self._prev_value = self._value
        self._action, self._value, self._think_steps_used = self.brain.think(
            self._raw_input, self._context,
            depth_gate_threshold=self.depth_gate_threshold,
        )

        concepts = self.brain.get_concepts()

        # Phase 1: Quantize concepts into discrete symbols
        self.symbols.quantize(concepts)
        self.symbols.learn(concepts)

        concept_bias = self.concept_hyp.get_action_bias(concepts, self._body_features)

        comp_bias = np.zeros(len(self._action))
        if self._feature_vec is not None:
            comp_bias = self.composable.record_and_evaluate(self._feature_vec)

        n = len(self._action)
        cb = concept_bias[:n] if len(concept_bias) >= n else np.pad(concept_bias, (0, n - len(concept_bias)))
        cpb = comp_bias[:n] if len(comp_bias) >= n else np.pad(comp_bias, (0, n - len(comp_bias)))

        self._action = np.tanh(self._action + cb * 0.4 + cpb * 0.3)

        return self._action


    def act(self):
        a = self._action
        speed = self.morphology.speed_multiplier

        # 6 generic force channels — physics determines meaning from context
        # a[0]: move_x        — movement force
        # a[1]: move_y        — movement force
        # a[2]: mouth         — intake/output with environment (+absorb, -emit)
        # a[3]: social        — directed force toward nearby agents (+give, -take)
        # a[4]: manipulate    — transform/combine/build force
        # a[5]: signal        — communication emission intensity

        mouth = float(a[2])       # [-1, 1]
        social = float(a[3])      # [-1, 1]
        manipulate = float(a[4])  # [-1, 1]
        signal_raw = float(a[5])  # [-1, 1]

        # Utterance: derived from signal channel + brain concepts
        # Signal strength determines if/how much the agent broadcasts
        signal_strength = (signal_raw + 1) / 2 * self.morphology.visibility
        concepts = self.brain.get_concepts()
        token_utterance = self.discrete_vocab.encode_utterance(concepts, signal_raw)
        if token_utterance is not None:
            self._last_spoken_tokens = token_utterance.copy()

        # Utterance vector: concepts modulated by signal
        utterance = concepts[:self._cfg.signal_dim] * signal_strength if len(concepts) >= self._cfg.signal_dim else np.zeros(self._cfg.signal_dim)

        return {
            "move_dx": float(a[0]) * speed,
            "move_dy": float(a[1]) * speed,
            "mouth": mouth,                    # [-1,1] +absorb -emit
            "social": social,                  # [-1,1] +give -take
            "manipulate": manipulate,          # [-1,1] +craft/build -deconstruct
            "signal": signal_strength,         # [0,1] broadcast intensity
            "utterance": utterance,
            "token_utterance": token_utterance,
        }


    def update(self, reward, energy_gained=0.0, damage_taken=0.0,
               temperature=0.5, mineral_found=0.0, toxin_exposure=0.0,
               plant_found=0.0, predator_near=0.0, substance_conc=0.0,
               water_here=0.0, height_here=0.0, radiation_dmg=0.0,
               medicine_result=0.0, fertility_here=0.0):
        self.ticks_alive += 1
        self.total_reward += reward

        # Reward EMAs for nostalgia: slow tracks long-term, fast tracks recent
        alpha_slow = 0.001
        alpha_fast = 0.05
        self._reward_ema_slow += alpha_slow * (reward - self._reward_ema_slow)
        self._reward_ema_fast += alpha_fast * (reward - self._reward_ema_fast)

        # Memory fullness: how much past experience is stored
        memory_fullness = self.memory.ltm.count / max(1, self.memory.ltm.capacity)

        effective_damage = damage_taken * (1.0 - self.morphology.damage_reduction)

        self.internal.update(
            self.body.energy, self.body.health,
            self._raw_input, self.self_model.surprise,
            temperature=temperature,
            nearby_agents=self._nearby_agent_count,
            positive_interaction=self._pending_positive_interaction,
            negative_interaction=self._pending_negative_interaction,
            social_reward=self._pending_social_reward,
            reward_ema_slow=self._reward_ema_slow,
            reward_ema_fast=self._reward_ema_fast,
            memory_fullness=memory_fullness,
        )
        # Reset pending social signals
        self._pending_social_reward = 0.0
        self._pending_positive_interaction = 0.0
        self._pending_negative_interaction = 0.0

        # Trust memory decay
        if self.ticks_alive % 20 == 0:
            self.trust_memory.decay(self.ticks_alive)

        # Social state decay (bonds, debts, grief)
        self.decay_social_state()

        actual_state = self.internal.as_vector()
        self.self_model.observe_actual(actual_state, self.learning_rate)

        self.brain.learn_world_model(self.brain.get_concepts())

        # Phase 4: Mortality update
        self.mortality.update_trends(self.body.energy, self.body.health, effective_damage)
        concepts = self.brain.get_concepts()
        age_ratio = self.body.age / max(1, self._cfg.max_lifespan)
        self.mortality.predict_survival(concepts, self.body.energy, self.body.health, age_ratio)
        self.mortality.learn_from_survival(survived=True, learning_rate=self.learning_rate)

        # Phase 3: Discrete language grounding
        if self._last_spoken_tokens is not None:
            for tid in self._last_spoken_tokens:
                self.discrete_vocab.ground_speaker(int(tid), concepts, lr=self.learning_rate)
            self._last_spoken_tokens = None
        if self._heard_tokens is not None:
            outcome_good = energy_gained > 0 or effective_damage < 0.01
            for tid in self._heard_tokens.tokens:
                self.discrete_vocab.ground_listener(
                    int(tid), concepts, outcome_good, lr=self.learning_rate * 0.5)
            self._heard_tokens = None

        experience = np.concatenate([self._raw_input, self.body.get_proprioception()])
        self.memory.store_experience(experience, reward)

        concepts = self.brain.get_concepts()
        self.concept_hyp.test_all(concepts, self._body_features, energy_gained, effective_damage,
                                  plant_found=plant_found, predator_near=predator_near,
                                  water_found=water_here, substance_useful=substance_conc)

        if self._feature_vec is not None:
            outcome_vec = np.array([
                energy_gained, -effective_damage,
                self._raw_input[0] if len(self._raw_input) > 0 else 0,
                self._raw_input[1] if len(self._raw_input) > 1 else 0,
                1.0 if effective_damage < 0.001 else 0.0,
                mineral_found, toxin_exposure,
                abs(temperature - 0.5) * 2,
                plant_found, predator_near, substance_conc,
                water_here, height_here, 0.0,  # reaction_seen placeholder
                0.0,  # structure_nearby placeholder
                radiation_dmg, medicine_result, fertility_here,
            ])
            self.hypotheses.test_hypotheses(self._feature_vec, outcome_vec)
            self.composable.test_rules(self._feature_vec, outcome_vec)
            if self._hyp_rng is not None:
                self.meta.update(
                    self.hypotheses.hypotheses, self._feature_vec,
                    outcome_vec, self._hyp_rng,
                )

        if np.any(self._heard_signal != 0):
            outcome = np.array([energy_gained, -effective_damage,
                                1.0 if effective_damage < 0.001 else 0.0])
            self.listener.predict(self._heard_signal)
            self.listener.learn(outcome, self.learning_rate)

        if self.ticks_alive % 50 == 0 and self._hyp_rng is not None:
            self.concept_hyp.evolve(self._hyp_rng)
            self.hypotheses.evolve_hypotheses(self._hyp_rng)
            self.composable.evolve(self._hyp_rng)

        td_error = reward + self._cfg.discount_gamma * self._value - self._prev_value
        self._apply_td_update(td_error)

        brain_cost = self._cfg.brain_metabolic_cost * self.brain.param_count
        morph_cost = self.morphology.total_metabolic_cost
        think_cost = self._think_steps_used * self._cfg.think_energy_cost

        # Isolation stress: high social_need increases metabolic cost
        # This is physics — loneliness literally costs energy (stress hormones)
        isolation_cost = self.internal.social_need * 0.002 * self._social_sensitivity_k
        # Grief cost: recent partner loss adds stress
        grief_cost = self._grief_level * 0.003

        temp_modifier = 0.7 + 0.6 * temperature
        total_cost = brain_cost + morph_cost + think_cost + isolation_cost + grief_cost
        self.body.metabolize(total_cost, temp_modifier)
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

        wm_surprise = min(1.0, self.brain.world_prediction_error * 5)
        reward += wm_surprise * self._cfg.exploration_bonus_scale * 0.5

        ch_stats = self.concept_hyp.stats
        if ch_stats["n_rules"] > 0:
            reward += ch_stats["mean_accuracy"] * 0.02

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

    def strengthen_bond(self, other_id: int, amount: float = 0.05):
        """Strengthen attachment bond with a specific agent."""
        old = self._bonds.get(other_id, 0.0)
        self._bonds[other_id] = min(1.0, old + amount)
        # Prune weakest if over limit
        if len(self._bonds) > self._max_bonds:
            weakest = min(self._bonds, key=self._bonds.get)
            if self._bonds[weakest] < 0.1:
                del self._bonds[weakest]

    def add_debt(self, helper_id: int, amount: float = 0.1):
        """Record that helper_id helped me — I owe them."""
        old = self._debts.get(helper_id, 0.0)
        self._debts[helper_id] = min(1.0, old + amount)

    def get_reciprocity_urge(self, other_id: int) -> float:
        """How much do I feel I should help this agent?"""
        debt = self._debts.get(other_id, 0.0)
        bond = self._bonds.get(other_id, 0.0)
        return debt * 0.5 + bond * 0.3

    def observe_partner_death(self, dead_id: int):
        """A bonded partner died — grief proportional to bond strength."""
        bond = self._bonds.pop(dead_id, 0.0)
        # No threshold — any bond produces proportional grief
        self._grief_level = min(1.0, self._grief_level + bond * 0.8)
        self._debts.pop(dead_id, None)

    def decay_social_state(self):
        """Decay grief, bonds, debts over time."""
        self._grief_level *= 0.995  # slow decay
        # Bonds decay if no interaction
        dead_bonds = []
        for aid in self._bonds:
            self._bonds[aid] *= 0.999
            if self._bonds[aid] < 0.01:
                dead_bonds.append(aid)
        for aid in dead_bonds:
            del self._bonds[aid]
        # Debts decay
        dead_debts = []
        for aid in self._debts:
            self._debts[aid] *= 0.998
            if self._debts[aid] < 0.01:
                dead_debts.append(aid)
        for aid in dead_debts:
            del self._debts[aid]

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
            if i < len(self.composable.rules):
                # Inherit proportionally: accuracy acts as weight, not gate
                self.composable.rules[i] = parent_agent.composable._mutate(rule, rng)

    def inherit_concept_hypotheses(self, parent_agent, rng):
        for i, hyp in enumerate(parent_agent.concept_hyp.hypotheses):
            if i < len(self.concept_hyp.hypotheses):
                self.concept_hyp.hypotheses[i] = self.concept_hyp._mutate(hyp, rng)

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
        from neural.bottleneck_brain import BottleneckBrain
        max_arch = np.array([4, 256, 256, 128, 64, 128])
        max_concept = np.array([32, 8, 0.05])
        extra = 9
        context_dim = cfg.context_dim + cfg.internal_state_dim + extra
        brain = BottleneckBrain(RAW_INPUT_DIM, context_dim, cfg.action_dim,
                                max_arch, max_concept)
        return brain.policy_param_count

    @classmethod
    def create_random(cls, cfg, position, rng):
        max_nn = cls.estimate_max_nn_params(cfg)
        genome = random_genome(cfg, max_nn, rng)
        agent = cls(genome, cfg, position)
        agent.body.init_genetic_frailty(rng)
        return agent
