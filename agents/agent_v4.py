
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
from agents.meta_concepts import MetaConceptSystem
from agents.grammar import GrammarSystem
from agents.theory_of_mind import TheoryOfMindSystem
from agents.episodic_memory import EpisodicMemorySystem
from agents.naming import NamingSystem
from agents.goal_system import GoalSystem
from agents.concept_workspace import ConceptWorkspace
from agents.norm_system import NormSystem
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
        self._daydream_rng = np.random.default_rng(self.id)
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

        # Phase 6: Meta-concepts
        meta_window = int(round(get_trait(genome, "meta_window")))
        meta_bn_size = int(round(get_trait(genome, "meta_bottleneck_size")))
        meta_wm_lr = get_trait(genome, "meta_wm_lr")
        self.meta_concepts = MetaConceptSystem(
            self.bottleneck_size, meta_window=meta_window,
            meta_bottleneck_size=meta_bn_size, meta_wm_lr=meta_wm_lr,
            action_dim=cfg.action_dim,
        )

        # Phase 7: Compositional grammar
        n_grammar_slots = int(round(get_trait(genome, "n_grammar_slots")))
        grammar_lr = get_trait(genome, "grammar_lr")
        grammar_weight = get_trait(genome, "grammar_weight")
        self.grammar = GrammarSystem(
            self.bottleneck_size, n_slots=n_grammar_slots,
            grammar_lr=grammar_lr, grammar_weight=grammar_weight,
        )

        # Phase 8: Theory of Mind
        tom_max = int(round(get_trait(genome, "tom_max_tracked")))
        tom_window = int(round(get_trait(genome, "tom_window")))
        tom_lr = get_trait(genome, "tom_lr")
        tom_weight = get_trait(genome, "tom_weight")
        self.tom = TheoryOfMindSystem(
            self.bottleneck_size, tom_max_tracked=tom_max,
            tom_window=tom_window, tom_lr=tom_lr, tom_weight=tom_weight,
            action_dim=cfg.action_dim,
        )

        # Phase 9: Episodic memory
        episodic_cap = int(round(get_trait(genome, "episodic_capacity")))
        surprise_thresh = get_trait(genome, "episode_surprise_threshold")
        consolidation = get_trait(genome, "consolidation_rate")
        emotion_weight = get_trait(genome, "episodic_emotion_weight")
        self.episodic = EpisodicMemorySystem(
            capacity=episodic_cap, bottleneck_size=self.bottleneck_size,
            action_dim=cfg.action_dim, surprise_threshold=surprise_thresh,
            consolidation_rate=consolidation, emotion_weight=emotion_weight,
        )
        self._episodic_replay_depth = int(round(get_trait(genome, "episodic_replay_depth")))

        # Phase 10: Naming/Reference
        name_cap = int(round(get_trait(genome, "name_capacity")))
        name_lr = get_trait(genome, "name_learning_rate")
        ref_weight = get_trait(genome, "referential_weight")
        self.naming = NamingSystem(
            capacity=name_cap, bottleneck_size=self.bottleneck_size,
            learning_rate=name_lr, referential_weight=ref_weight,
        )
        self._attended_entity = None  # (entity_type, entity_id) or None

        # Phase 12: Persistent goals
        goal_depth = int(round(get_trait(genome, "goal_stack_depth")))
        commitment = get_trait(genome, "commitment_strength")
        patience = get_trait(genome, "patience")
        horizon = get_trait(genome, "goal_horizon")
        goal_comm = get_trait(genome, "goal_communication_weight")
        self.goals = GoalSystem(
            max_depth=goal_depth, commitment_strength=commitment,
            patience=patience, horizon=horizon,
            communication_weight=goal_comm,
            bottleneck_size=self.bottleneck_size,
            action_dim=cfg.action_dim,
        )

        # Phase 13: Emergence infrastructure
        workspace_slots = int(round(get_trait(genome, "workspace_slots")))
        workspace_gate = get_trait(genome, "workspace_gate")
        self.workspace = ConceptWorkspace(
            n_slots=workspace_slots,
            bottleneck_size=self.bottleneck_size,
            gate_strength=workspace_gate,
        )
        self._think_branch_count = max(1, int(round(
            get_trait(genome, "think_branch_count"))))
        norm_cap = int(round(get_trait(genome, "norm_capacity")))
        norm_sens = get_trait(genome, "norm_sensitivity")
        norm_inh = get_trait(genome, "norm_inheritance")
        self.norms = NormSystem(
            capacity=norm_cap, bottleneck_size=self.bottleneck_size,
            sensitivity=norm_sens, inheritance_rate=norm_inh,
        )
        self.naming.abstract_capacity = int(round(get_trait(genome, "abstract_naming")))
        self._temporal_encoding = get_trait(genome, "temporal_encoding")
        self.grammar.temporal_encoding = self._temporal_encoding
        # Group identity
        self._group_identity_weight = get_trait(genome, "group_identity_weight")
        self._in_group_cooperation = get_trait(genome, "in_group_cooperation")
        self._nearby_in_group_ratio = 0.0  # updated each tick from main loop
        # Phase 14: Selective pressures
        self._reputation_sensitivity = get_trait(genome, "reputation_sensitivity")
        self._group_benefit_sensitivity = get_trait(genome, "group_benefit_sensitivity")
        self.reputation = 0.0            # [-1, +1] cooperation/defection history
        self.teaching_prestige = 0.0     # [0, 1] teaching success history
        self._nearby_avg_reputation = 0.0  # updated each tick from main loop
        self._group_safety_factor = 1.0    # updated each tick from main loop

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
        self._last_structured_utterance = None  # StructuredUtterance or None

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

        # Phase 6: Meta-concept summary
        meta_vec = self.meta_concepts.get_meta_summary(max_dim=16)

        # Phase 8: Theory of Mind summary
        tom_vec = self.tom.get_tom_summary(max_dim=4)

        # Phase 9: Episodic memory context
        concepts_for_query = self.brain.get_concepts() if hasattr(self.brain, '_last_bottleneck') else np.zeros(self.bottleneck_size)
        emotional_valence = float(np.sum(np.abs(self.internal.as_vector() - 0.5)))
        episodic_vec = self.episodic.get_context_vector(
            concepts_for_query, emotional_valence, self.ticks_alive, max_dim=8)

        # Phase 10: Naming/referent context
        att_type = self._attended_entity[0] if self._attended_entity else None
        att_id = self._attended_entity[1] if self._attended_entity else None
        referent_vec = self.naming.get_referent_context(att_type, att_id, max_dim=4)

        # Phase 12: Goal context
        goal_vec = self.goals.get_goal_context(max_dim=6)

        # Phase 13: Workspace context
        workspace_vec = self.workspace.get_context_vector(concepts_for_query, max_dim=6)

        # Phase 13: Norm context
        norm_vec = self.norms.get_context_vector(max_dim=4)

        # Phase 13: Temporal context (derived from episodic memory)
        temporal_vec = np.zeros(2, dtype=np.float64)
        if self._temporal_encoding > 0.3 and self.episodic.episodes:
            last_tick = max(ep.tick for ep in self.episodic.episodes)
            recency = float(np.exp(-0.01 * max(0, self.ticks_alive - last_tick)))
            temporal_vec[0] = recency
            if len(self.episodic.episodes) >= 3:
                ticks = sorted(ep.tick for ep in self.episodic.episodes)
                intervals = np.diff(ticks).astype(np.float64)
                if len(intervals) > 1:
                    temporal_vec[1] = 1.0 / (1.0 + float(np.std(intervals) / (np.mean(intervals) + 1e-8)))
        self._temporal_context = temporal_vec

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
            meta_vec,   # 16 dims: meta-concepts (padded)
            tom_vec,    # 4 dims: ToM prediction of most-attended other
            episodic_vec,  # 8 dims: episodic memory context
            referent_vec,  # 4 dims: naming/referent context
            goal_vec,      # 6 dims: goal context
            workspace_vec, # 6 dims: workspace context (Phase 13)
            norm_vec,      # 4 dims: norm context (Phase 13)
            temporal_vec,  # 2 dims: temporal context (Phase 13)
            np.array([     # 3 dims: group identity context (Phase 13)
                self._nearby_in_group_ratio,
                self._group_identity_weight,
                self._in_group_cooperation,
            ], dtype=np.float64),
            np.array([     # 3 dims: reputation, group safety (Phase 14)
                self.reputation,
                self._nearby_avg_reputation,
                self._group_safety_factor,
            ], dtype=np.float64),
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

        # Phase 6: Record concepts into meta-concept buffer (GPU path)
        self.meta_concepts.record_concepts(concepts)
        self.meta_concepts.encode_meta()

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
            branch_count=self._think_branch_count,
            workspace=self.workspace,
        )

        concepts = self.brain.get_concepts()

        # Phase 6: Record concepts into meta-concept buffer and encode
        self.meta_concepts.record_concepts(concepts)
        self.meta_concepts.encode_meta()

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

        # Phase 12: Goal action bias
        goal_bias = self.goals.get_action_bias(concepts)
        gb = goal_bias[:n] if len(goal_bias) >= n else np.pad(goal_bias, (0, n - len(goal_bias)))

        self._action = np.tanh(self._action + cb * 0.4 + cpb * 0.3 + gb * 0.3)

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

        # Phase 12: Blend goal signal into concepts for utterance
        goal_signal = self.goals.get_goal_signal()
        if np.any(goal_signal != 0):
            n_gs = min(len(concepts), len(goal_signal))
            concepts_for_utt = concepts.copy()
            concepts_for_utt[:n_gs] = concepts[:n_gs] * 0.7 + goal_signal[:n_gs] * 0.3
        else:
            concepts_for_utt = concepts

        # Phase 10: Get referent token for naming
        referent_token = None
        if self._attended_entity is not None:
            referent_token = self.naming.encode_referent(
                self._attended_entity[0], self._attended_entity[1])

        # Phase 7: Blend structured grammar with flat token encoding
        gw = self.grammar.grammar_weight
        token_utterance = None
        if gw > 0.5:
            # Prefer structured utterance
            token_utterance = self.grammar.encode_structured(
                concepts_for_utt, signal_raw, self.discrete_vocab,
                referent_token=referent_token,
                referential_weight=self.naming.referential_weight,
                temporal_context=getattr(self, '_temporal_context', None))
        if token_utterance is None:
            # Fallback to flat encoding
            token_utterance = self.discrete_vocab.encode_utterance(concepts, signal_raw)

        if token_utterance is not None:
            # Store tokens for speaker grounding (always np.ndarray of token IDs)
            if hasattr(token_utterance, 'tokens'):
                self._last_spoken_tokens = token_utterance.tokens.copy()
            elif hasattr(token_utterance, 'copy'):
                self._last_spoken_tokens = token_utterance.copy()
            else:
                self._last_spoken_tokens = token_utterance
        self._last_structured_utterance = token_utterance

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

        # === Phase 6: Meta-concept learning ===
        meta_encoded = self.meta_concepts.encode_meta()
        self.meta_concepts.predict_next_meta(self._action)
        self.meta_concepts.learn_meta_wm(meta_encoded)
        # Meta-concept accuracy reward
        meta_acc = self.meta_concepts.cumulative_accuracy
        if meta_acc > 0.3:
            reward += meta_acc * 0.02

        # === Phase 7: Grammar grounding ===
        if hasattr(self, '_last_structured_utterance') and self._last_structured_utterance is not None:
            utt = self._last_structured_utterance
            concepts_now = self.brain.get_concepts()
            if hasattr(utt, 'roles'):
                for slot_idx in range(min(len(utt.roles), self.grammar.n_slots)):
                    self.grammar.ground_speaker_roles(
                        int(utt.roles[slot_idx]), concepts_now, self.learning_rate)
            self._last_structured_utterance = None

        # === Phase 8: Theory of Mind update ===
        if self.ticks_alive % 5 == 0:
            proximity_map = {}
            self.tom.update_attention(self._bonds, proximity_map, self.ticks_alive)
        # ToM accuracy reward
        tom_acc = self.tom.cumulative_accuracy
        if tom_acc > 0.3:
            reward += tom_acc * 0.02

        # === Phase 9: Episodic memory ===
        concepts_now = self.brain.get_concepts()
        emotional_val = float(np.sum(np.abs(self.internal.as_vector() - 0.5)))
        nearby_ids = [aid for aid in self._bonds.keys()][:3]
        self.episodic.maybe_store(
            concepts_now, self.body.position, self.ticks_alive,
            self.internal.as_vector(), nearby_ids, self._action,
            reward, self.self_model.surprise,
        )
        if self.ticks_alive % 50 == 0:
            self.episodic.consolidate(self.ticks_alive)

        # === Phase 10: Naming decay ===
        if self.ticks_alive % 20 == 0:
            self.naming.decay(self.ticks_alive)

        # === Phase 12: Goal progress update ===
        self.goals.update_progress(concepts_now, reward, self.ticks_alive)
        if self.body.health < 0.15:
            self.goals.abandon_all()
        # Goal selection every 20 ticks
        if self.ticks_alive % 20 == 0 and self.goals.max_depth > 0:
            episodic_targets = []
            top_episodes = self.episodic.retrieve(
                concepts_now, emotional_val, self.ticks_alive, top_k=2)
            for ep in top_episodes:
                if ep.outcome > 0:
                    episodic_targets.append(ep.concepts)
            self.goals.select_goal(
                self.internal.as_vector(), concepts_now,
                episodic_targets=episodic_targets if episodic_targets else None,
            )
        # Goal completion reward
        if self.goals.check_and_clear_completion():
            reward += 0.5

        # === Phase 13: Norm evaluation ===
        norm_modulation = self.norms.evaluate(concepts_now)
        reward += norm_modulation
        self.norms.learn_from_outcome(concepts_now, reward)
        if self.ticks_alive % 100 == 0:
            self.norms.decay()

        # === Phase 13: Abstract naming ===
        if self.naming.abstract_capacity > 0 and self.ticks_alive % 30 == 0:
            emotional_sal = float(np.sum(np.abs(self.internal.as_vector() - 0.5)))
            if self.naming.should_mint_abstract(concepts_now, emotional_sal):
                used_tokens = set(b.token_id for b in self.naming.name_registry.values())
                for tid in range(self.discrete_vocab.vocab_active):
                    if tid not in used_tokens:
                        self.naming.mint_abstract_name(tid, concepts_now, self.ticks_alive)
                        break

        # === DAYDREAM / INSPIRATION ===
        # Curiosity + boredom drives aimless imagination.
        # Low surprise = world is predictable = boring = time to daydream.
        # High curiosity = innate drive to explore concept space.
        boredom = max(0.0, 0.5 - self.self_model.surprise)  # bored when unsurprised
        curiosity = self.internal.curiosity
        daydream_drive = boredom * 0.6 + curiosity * 0.4
        # Daydream probability: ~5-15% per tick when bored+curious
        if self._daydream_rng.random() < daydream_drive * 0.15:
            dream_value, dream_novelty = self.brain.daydream(self._daydream_rng)
            # Novelty reward: "that was a new thought" — regardless of usefulness
            if dream_novelty > 0.3:
                reward += dream_novelty * 0.05 * curiosity
            # Inspiration reward: novel AND valuable daydream
            if dream_novelty > 0.5 and dream_value > 0.1:
                reward += 0.2  # eureka moment

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
            # Phase 7: Grammar listener grounding
            if hasattr(self._heard_tokens, 'roles'):
                for slot_idx in range(min(len(self._heard_tokens.roles), self.grammar.n_slots)):
                    self.grammar.ground_listener_roles(
                        int(self._heard_tokens.roles[slot_idx]), concepts,
                        outcome_good, lr=self.learning_rate * 0.5)
            # Phase 10: Naming grounding — try to ground heard tokens to attended entity
            if self._attended_entity is not None:
                att_type, att_id = self._attended_entity
                for tid in self._heard_tokens.tokens:
                    self.naming.attempt_grounding(
                        int(tid), att_id, att_type,
                        concepts, self.body.position, self.ticks_alive)
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
        meta_cost = self._cfg.brain_metabolic_cost * self.meta_concepts.param_count
        tom_cost = self._cfg.brain_metabolic_cost * self.tom.param_count
        episodic_cost = self.episodic.capacity * 0.00005
        naming_cost = self.naming.capacity * 0.00002
        goal_cost = self.goals.max_depth * 0.0003
        workspace_cost = self.workspace.n_slots * 0.0003
        norm_cost = self.norms.capacity * 0.0001
        branch_cost = max(0, self._think_branch_count - 1) * self._cfg.think_energy_cost * self._think_steps_used
        brain_cost += meta_cost + tom_cost + episodic_cost + naming_cost + goal_cost + workspace_cost + norm_cost + branch_cost
        morph_cost = self.morphology.total_metabolic_cost
        think_cost = self._think_steps_used * self._cfg.think_energy_cost

        # Phase 14: Quadratic isolation stress — loneliness is metabolically expensive
        sn = self.internal.social_need
        isolation_cost = sn * 0.004 + sn ** 2 * 0.006 * self._social_sensitivity_k
        # Grief cost: recent partner loss adds stress
        grief_cost = self._grief_level * 0.003

        # Phase 14: Reputation decay toward neutral
        self.reputation *= 0.9995
        self.teaching_prestige *= 0.999

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
            reward += 1.5  # massive teaching reward — cultural transmission is survival

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
        # Diminishing returns: each child raises the energy bar
        # Prevents single agents from dominating the gene pool
        child_penalty = min(0.3, self.children_count * 0.005)
        effective_threshold = threshold + child_penalty
        return (
            self.body.energy > effective_threshold
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

    def inherit_grammar(self, parent_agent, rng):
        """Inherit grammar role embeddings from parent with small mutation."""
        n = min(self.grammar.n_slots, parent_agent.grammar.n_slots)
        bn = min(self.grammar.bottleneck_size, parent_agent.grammar.bottleneck_size)
        for i in range(n):
            self.grammar.role_embeddings[i, :bn] = parent_agent.grammar.role_embeddings[i, :bn].copy()
            # Small mutation
            if rng.random() < 0.3:
                self.grammar.role_embeddings[i] += rng.normal(0, 0.05, size=self.grammar.bottleneck_size)
                norm = np.linalg.norm(self.grammar.role_embeddings[i])
                if norm > 1e-8:
                    self.grammar.role_embeddings[i] /= norm

    def inherit_meta_concepts(self, parent_agent, rng):
        """Inherit meta-concept encoder weights with mutation."""
        # Copy encoder weights if dimensions match
        if (self.meta_concepts._encoder.W.shape == parent_agent.meta_concepts._encoder.W.shape):
            self.meta_concepts._encoder.W[:] = parent_agent.meta_concepts._encoder.W
            self.meta_concepts._encoder.b[:] = parent_agent.meta_concepts._encoder.b
            mask = rng.random(self.meta_concepts._encoder.W.shape) < 0.1
            self.meta_concepts._encoder.W[mask] += rng.normal(0, 0.05, size=mask.sum())

    def inherit_tom(self, parent_agent, rng):
        """Inherit Theory of Mind network weights with mutation."""
        if (self.tom._hidden_layer.W.shape == parent_agent.tom._hidden_layer.W.shape):
            self.tom._hidden_layer.W[:] = parent_agent.tom._hidden_layer.W
            self.tom._hidden_layer.b[:] = parent_agent.tom._hidden_layer.b
            self.tom._output_layer.W[:] = parent_agent.tom._output_layer.W
            self.tom._output_layer.b[:] = parent_agent.tom._output_layer.b
            mask = rng.random(self.tom._hidden_layer.W.shape) < 0.1
            self.tom._hidden_layer.W[mask] += rng.normal(0, 0.05, size=mask.sum())

    def inherit_episodic(self, parent_agent, rng):
        """Inherit top episodic memories from parent (compressed)."""
        if not parent_agent.episodic.episodes:
            return
        # Inherit top-3 highest-outcome episodes as compressed gist
        sorted_eps = sorted(parent_agent.episodic.episodes,
                            key=lambda e: e.outcome, reverse=True)
        for ep in sorted_eps[:3]:
            if ep.outcome > 0 and len(self.episodic.episodes) < self.episodic.capacity:
                from agents.episodic_memory import Episode
                child_ep = Episode(
                    ep.concepts, ep.position, ep.tick,
                    ep.emotional_valence, [], ep.action_taken, ep.outcome,
                    ep.surprise)
                child_ep.detail_level = 0.3  # inherited memories are vague
                self.episodic.episodes.append(child_ep)

    def inherit_naming(self, parent_agent, rng):
        """Inherit strongest name bindings from parent."""
        sorted_names = sorted(parent_agent.naming.name_registry.values(),
                              key=lambda b: b.strength, reverse=True)
        for binding in sorted_names[:self.naming.capacity]:
            if rng.random() < binding.strength:
                self.naming.assign_name(
                    binding.token_id, binding.entity_type, binding.entity_id,
                    binding.concept_signature, binding.last_seen_pos,
                    binding.last_seen_tick)

    def inherit_goals_template(self, parent_agent, rng):
        """Inherit goal-setting tendencies (no active goals, just parameters already in genome)."""
        pass  # Goal system parameters are gene-controlled, no learned state to inherit

    def inherit_workspace(self, parent_agent, rng):
        """Inherit workspace contents from parent (compressed)."""
        if parent_agent.workspace.n_slots == 0 or self.workspace.n_slots == 0:
            return
        n = min(self.workspace.n_slots, parent_agent.workspace.n_slots)
        bn = min(self.workspace.bottleneck_size, parent_agent.workspace.bottleneck_size)
        for i in range(n):
            self.workspace.slots[i, :bn] = parent_agent.workspace.slots[i, :bn] * 0.5
            if rng.random() < 0.2:
                self.workspace.slots[i] += rng.normal(
                    0, 0.1, size=self.workspace.bottleneck_size).astype(np.float32)

    def inherit_norms(self, parent_agent, rng):
        """Inherit norms from parent with mutation."""
        if parent_agent.norms.capacity == 0 or self.norms.capacity == 0:
            return
        mutation_rate = get_trait(self.genome, "mutation_rate")
        self.norms.inherit_from(parent_agent.norms.norms, rng, mutation_rate)

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
        extra = 9  # max extra_sensor_dim from morphology
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
