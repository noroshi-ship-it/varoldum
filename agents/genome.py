
import numpy as np
from config import Config

LOCI = {
    "sensor_range":          (0, 2.0, 8.0),
    "sensor_cone_width":     (1, 0.5, 3.14),
    "body_size":             (2, 0.5, 2.0),
    "max_lifespan_factor":   (3, 0.5, 2.0),
    "mutation_rate":          (4, 0.03, 0.2),
    "learning_rate":          (5, 0.001, 0.1),
    "hunger_sensitivity":     (6, 0.1, 2.0),
    "fear_sensitivity":       (7, 0.1, 2.0),
    "curiosity_sensitivity":  (8, 0.1, 2.0),
    "reproduction_threshold": (9, 0.5, 0.9),
    "stm_capacity":           (10, 4.0, 16.0),
    "ltm_capacity":           (11, 8.0, 64.0),
}

NUM_TRAIT_GENES = 12

ARCH_OFFSET = 12
ARCH_COUNT = 6

MORPH_OFFSET = 18
MORPH_COUNT = 10

CONCEPT_OFFSET = 28
CONCEPT_COUNT = 3

CONCEPT_LOCI = {
    "bottleneck_size":  (CONCEPT_OFFSET + 0, 8.0, 128.0),
    "think_steps":      (CONCEPT_OFFSET + 1, 0.0, 8.0),
    "world_model_lr":   (CONCEPT_OFFSET + 2, 0.001, 0.05),
}

# Abstract thinking genes (Phase 1-4)
ABSTRACT_OFFSET = 31
ABSTRACT_COUNT = 8

ABSTRACT_LOCI = {
    # Phase 1: Symbol system
    "n_symbols":              (ABSTRACT_OFFSET + 0, 4.0, 32.0),
    "n_symbol_slots":         (ABSTRACT_OFFSET + 1, 1.0, 4.0),
    "symbol_lr":              (ABSTRACT_OFFSET + 2, 0.001, 0.05),
    # Phase 2: Recursive thought
    "depth_gate_threshold":   (ABSTRACT_OFFSET + 3, -0.5, 0.5),
    # Phase 3: Discrete language
    "vocab_active":           (ABSTRACT_OFFSET + 4, 2.0, 64.0),
    "utterance_length":       (ABSTRACT_OFFSET + 5, 1.0, 8.0),
    "listen_weight":          (ABSTRACT_OFFSET + 6, 0.0, 1.0),
    # Phase 4: Mortality awareness
    "mortality_sensitivity":  (ABSTRACT_OFFSET + 7, 0.0, 2.0),
}

# Society engine genes (Phase 5)
SOCIETY_OFFSET = ABSTRACT_OFFSET + ABSTRACT_COUNT  # 39
SOCIETY_COUNT = 8

SOCIETY_LOCI = {
    # Phase 5A: Social emotions
    "social_sensitivity":       (SOCIETY_OFFSET + 0, 0.0, 2.0),
    "trust_sensitivity":        (SOCIETY_OFFSET + 1, 0.0, 2.0),
    "social_reward_sensitivity": (SOCIETY_OFFSET + 2, 0.0, 2.0),
    # Phase 5D: Trade
    "trade_willingness":        (SOCIETY_OFFSET + 3, 0.0, 1.0),
    "utility_pref_0":           (SOCIETY_OFFSET + 4, 0.0, 1.0),
    "utility_pref_1":           (SOCIETY_OFFSET + 5, 0.0, 1.0),
    "inventory_capacity":       (SOCIETY_OFFSET + 6, 2.0, 6.0),
    # Nostalgia
    "nostalgia_sensitivity":    (SOCIETY_OFFSET + 7, 0.0, 2.0),
}

# Meta-cognition genes (Phase 6-8)
META_OFFSET = SOCIETY_OFFSET + SOCIETY_COUNT  # 47
META_COUNT = 10

META_LOCI = {
    # Phase 6: Meta-concepts
    "meta_window":          (META_OFFSET + 0, 4.0, 16.0),
    "meta_bottleneck_size": (META_OFFSET + 1, 2.0, 16.0),
    "meta_wm_lr":           (META_OFFSET + 2, 0.001, 0.05),
    # Phase 7: Compositional grammar
    "n_grammar_slots":      (META_OFFSET + 3, 2.0, 10.0),
    "grammar_lr":           (META_OFFSET + 4, 0.001, 0.05),
    "grammar_weight":       (META_OFFSET + 5, 0.0, 1.0),
    # Phase 8: Theory of Mind
    "tom_max_tracked":      (META_OFFSET + 6, 2.0, 6.0),
    "tom_window":           (META_OFFSET + 7, 4.0, 12.0),
    "tom_lr":               (META_OFFSET + 8, 0.001, 0.05),
    "tom_weight":           (META_OFFSET + 9, 0.0, 1.0),
}

# Cognitive genes (Phase 9-12): Episodic memory, Naming, Culture, Goals
COGNITIVE_OFFSET = META_OFFSET + META_COUNT  # 57
COGNITIVE_COUNT = 16

COGNITIVE_LOCI = {
    # Phase 9: Episodic memory
    "episodic_capacity":          (COGNITIVE_OFFSET + 0, 4.0, 32.0),
    "episode_surprise_threshold": (COGNITIVE_OFFSET + 1, 0.1, 0.8),
    "episodic_replay_depth":      (COGNITIVE_OFFSET + 2, 0.0, 3.0),
    "consolidation_rate":         (COGNITIVE_OFFSET + 3, 0.01, 0.2),
    # Phase 10: Naming/Reference
    "name_capacity":              (COGNITIVE_OFFSET + 4, 2.0, 16.0),
    "name_learning_rate":         (COGNITIVE_OFFSET + 5, 0.005, 0.05),
    "referential_weight":         (COGNITIVE_OFFSET + 6, 0.0, 1.0),
    # Phase 11: Cumulative culture
    "cultural_receptivity":       (COGNITIVE_OFFSET + 7, 0.0, 2.0),
    "innovation_drive":           (COGNITIVE_OFFSET + 8, 0.0, 2.0),
    "teaching_investment":        (COGNITIVE_OFFSET + 9, 0.0, 2.0),
    # Phase 12: Persistent goals
    "goal_stack_depth":           (COGNITIVE_OFFSET + 10, 0.0, 4.0),
    "commitment_strength":        (COGNITIVE_OFFSET + 11, 0.0, 2.0),
    "patience":                   (COGNITIVE_OFFSET + 12, 10.0, 200.0),
    "goal_horizon":               (COGNITIVE_OFFSET + 13, 0.0, 1.0),
    "goal_communication_weight":  (COGNITIVE_OFFSET + 14, 0.0, 1.0),
    "episodic_emotion_weight":    (COGNITIVE_OFFSET + 15, 0.0, 2.0),
}

# Emergence infrastructure genes (Phase 13)
EMERGENCE_OFFSET = COGNITIVE_OFFSET + COGNITIVE_COUNT  # 73
EMERGENCE_COUNT = 12

EMERGENCE_LOCI = {
    "workspace_slots":       (EMERGENCE_OFFSET + 0, 0.0, 6.0),
    "workspace_gate":        (EMERGENCE_OFFSET + 1, 0.0, 1.0),
    "think_branch_count":    (EMERGENCE_OFFSET + 2, 1.0, 4.0),
    "norm_capacity":         (EMERGENCE_OFFSET + 3, 0.0, 8.0),
    "norm_sensitivity":      (EMERGENCE_OFFSET + 4, 0.0, 1.0),
    "abstract_naming":       (EMERGENCE_OFFSET + 5, 0.0, 4.0),
    "temporal_encoding":     (EMERGENCE_OFFSET + 6, 0.0, 1.0),
    "composition_depth":     (EMERGENCE_OFFSET + 7, 0.0, 3.0),
    "counterfactual_weight": (EMERGENCE_OFFSET + 8, 0.0, 1.0),
    "norm_inheritance":      (EMERGENCE_OFFSET + 9, 0.0, 1.0),
    # Group identity
    "group_identity_weight": (EMERGENCE_OFFSET + 10, 0.0, 1.0),
    "in_group_cooperation":  (EMERGENCE_OFFSET + 11, 0.0, 2.0),
}

FIXED_GENE_COUNT = NUM_TRAIT_GENES + ARCH_COUNT + MORPH_COUNT + CONCEPT_COUNT + ABSTRACT_COUNT + SOCIETY_COUNT + META_COUNT + COGNITIVE_COUNT + EMERGENCE_COUNT


def random_genome(cfg: Config, max_nn_params: int, rng: np.random.Generator) -> np.ndarray:
    genes = np.zeros(FIXED_GENE_COUNT + max_nn_params)

    for name, (idx, lo, hi) in LOCI.items():
        genes[idx] = rng.uniform(lo, hi)

    genes[ARCH_OFFSET + 0] = rng.uniform(1.0, 2.5)
    genes[ARCH_OFFSET + 1] = rng.uniform(16, 48)
    genes[ARCH_OFFSET + 2] = rng.uniform(12, 32)
    genes[ARCH_OFFSET + 3] = rng.uniform(8, 24)
    genes[ARCH_OFFSET + 4] = rng.uniform(8, 16)
    genes[ARCH_OFFSET + 5] = rng.uniform(16, 64)

    for i in range(MORPH_COUNT):
        genes[MORPH_OFFSET + i] = rng.uniform(0.0, 0.6)

    genes[CONCEPT_OFFSET + 0] = rng.uniform(8.0, 24.0)
    genes[CONCEPT_OFFSET + 1] = rng.uniform(1.0, 3.0)
    genes[CONCEPT_OFFSET + 2] = rng.uniform(0.005, 0.02)

    # Abstract thinking genes
    genes[ABSTRACT_OFFSET + 0] = rng.uniform(6.0, 16.0)    # n_symbols
    genes[ABSTRACT_OFFSET + 1] = rng.uniform(1.0, 3.0)     # n_symbol_slots
    genes[ABSTRACT_OFFSET + 2] = rng.uniform(0.005, 0.02)  # symbol_lr
    genes[ABSTRACT_OFFSET + 3] = rng.uniform(-0.1, 0.1)    # depth_gate_threshold
    genes[ABSTRACT_OFFSET + 4] = rng.uniform(6.0, 20.0)    # vocab_active (higher start)
    genes[ABSTRACT_OFFSET + 5] = rng.uniform(1.0, 3.0)     # utterance_length (higher start)
    genes[ABSTRACT_OFFSET + 6] = rng.uniform(0.2, 0.6)     # listen_weight
    genes[ABSTRACT_OFFSET + 7] = rng.uniform(0.5, 1.5)     # mortality_sensitivity

    # Society genes
    genes[SOCIETY_OFFSET + 0] = rng.uniform(0.5, 1.5)     # social_sensitivity
    genes[SOCIETY_OFFSET + 1] = rng.uniform(0.5, 1.5)     # trust_sensitivity
    genes[SOCIETY_OFFSET + 2] = rng.uniform(0.5, 1.5)     # social_reward_sensitivity
    genes[SOCIETY_OFFSET + 3] = rng.uniform(0.2, 0.6)     # trade_willingness
    genes[SOCIETY_OFFSET + 4] = rng.uniform(0.2, 0.8)     # utility_pref_0
    genes[SOCIETY_OFFSET + 5] = rng.uniform(0.2, 0.8)     # utility_pref_1
    genes[SOCIETY_OFFSET + 6] = rng.uniform(2.0, 4.0)     # inventory_capacity
    genes[SOCIETY_OFFSET + 7] = rng.uniform(0.5, 1.5)     # nostalgia_sensitivity

    # Meta-cognition genes (Phase 6-8)
    genes[META_OFFSET + 0] = rng.uniform(4.0, 8.0)        # meta_window
    genes[META_OFFSET + 1] = rng.uniform(2.0, 8.0)        # meta_bottleneck_size
    genes[META_OFFSET + 2] = rng.uniform(0.005, 0.02)     # meta_wm_lr
    genes[META_OFFSET + 3] = rng.uniform(2.0, 3.0)        # n_grammar_slots
    genes[META_OFFSET + 4] = rng.uniform(0.005, 0.02)     # grammar_lr
    genes[META_OFFSET + 5] = rng.uniform(0.0, 0.3)        # grammar_weight (start low)
    genes[META_OFFSET + 6] = rng.uniform(2.0, 3.0)        # tom_max_tracked
    genes[META_OFFSET + 7] = rng.uniform(4.0, 8.0)        # tom_window
    genes[META_OFFSET + 8] = rng.uniform(0.005, 0.02)     # tom_lr
    genes[META_OFFSET + 9] = rng.uniform(0.0, 0.3)        # tom_weight (start low)

    # Cognitive genes (Phase 9-12)
    genes[COGNITIVE_OFFSET + 0] = rng.uniform(4.0, 12.0)       # episodic_capacity
    genes[COGNITIVE_OFFSET + 1] = rng.uniform(0.2, 0.5)        # episode_surprise_threshold
    genes[COGNITIVE_OFFSET + 2] = rng.uniform(0.0, 1.0)        # episodic_replay_depth
    genes[COGNITIVE_OFFSET + 3] = rng.uniform(0.02, 0.1)       # consolidation_rate
    genes[COGNITIVE_OFFSET + 4] = rng.uniform(2.0, 6.0)        # name_capacity
    genes[COGNITIVE_OFFSET + 5] = rng.uniform(0.01, 0.03)      # name_learning_rate
    genes[COGNITIVE_OFFSET + 6] = rng.uniform(0.0, 0.3)        # referential_weight (start low)
    genes[COGNITIVE_OFFSET + 7] = rng.uniform(0.0, 0.5)        # cultural_receptivity (start low)
    genes[COGNITIVE_OFFSET + 8] = rng.uniform(0.0, 0.5)        # innovation_drive (start low)
    genes[COGNITIVE_OFFSET + 9] = rng.uniform(0.0, 0.5)        # teaching_investment (start low)
    genes[COGNITIVE_OFFSET + 10] = rng.uniform(0.0, 1.0)       # goal_stack_depth (start low)
    genes[COGNITIVE_OFFSET + 11] = rng.uniform(0.0, 0.5)       # commitment_strength (start low)
    genes[COGNITIVE_OFFSET + 12] = rng.uniform(20.0, 80.0)     # patience
    genes[COGNITIVE_OFFSET + 13] = rng.uniform(0.0, 0.3)       # goal_horizon (start short-term)
    genes[COGNITIVE_OFFSET + 14] = rng.uniform(0.0, 0.3)       # goal_communication_weight
    genes[COGNITIVE_OFFSET + 15] = rng.uniform(0.0, 0.5)       # episodic_emotion_weight

    # Emergence infrastructure genes (Phase 13) — start at minimum/low
    genes[EMERGENCE_OFFSET + 0] = rng.uniform(0.0, 1.0)        # workspace_slots (start small)
    genes[EMERGENCE_OFFSET + 1] = rng.uniform(0.0, 0.2)        # workspace_gate (start weak)
    genes[EMERGENCE_OFFSET + 2] = rng.uniform(1.0, 1.5)        # think_branch_count (start ~1)
    genes[EMERGENCE_OFFSET + 3] = rng.uniform(0.0, 1.0)        # norm_capacity (start small)
    genes[EMERGENCE_OFFSET + 4] = rng.uniform(0.0, 0.2)        # norm_sensitivity (start weak)
    genes[EMERGENCE_OFFSET + 5] = rng.uniform(0.0, 0.5)        # abstract_naming (start small)
    genes[EMERGENCE_OFFSET + 6] = rng.uniform(0.0, 0.2)        # temporal_encoding (start weak)
    genes[EMERGENCE_OFFSET + 7] = rng.uniform(0.0, 0.5)        # composition_depth (start low)
    genes[EMERGENCE_OFFSET + 8] = rng.uniform(0.0, 0.2)        # counterfactual_weight (start weak)
    genes[EMERGENCE_OFFSET + 9] = rng.uniform(0.0, 0.3)        # norm_inheritance (start low)
    genes[EMERGENCE_OFFSET + 10] = rng.uniform(0.0, 0.3)       # group_identity_weight (start low)
    genes[EMERGENCE_OFFSET + 11] = rng.uniform(0.0, 0.5)       # in_group_cooperation (start low)

    nn_start = FIXED_GENE_COUNT
    genes[nn_start:] = rng.standard_normal(max_nn_params) * 0.3

    return genes


def get_abstract_genes(genome: np.ndarray) -> np.ndarray:
    return genome[ABSTRACT_OFFSET:ABSTRACT_OFFSET + ABSTRACT_COUNT]


def get_society_genes(genome: np.ndarray) -> np.ndarray:
    return genome[SOCIETY_OFFSET:SOCIETY_OFFSET + SOCIETY_COUNT]


def get_trait(genome: np.ndarray, name: str) -> float:
    if name in LOCI:
        idx, lo, hi = LOCI[name]
    elif name in CONCEPT_LOCI:
        idx, lo, hi = CONCEPT_LOCI[name]
    elif name in ABSTRACT_LOCI:
        idx, lo, hi = ABSTRACT_LOCI[name]
    elif name in SOCIETY_LOCI:
        idx, lo, hi = SOCIETY_LOCI[name]
    elif name in META_LOCI:
        idx, lo, hi = META_LOCI[name]
    elif name in COGNITIVE_LOCI:
        idx, lo, hi = COGNITIVE_LOCI[name]
    elif name in EMERGENCE_LOCI:
        idx, lo, hi = EMERGENCE_LOCI[name]
    else:
        raise KeyError(f"Unknown trait: {name}")
    return float(np.clip(genome[idx], lo, hi))


def get_arch_genes(genome: np.ndarray) -> np.ndarray:
    return genome[ARCH_OFFSET:ARCH_OFFSET + ARCH_COUNT]


def get_morph_genes(genome: np.ndarray) -> np.ndarray:
    return genome[MORPH_OFFSET:MORPH_OFFSET + MORPH_COUNT]


def get_nn_weights(genome: np.ndarray) -> np.ndarray:
    return genome[FIXED_GENE_COUNT:]


def get_meta_genes(genome: np.ndarray) -> np.ndarray:
    return genome[META_OFFSET:META_OFFSET + META_COUNT]


def get_cognitive_genes(genome: np.ndarray) -> np.ndarray:
    return genome[COGNITIVE_OFFSET:COGNITIVE_OFFSET + COGNITIVE_COUNT]


def get_emergence_genes(genome: np.ndarray) -> np.ndarray:
    return genome[EMERGENCE_OFFSET:EMERGENCE_OFFSET + EMERGENCE_COUNT]


def get_concept_genes(genome: np.ndarray) -> np.ndarray:
    return genome[CONCEPT_OFFSET:CONCEPT_OFFSET + CONCEPT_COUNT]


def clamp_genome(genome: np.ndarray):
    for name, (idx, lo, hi) in LOCI.items():
        genome[idx] = np.clip(genome[idx], lo, hi)
    genome[ARCH_OFFSET + 0] = np.clip(genome[ARCH_OFFSET + 0], 1, 4)
    for i in range(1, 5):
        genome[ARCH_OFFSET + i] = np.clip(genome[ARCH_OFFSET + i], 8, 256)
    genome[ARCH_OFFSET + 5] = np.clip(genome[ARCH_OFFSET + 5], 16, 128)
    for i in range(MORPH_COUNT):
        genome[MORPH_OFFSET + i] = np.clip(genome[MORPH_OFFSET + i], 0, 1)
    for name, (idx, lo, hi) in CONCEPT_LOCI.items():
        genome[idx] = np.clip(genome[idx], lo, hi)
    for name, (idx, lo, hi) in ABSTRACT_LOCI.items():
        genome[idx] = np.clip(genome[idx], lo, hi)
    for name, (idx, lo, hi) in SOCIETY_LOCI.items():
        genome[idx] = np.clip(genome[idx], lo, hi)
    for name, (idx, lo, hi) in META_LOCI.items():
        genome[idx] = np.clip(genome[idx], lo, hi)
    for name, (idx, lo, hi) in COGNITIVE_LOCI.items():
        genome[idx] = np.clip(genome[idx], lo, hi)
    for name, (idx, lo, hi) in EMERGENCE_LOCI.items():
        genome[idx] = np.clip(genome[idx], lo, hi)
