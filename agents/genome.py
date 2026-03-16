
import numpy as np
from config import Config

LOCI = {
    "sensor_range":          (0, 2.0, 8.0),
    "sensor_cone_width":     (1, 0.5, 3.14),
    "body_size":             (2, 0.5, 2.0),
    "max_lifespan_factor":   (3, 0.5, 2.0),
    "mutation_rate":          (4, 0.01, 0.2),
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
    "bottleneck_size":  (CONCEPT_OFFSET + 0, 8.0, 64.0),
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
    "vocab_active":           (ABSTRACT_OFFSET + 4, 2.0, 16.0),
    "utterance_length":       (ABSTRACT_OFFSET + 5, 1.0, 3.0),
    "listen_weight":          (ABSTRACT_OFFSET + 6, 0.0, 1.0),
    # Phase 4: Mortality awareness
    "mortality_sensitivity":  (ABSTRACT_OFFSET + 7, 0.0, 2.0),
}

FIXED_GENE_COUNT = NUM_TRAIT_GENES + ARCH_COUNT + MORPH_COUNT + CONCEPT_COUNT + ABSTRACT_COUNT


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
    genes[ABSTRACT_OFFSET + 4] = rng.uniform(4.0, 10.0)    # vocab_active
    genes[ABSTRACT_OFFSET + 5] = rng.uniform(1.0, 2.0)     # utterance_length
    genes[ABSTRACT_OFFSET + 6] = rng.uniform(0.2, 0.6)     # listen_weight
    genes[ABSTRACT_OFFSET + 7] = rng.uniform(0.5, 1.5)     # mortality_sensitivity

    nn_start = FIXED_GENE_COUNT
    genes[nn_start:] = rng.standard_normal(max_nn_params) * 0.3

    return genes


def get_abstract_genes(genome: np.ndarray) -> np.ndarray:
    return genome[ABSTRACT_OFFSET:ABSTRACT_OFFSET + ABSTRACT_COUNT]


def get_trait(genome: np.ndarray, name: str) -> float:
    if name in LOCI:
        idx, lo, hi = LOCI[name]
    elif name in CONCEPT_LOCI:
        idx, lo, hi = CONCEPT_LOCI[name]
    elif name in ABSTRACT_LOCI:
        idx, lo, hi = ABSTRACT_LOCI[name]
    else:
        raise KeyError(f"Unknown trait: {name}")
    return float(np.clip(genome[idx], lo, hi))


def get_arch_genes(genome: np.ndarray) -> np.ndarray:
    return genome[ARCH_OFFSET:ARCH_OFFSET + ARCH_COUNT]


def get_morph_genes(genome: np.ndarray) -> np.ndarray:
    return genome[MORPH_OFFSET:MORPH_OFFSET + MORPH_COUNT]


def get_nn_weights(genome: np.ndarray) -> np.ndarray:
    return genome[FIXED_GENE_COUNT:]


def get_concept_genes(genome: np.ndarray) -> np.ndarray:
    return genome[CONCEPT_OFFSET:CONCEPT_OFFSET + CONCEPT_COUNT]


def clamp_genome(genome: np.ndarray):
    for name, (idx, lo, hi) in LOCI.items():
        genome[idx] = np.clip(genome[idx], lo, hi)
    genome[ARCH_OFFSET + 0] = np.clip(genome[ARCH_OFFSET + 0], 1, 4)
    for i in range(1, 5):
        genome[ARCH_OFFSET + i] = np.clip(genome[ARCH_OFFSET + i], 8, 128)
    genome[ARCH_OFFSET + 5] = np.clip(genome[ARCH_OFFSET + 5], 16, 128)
    for i in range(MORPH_COUNT):
        genome[MORPH_OFFSET + i] = np.clip(genome[MORPH_OFFSET + i], 0, 1)
    for name, (idx, lo, hi) in CONCEPT_LOCI.items():
        genome[idx] = np.clip(genome[idx], lo, hi)
    for name, (idx, lo, hi) in ABSTRACT_LOCI.items():
        genome[idx] = np.clip(genome[idx], lo, hi)
