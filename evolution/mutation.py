
import numpy as np
from agents.genome import (
    LOCI, NUM_TRAIT_GENES, ARCH_OFFSET, ARCH_COUNT,
    MORPH_OFFSET, MORPH_COUNT, ABSTRACT_OFFSET, ABSTRACT_COUNT,
    SOCIETY_OFFSET, SOCIETY_COUNT,
    FIXED_GENE_COUNT, clamp_genome
)


def mutate(genome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    child = genome.copy()

    mr_idx, mr_lo, mr_hi = LOCI["mutation_rate"]
    mutation_rate = float(np.clip(genome[mr_idx], mr_lo, mr_hi))

    for i in range(NUM_TRAIT_GENES):
        if rng.random() < 0.5:
            child[i] += rng.normal(0, mutation_rate)

    for i in range(ARCH_COUNT):
        if rng.random() < 0.25:  # more frequent arch mutations
            idx = ARCH_OFFSET + i
            if i == 0:
                child[idx] += rng.choice([-1, 0, 0, 1])
            else:
                child[idx] += rng.normal(0, mutation_rate * 16)  # bigger jumps in layer sizes

    for i in range(MORPH_COUNT):
        if rng.random() < 0.3:
            idx = MORPH_OFFSET + i
            child[idx] += rng.normal(0, mutation_rate * 2)

    # Abstract thinking genes
    for i in range(ABSTRACT_COUNT):
        if rng.random() < 0.3:
            idx = ABSTRACT_OFFSET + i
            child[idx] += rng.normal(0, mutation_rate * 2)

    # Society genes
    for i in range(SOCIETY_COUNT):
        if rng.random() < 0.3:
            idx = SOCIETY_OFFSET + i
            child[idx] += rng.normal(0, mutation_rate * 2)

    nn_start = FIXED_GENE_COUNT
    nn_weights = child[nn_start:]
    mask = rng.random(len(nn_weights)) < mutation_rate
    nn_weights[mask] += rng.normal(0, mutation_rate * 0.5, size=mask.sum())

    if rng.random() < 0.01:
        idx = rng.integers(nn_start, len(child))
        child[idx] = rng.normal(0, 0.5)

    clamp_genome(child)
    return child


def crossover(parent_a: np.ndarray, parent_b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    length = min(len(parent_a), len(parent_b))

    p1, p2 = sorted(rng.integers(0, FIXED_GENE_COUNT, size=2))
    child = parent_a[:length].copy()
    child[p1:p2] = parent_b[p1:p2]

    nn_len = length - FIXED_GENE_COUNT
    if nn_len > 10:
        p3, p4 = sorted(rng.integers(0, nn_len, size=2))
        child[FIXED_GENE_COUNT + p3:FIXED_GENE_COUNT + p4] = \
            parent_b[FIXED_GENE_COUNT + p3:FIXED_GENE_COUNT + p4]

    clamp_genome(child)
    return child
