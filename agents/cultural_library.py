
import numpy as np
import hashlib


class CulturalEntry:
    __slots__ = ['encoded_rule', 'accuracy', 'contributor_id', 'generation',
                 'tick', 'provenance', 'innovation_score', 'use_count',
                 'derived_from']

    def __init__(self, encoded_rule: np.ndarray, accuracy: float,
                 contributor_id: int, generation: int, tick: int,
                 parent_rule_hash: str | None = None):
        self.encoded_rule = encoded_rule.copy()
        self.accuracy = float(accuracy)
        self.contributor_id = contributor_id
        self.generation = generation
        self.tick = tick
        self.provenance = f"agent#{contributor_id}_gen{generation}"
        self.innovation_score = 0.0
        self.use_count = 0
        self.derived_from = parent_rule_hash

    def rule_hash(self) -> str:
        return hashlib.md5(self.encoded_rule.tobytes()).hexdigest()[:8]


class CulturalLibrary:
    """Per-lineage cumulative rule library with versioning and innovation tracking."""

    def __init__(self, max_rules: int = 24):
        self.max_rules = max_rules
        self.entries: list[CulturalEntry] = []

    def contribute(self, encoded_rule: np.ndarray, accuracy: float,
                   contributor_id: int, generation: int, tick: int,
                   parent_rule_hash: str | None = None) -> bool:
        """Accept a rule if it's better than the worst existing rule."""
        if accuracy < 0.1:
            return False

        entry = CulturalEntry(encoded_rule, accuracy, contributor_id,
                              generation, tick, parent_rule_hash)

        # Compute innovation score
        entry.innovation_score = self._compute_innovation(encoded_rule)

        if len(self.entries) < self.max_rules:
            self.entries.append(entry)
            return True

        # Find worst entry
        worst_idx = 0
        worst_val = float('inf')
        for i, e in enumerate(self.entries):
            val = e.accuracy * (1.0 + 0.05 * e.use_count)
            if val < worst_val:
                worst_val = val
                worst_idx = i

        # Only replace if better
        new_val = accuracy * (1.0 + 0.1 * entry.innovation_score)
        if new_val > worst_val:
            self.entries[worst_idx] = entry
            return True
        return False

    def study(self, rng: np.random.Generator, agent_generation: int) -> CulturalEntry | None:
        """Select a rule to study, weighted by quality and relevance."""
        if not self.entries:
            return None

        weights = np.zeros(len(self.entries))
        for i, e in enumerate(self.entries):
            # Prefer accurate, well-used, and generationally recent rules
            gen_relevance = 1.0 / (1.0 + 0.01 * abs(agent_generation - e.generation))
            use_bonus = 1.0 + 0.1 * min(e.use_count, 10)
            weights[i] = e.accuracy * use_bonus * gen_relevance

        total = weights.sum()
        if total < 1e-8:
            return None

        weights /= total
        idx = rng.choice(len(self.entries), p=weights)
        self.entries[idx].use_count += 1
        return self.entries[idx]

    def get_innovation_score(self, new_rule_encoded: np.ndarray) -> float:
        """How different is this rule from anything in the library?"""
        return self._compute_innovation(new_rule_encoded)

    def _compute_innovation(self, rule_encoded: np.ndarray) -> float:
        """Cosine distance from nearest existing rule."""
        if not self.entries:
            return 1.0

        rule_norm = np.linalg.norm(rule_encoded)
        if rule_norm < 1e-8:
            return 0.0

        min_dist = 2.0
        for e in self.entries:
            e_norm = np.linalg.norm(e.encoded_rule)
            if e_norm < 1e-8:
                continue
            n = min(len(rule_encoded), len(e.encoded_rule))
            cos_sim = float(np.dot(rule_encoded[:n], e.encoded_rule[:n])) / (rule_norm * e_norm)
            dist = 1.0 - cos_sim
            if dist < min_dist:
                min_dist = dist

        return float(np.clip(min_dist, 0.0, 1.0))

    def get_cultural_complexity(self) -> float:
        """Metric for cultural sophistication."""
        if not self.entries:
            return 0.0
        mean_acc = float(np.mean([e.accuracy for e in self.entries]))
        fill_ratio = len(self.entries) / self.max_rules
        return mean_acc * fill_ratio

    @property
    def rule_count(self) -> int:
        return len(self.entries)

    @property
    def mean_accuracy(self) -> float:
        if not self.entries:
            return 0.0
        return float(np.mean([e.accuracy for e in self.entries]))

    @property
    def mean_innovation(self) -> float:
        if not self.entries:
            return 0.0
        return float(np.mean([e.innovation_score for e in self.entries]))
