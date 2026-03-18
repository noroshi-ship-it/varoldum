
import numpy as np


class Norm:
    __slots__ = ['pattern', 'valence', 'strength', 'fire_count', 'accuracy']

    def __init__(self, pattern: np.ndarray, valence: float, strength: float):
        self.pattern = pattern.copy().astype(np.float32)
        self.valence = float(np.clip(valence, -1.0, 1.0))
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.fire_count = 0
        self.accuracy = 0.5


class NormSystem:
    """Evolvable internal rules: concept_pattern -> valence modulation.

    Norms are INHERITED from parent (with mutation) — a second inheritance
    channel. Norms can also CRYSTALLIZE from repeated experience.
    Zero capacity = no norms = zero cost = backward compatible.
    """

    __slots__ = [
        'capacity', 'bottleneck_size', 'sensitivity', 'inheritance_rate',
        'norms', '_crystallization_buffer', '_buffer_max',
        '_last_reward_modulation',
    ]

    def __init__(self, capacity: int, bottleneck_size: int,
                 sensitivity: float, inheritance_rate: float):
        self.capacity = max(0, int(capacity))
        self.bottleneck_size = bottleneck_size
        self.sensitivity = float(np.clip(sensitivity, 0.0, 1.0))
        self.inheritance_rate = float(np.clip(inheritance_rate, 0.0, 1.0))
        self.norms: list[Norm] = []
        self._crystallization_buffer: list[tuple] = []
        self._buffer_max = 16
        self._last_reward_modulation = 0.0

    def evaluate(self, concepts: np.ndarray, threshold: float = 0.5) -> float:
        """Check concepts against all norms. Return reward modulation."""
        if self.capacity == 0 or not self.norms:
            self._last_reward_modulation = 0.0
            return 0.0

        n = min(len(concepts), self.bottleneck_size)
        c = concepts[:n].astype(np.float32)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-8:
            self._last_reward_modulation = 0.0
            return 0.0
        c_unit = c / c_norm

        modulation = 0.0
        for norm in self.norms:
            plen = min(n, len(norm.pattern))
            p = norm.pattern[:plen]
            p_norm = np.linalg.norm(p)
            if p_norm < 1e-8:
                continue
            similarity = float(np.dot(c_unit[:plen], p / p_norm))
            if similarity > threshold:
                firing = (similarity - threshold) / (1.0 - threshold + 1e-8)
                modulation += norm.valence * norm.strength * firing
                norm.fire_count += 1

        self._last_reward_modulation = modulation * self.sensitivity
        return self._last_reward_modulation

    def learn_from_outcome(self, concepts: np.ndarray, reward: float):
        """Track correlation between norm firings and outcomes.
        Crystallize new norms from consistent patterns."""
        if self.capacity == 0:
            return

        n = min(len(concepts), self.bottleneck_size)
        c = concepts[:n].astype(np.float32)
        c_norm_val = np.linalg.norm(c)

        # Update existing norm accuracy
        if c_norm_val > 1e-8:
            c_unit = c / c_norm_val
            for norm in self.norms:
                p = norm.pattern[:n]
                p_norm = np.linalg.norm(p)
                if p_norm < 1e-8:
                    continue
                sim = float(np.dot(c_unit, p / p_norm))
                if sim > 0.3:
                    agreement = 1.0 if (norm.valence * reward > 0) else 0.0
                    norm.accuracy += 0.05 * (agreement - norm.accuracy)
                    if norm.accuracy > 0.6:
                        norm.strength = min(1.0, norm.strength + 0.01)
                    elif norm.accuracy < 0.3:
                        norm.strength *= 0.95

        # Crystallization buffer
        if c_norm_val > 1e-8 and abs(reward) > 0.1:
            self._crystallization_buffer.append((c.copy(), reward))
            if len(self._crystallization_buffer) > self._buffer_max:
                self._crystallization_buffer.pop(0)

        # Try to crystallize a new norm
        if (len(self._crystallization_buffer) >= 8
                and len(self.norms) < self.capacity):
            self._try_crystallize()

    def _try_crystallize(self):
        """Check if recent experiences form a consistent pattern -> norm."""
        patterns = np.array([p for p, _ in self._crystallization_buffer[-8:]])
        outcomes = np.array([o for _, o in self._crystallization_buffer[-8:]])

        weights = np.abs(outcomes)
        if np.sum(weights) < 1e-8:
            return
        avg_pattern = (patterns.T @ weights) / np.sum(weights)

        avg_outcome = float(np.mean(outcomes))
        outcome_std = float(np.std(outcomes))

        if abs(avg_outcome) > 0.15 and outcome_std < abs(avg_outcome) * 1.5:
            valence = 1.0 if avg_outcome > 0 else -1.0
            new_norm = Norm(avg_pattern, valence, strength=0.3)
            self.norms.append(new_norm)
            self._crystallization_buffer.clear()

    def get_context_vector(self, max_dim: int = 4) -> np.ndarray:
        """Return fixed-size context from active norm signals."""
        result = np.zeros(max_dim, dtype=np.float64)
        if self.capacity == 0 or not self.norms:
            return result

        strongest = max(self.norms, key=lambda n: n.strength)
        result[0] = strongest.valence * strongest.strength
        result[1] = strongest.strength
        active = sum(1 for n in self.norms if n.fire_count > 0) / max(1, len(self.norms))
        result[2] = active
        result[3] = np.clip(self._last_reward_modulation, -1, 1)
        return result

    def decay(self):
        """Remove weak norms."""
        self.norms = [n for n in self.norms if n.strength > 0.05]

    def inherit_from(self, parent_norms: list, rng: np.random.Generator,
                     mutation_rate: float = 0.1):
        """Copy parent norms with mutation."""
        for pn in parent_norms:
            if rng.random() > self.inheritance_rate:
                continue
            if len(self.norms) >= self.capacity:
                break
            new_pattern = pn.pattern.copy()
            if rng.random() < 0.3:
                new_pattern += rng.normal(0, mutation_rate,
                                          size=new_pattern.shape).astype(np.float32)
            new_valence = pn.valence
            if rng.random() < 0.05:
                new_valence *= -1
            new_strength = float(np.clip(pn.strength + rng.normal(0, 0.1), 0.05, 1.0))
            self.norms.append(Norm(new_pattern, new_valence, new_strength))

    @property
    def stats(self) -> dict:
        return {
            "capacity": self.capacity,
            "n_norms": len(self.norms),
            "mean_strength": float(np.mean([n.strength for n in self.norms])) if self.norms else 0.0,
            "mean_accuracy": float(np.mean([n.accuracy for n in self.norms])) if self.norms else 0.0,
            "total_firings": sum(n.fire_count for n in self.norms),
            "last_modulation": float(self._last_reward_modulation),
        }
