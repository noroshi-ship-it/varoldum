
import numpy as np


NUM_BODY_FEATURES = 6


class ConceptCondition:
    __slots__ = ['feature_idx', 'comparator', 'threshold', 'is_concept']

    def __init__(self, feature_idx: int, comparator: int, threshold: float,
                 is_concept: bool = True):
        self.feature_idx = feature_idx
        self.comparator = comparator % 2
        self.threshold = np.clip(threshold, -1.0, 1.0)
        self.is_concept = is_concept

    def evaluate(self, concepts: np.ndarray, body_features: np.ndarray) -> bool:
        if self.is_concept:
            if self.feature_idx >= len(concepts):
                return False
            val = concepts[self.feature_idx]
        else:
            if self.feature_idx >= len(body_features):
                return False
            val = body_features[self.feature_idx]

        if self.comparator == 0:
            return val > self.threshold
        return val < self.threshold

    def describe(self, bottleneck_size: int) -> str:
        if self.is_concept:
            name = f"C{self.feature_idx}"
        else:
            names = ["energy", "health", "hunger", "fear", "curiosity", "temp_comfort"]
            name = names[self.feature_idx] if self.feature_idx < len(names) else f"body_{self.feature_idx}"
        comp = ">" if self.comparator == 0 else "<"
        return f"{name}{comp}{self.threshold:.2f}"


class ConceptHypothesis:
    __slots__ = ['conditions', 'outcome', 'action_bias',
                 'tests', 'successes', 'age']

    ENERGY_UP = 0
    ENERGY_DOWN = 1
    SAFE = 2
    DANGER = 3
    NUM_OUTCOMES = 4

    OUTCOME_NAMES = ["ENERGY_UP", "ENERGY_DOWN", "SAFE", "DANGER"]

    def __init__(self, conditions, outcome, action_bias):
        self.conditions = conditions
        self.outcome = outcome % self.NUM_OUTCOMES
        self.action_bias = action_bias
        self.tests = 0
        self.successes = 0
        self.age = 0

    @property
    def accuracy(self):
        if self.tests < 3:
            return 0.5
        return self.successes / self.tests

    @property
    def confidence(self):
        return 1.0 - 1.0 / (1.0 + self.tests * 0.1)

    def evaluate(self, concepts, body_features):
        return all(c.evaluate(concepts, body_features) for c in self.conditions)

    def test(self, energy_delta, damage):
        self.tests += 1
        self.age += 1
        correct = False
        if self.outcome == self.ENERGY_UP:
            correct = energy_delta > 0.01
        elif self.outcome == self.ENERGY_DOWN:
            correct = energy_delta < -0.01
        elif self.outcome == self.SAFE:
            correct = damage < 0.001
        elif self.outcome == self.DANGER:
            correct = damage > 0.01
        if correct:
            self.successes += 1

    def describe(self, bottleneck_size):
        conds = " AND ".join(c.describe(bottleneck_size) for c in self.conditions)
        outcome = self.OUTCOME_NAMES[self.outcome]
        return f"IF {conds} THEN {outcome} [acc={self.accuracy:.2f}, n={self.tests}]"


class ConceptHypothesisSystem:

    def __init__(self, bottleneck_size: int, max_hyp: int = 8, action_dim: int = 8):
        self.bottleneck_size = bottleneck_size
        self.max_hyp = max_hyp
        self.action_dim = action_dim
        self.hypotheses: list[ConceptHypothesis] = []

    def init_random(self, rng):
        for _ in range(self.max_hyp):
            self.hypotheses.append(self._random_hyp(rng))

    def _random_hyp(self, rng):
        n_conds = rng.integers(1, 4)
        conditions = []
        for _ in range(n_conds):
            is_concept = rng.random() < 0.7
            if is_concept:
                idx = int(rng.integers(0, max(1, self.bottleneck_size)))
            else:
                idx = int(rng.integers(0, NUM_BODY_FEATURES))
            conditions.append(ConceptCondition(
                feature_idx=idx,
                comparator=int(rng.integers(0, 2)),
                threshold=float(rng.uniform(-0.8, 0.8)),
                is_concept=is_concept,
            ))
        outcome = int(rng.integers(0, ConceptHypothesis.NUM_OUTCOMES))
        action_bias = rng.standard_normal(self.action_dim) * 0.3
        return ConceptHypothesis(conditions, outcome, action_bias)

    def get_action_bias(self, concepts, body_features):
        total = np.zeros(self.action_dim)
        for hyp in self.hypotheses:
            if hyp.evaluate(concepts, body_features):
                weight = hyp.accuracy * hyp.confidence
                total += hyp.action_bias * weight
        norm = np.linalg.norm(total)
        if norm > 2.0:
            total *= 2.0 / norm
        return total

    def test_all(self, concepts, body_features, energy_delta, damage):
        for hyp in self.hypotheses:
            if hyp.evaluate(concepts, body_features):
                hyp.test(energy_delta, damage)

    def evolve(self, rng):
        if len(self.hypotheses) < 2:
            return
        scored = [(h, h.accuracy * h.confidence) for h in self.hypotheses]
        scored.sort(key=lambda x: x[1], reverse=True)
        n_replace = max(1, len(scored) // 4)
        top = [s[0] for s in scored[:n_replace]]
        for i in range(len(scored) - n_replace, len(scored)):
            parent = top[i % len(top)]
            self.hypotheses[i] = self._mutate(parent, rng)

    def _mutate(self, parent, rng):
        new_conds = []
        for c in parent.conditions:
            if rng.random() < 0.3:
                is_concept = c.is_concept if rng.random() > 0.2 else (rng.random() < 0.7)
                if is_concept:
                    idx = c.feature_idx if rng.random() > 0.2 else int(rng.integers(0, max(1, self.bottleneck_size)))
                else:
                    idx = c.feature_idx if rng.random() > 0.2 else int(rng.integers(0, NUM_BODY_FEATURES))
                new_conds.append(ConceptCondition(
                    idx,
                    c.comparator if rng.random() > 0.3 else int(rng.integers(0, 2)),
                    float(np.clip(c.threshold + rng.normal(0, 0.15), -1, 1)),
                    is_concept,
                ))
            else:
                new_conds.append(ConceptCondition(
                    c.feature_idx, c.comparator, c.threshold, c.is_concept))

        if rng.random() < 0.1 and len(new_conds) < 3:
            is_concept = rng.random() < 0.7
            idx = int(rng.integers(0, max(1, self.bottleneck_size) if is_concept else NUM_BODY_FEATURES))
            new_conds.append(ConceptCondition(idx, int(rng.integers(0, 2)),
                                              float(rng.uniform(-0.8, 0.8)), is_concept))
        elif rng.random() < 0.1 and len(new_conds) > 1:
            new_conds.pop(int(rng.integers(0, len(new_conds))))

        outcome = parent.outcome if rng.random() > 0.15 else int(rng.integers(0, ConceptHypothesis.NUM_OUTCOMES))
        action_bias = parent.action_bias.copy()
        mask = rng.random(len(action_bias)) < 0.3
        action_bias[mask] += rng.normal(0, 0.2, size=mask.sum())

        return ConceptHypothesis(new_conds, outcome, action_bias)

    def get_best(self, min_tests=10, min_acc=0.6):
        return [h for h in self.hypotheses if h.tests >= min_tests and h.accuracy >= min_acc]

    def describe_best(self):
        best = self.get_best(min_tests=5, min_acc=0.55)
        best.sort(key=lambda h: h.accuracy * h.confidence, reverse=True)
        return [h.describe(self.bottleneck_size) for h in best[:5]]

    @property
    def stats(self):
        tested = [h for h in self.hypotheses if h.tests >= 5]
        if not tested:
            return {"n_rules": 0, "mean_accuracy": 0, "best_accuracy": 0}
        return {
            "n_rules": len(tested),
            "mean_accuracy": float(np.mean([h.accuracy for h in tested])),
            "best_accuracy": float(max(h.accuracy for h in tested)),
        }
