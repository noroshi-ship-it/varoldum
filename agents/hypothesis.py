
import numpy as np
from enum import IntEnum


class Feature(IntEnum):
    RESOURCE_HERE = 0
    HAZARD_HERE = 1
    AGENT_NEARBY = 2
    SIGNAL_HERE = 3
    MY_ENERGY = 4
    MY_HEALTH = 5
    MY_HUNGER = 6
    MY_FEAR = 7
    MY_CURIOSITY = 8
    SEASON_PHASE = 9
    LIGHT_LEVEL = 10
    RESOURCE_AHEAD = 11
    HAZARD_AHEAD = 12
    TEMPERATURE_HERE = 13
    MINERAL_HERE = 14
    TOXIN_HERE = 15
    SCENT_HERE = 16
    MY_MINERAL_CARRIED = 17
    MY_SPEED = 18
    TEMP_COMFORT = 19
    NUM_FEATURES = 20


class Comparator(IntEnum):
    GREATER = 0
    LESS = 1


class PredictedOutcome(IntEnum):
    ENERGY_UP = 0
    ENERGY_DOWN = 1
    HEALTH_UP = 2
    HEALTH_DOWN = 3
    FIND_RESOURCE = 4
    FIND_DANGER = 5
    SAFE = 6
    FIND_MINERAL = 7
    TOXIN_DAMAGE = 8
    TEMPERATURE_STRESS = 9
    NUM_OUTCOMES = 10


class Condition:
    __slots__ = ['feature', 'comparator', 'threshold']

    def __init__(self, feature: int, comparator: int, threshold: float):
        self.feature = feature % int(Feature.NUM_FEATURES)
        self.comparator = comparator % 2
        self.threshold = np.clip(threshold, 0.0, 1.0)

    def evaluate(self, features: np.ndarray) -> bool:
        val = features[self.feature] if self.feature < len(features) else 0.0
        if self.comparator == Comparator.GREATER:
            return val > self.threshold
        return val < self.threshold

    def encode(self) -> np.ndarray:
        return np.array([self.feature, self.comparator, self.threshold])

    @staticmethod
    def decode(arr: np.ndarray) -> "Condition":
        return Condition(int(arr[0]), int(arr[1]), float(arr[2]))

    def describe(self) -> str:
        fname = Feature(self.feature).name if self.feature < Feature.NUM_FEATURES else "?"
        comp = ">" if self.comparator == Comparator.GREATER else "<"
        return f"{fname} {comp} {self.threshold:.2f}"


class Hypothesis:
    __slots__ = ['conditions', 'outcome', 'action_bias',
                 'tests', 'successes', 'age', 'active']

    def __init__(self, conditions: list[Condition], outcome: int, action_bias: np.ndarray):
        self.conditions = conditions
        self.outcome = outcome % int(PredictedOutcome.NUM_OUTCOMES)
        self.action_bias = action_bias
        self.tests = 0
        self.successes = 0
        self.age = 0
        self.active = True

    @property
    def accuracy(self) -> float:
        if self.tests < 3:
            return 0.5
        return self.successes / self.tests

    @property
    def confidence(self) -> float:
        return 1.0 - 1.0 / (1.0 + self.tests * 0.1)

    def evaluate(self, features: np.ndarray) -> bool:
        return all(c.evaluate(features) for c in self.conditions)

    def test_prediction(self, actual_outcome: np.ndarray):
        self.tests += 1
        self.age += 1

        predicted = self.outcome
        correct = False

        if predicted == PredictedOutcome.ENERGY_UP:
            correct = actual_outcome[0] > 0.01
        elif predicted == PredictedOutcome.ENERGY_DOWN:
            correct = actual_outcome[0] < -0.01
        elif predicted == PredictedOutcome.HEALTH_UP:
            correct = actual_outcome[1] > 0.001
        elif predicted == PredictedOutcome.HEALTH_DOWN:
            correct = actual_outcome[1] < -0.001
        elif predicted == PredictedOutcome.FIND_RESOURCE:
            correct = actual_outcome[2] > 0.2
        elif predicted == PredictedOutcome.FIND_DANGER:
            correct = actual_outcome[3] > 0.2
        elif predicted == PredictedOutcome.SAFE:
            correct = actual_outcome[3] < 0.1 and actual_outcome[1] >= 0
        elif predicted == PredictedOutcome.FIND_MINERAL:
            correct = len(actual_outcome) > 5 and actual_outcome[5] > 0.1
        elif predicted == PredictedOutcome.TOXIN_DAMAGE:
            correct = len(actual_outcome) > 6 and actual_outcome[6] > 0.1
        elif predicted == PredictedOutcome.TEMPERATURE_STRESS:
            correct = len(actual_outcome) > 7 and actual_outcome[7] > 0.3

        if correct:
            self.successes += 1

    def describe(self) -> str:
        conds = " AND ".join(c.describe() for c in self.conditions)
        outcome_name = PredictedOutcome(self.outcome).name
        return (f"IF {conds} THEN {outcome_name} "
                f"[acc={self.accuracy:.2f}, n={self.tests}, conf={self.confidence:.2f}]")

    def encode(self) -> np.ndarray:
        parts = []
        parts.append(np.array([len(self.conditions)]))
        for i in range(3):
            if i < len(self.conditions):
                parts.append(self.conditions[i].encode())
            else:
                parts.append(np.zeros(3))
        parts.append(np.array([self.outcome]))
        parts.append(self.action_bias)
        return np.concatenate(parts)

    @staticmethod
    def decode(arr: np.ndarray, action_dim: int = 6) -> "Hypothesis":
        n_conds = max(1, min(3, int(arr[0])))
        conditions = []
        idx = 1
        for i in range(n_conds):
            conditions.append(Condition.decode(arr[idx:idx + 3]))
            idx += 3
        idx = 1 + 9
        outcome = int(arr[idx]) if idx < len(arr) else 0
        idx += 1
        action_bias = arr[idx:idx + action_dim] if idx + action_dim <= len(arr) else np.zeros(action_dim)
        return Hypothesis(conditions, outcome, action_bias)


HYPOTHESIS_ENCODING_SIZE = 1 + 9 + 1 + 6


class HypothesisSystem:

    def __init__(self, max_hypotheses: int = 8, action_dim: int = 6):
        self.max_hypotheses = max_hypotheses
        self.action_dim = action_dim
        self.hypotheses: list[Hypothesis] = []
        self._feature_vec = np.zeros(int(Feature.NUM_FEATURES))

    def init_random(self, rng: np.random.Generator):
        for _ in range(self.max_hypotheses):
            self.hypotheses.append(self._random_hypothesis(rng))

    def _random_hypothesis(self, rng: np.random.Generator) -> Hypothesis:
        n_conds = rng.integers(1, 4)
        conditions = []
        for _ in range(n_conds):
            conditions.append(Condition(
                feature=int(rng.integers(0, int(Feature.NUM_FEATURES))),
                comparator=int(rng.integers(0, 2)),
                threshold=float(rng.uniform(0.1, 0.9)),
            ))
        outcome = int(rng.integers(0, int(PredictedOutcome.NUM_OUTCOMES)))
        action_bias = rng.standard_normal(self.action_dim) * 0.3
        return Hypothesis(conditions, outcome, action_bias)

    def build_feature_vector(
        self,
        sensor_vec: np.ndarray,
        energy: float,
        health: float,
        hunger: float,
        fear: float,
        curiosity: float,
        season: float,
        light: float,
        mineral_carried: float = 0.0,
        speed: float = 0.0,
        temp_comfort: float = 0.5,
    ) -> np.ndarray:
        f = self._feature_vec
        if len(sensor_vec) >= 8:
            f[Feature.RESOURCE_HERE] = sensor_vec[0]
            f[Feature.HAZARD_HERE] = sensor_vec[1]
            f[Feature.AGENT_NEARBY] = sensor_vec[2]
            f[Feature.SIGNAL_HERE] = sensor_vec[3]
            f[Feature.TEMPERATURE_HERE] = sensor_vec[4]
            f[Feature.MINERAL_HERE] = sensor_vec[5]
            f[Feature.TOXIN_HERE] = sensor_vec[6]
            f[Feature.SCENT_HERE] = sensor_vec[7]
        if len(sensor_vec) >= 16:
            f[Feature.RESOURCE_AHEAD] = sensor_vec[8]
            f[Feature.HAZARD_AHEAD] = sensor_vec[9]
        f[Feature.MY_ENERGY] = energy
        f[Feature.MY_HEALTH] = health
        f[Feature.MY_HUNGER] = hunger
        f[Feature.MY_FEAR] = fear
        f[Feature.MY_CURIOSITY] = curiosity
        f[Feature.SEASON_PHASE] = season
        f[Feature.LIGHT_LEVEL] = light
        f[Feature.MY_MINERAL_CARRIED] = mineral_carried
        f[Feature.MY_SPEED] = speed
        f[Feature.TEMP_COMFORT] = temp_comfort
        return f

    def get_action_bias(self, features: np.ndarray) -> np.ndarray:
        total_bias = np.zeros(self.action_dim)
        for hyp in self.hypotheses:
            if not hyp.active:
                continue
            if hyp.evaluate(features):
                weight = hyp.accuracy * hyp.confidence
                total_bias += hyp.action_bias * weight
        norm = np.linalg.norm(total_bias)
        if norm > 2.0:
            total_bias *= 2.0 / norm
        return total_bias

    def test_hypotheses(self, features: np.ndarray, actual_outcome: np.ndarray):
        for hyp in self.hypotheses:
            if hyp.active and hyp.evaluate(features):
                hyp.test_prediction(actual_outcome)

    def evolve_hypotheses(self, rng: np.random.Generator):
        if len(self.hypotheses) < 2:
            return

        scored = [(h, h.accuracy * h.confidence) for h in self.hypotheses]
        scored.sort(key=lambda x: x[1], reverse=True)

        n_replace = max(1, len(scored) // 4)
        top = [s[0] for s in scored[:n_replace]]
        bottom_indices = list(range(len(scored) - n_replace, len(scored)))

        for i, idx in enumerate(bottom_indices):
            parent = top[i % len(top)]
            child = self._mutate_hypothesis(parent, rng)
            self.hypotheses[idx] = child

    def _mutate_hypothesis(self, parent: Hypothesis, rng: np.random.Generator) -> Hypothesis:
        new_conditions = []
        for c in parent.conditions:
            if rng.random() < 0.3:
                new_conditions.append(Condition(
                    feature=c.feature if rng.random() > 0.2 else int(rng.integers(0, int(Feature.NUM_FEATURES))),
                    comparator=c.comparator if rng.random() > 0.3 else int(rng.integers(0, 2)),
                    threshold=float(np.clip(c.threshold + rng.normal(0, 0.15), 0, 1)),
                ))
            else:
                new_conditions.append(Condition(c.feature, c.comparator, c.threshold))

        if rng.random() < 0.1 and len(new_conditions) < 3:
            new_conditions.append(Condition(
                int(rng.integers(0, int(Feature.NUM_FEATURES))),
                int(rng.integers(0, 2)),
                float(rng.uniform(0.1, 0.9)),
            ))
        elif rng.random() < 0.1 and len(new_conditions) > 1:
            new_conditions.pop(rng.integers(0, len(new_conditions)))

        outcome = parent.outcome
        if rng.random() < 0.15:
            outcome = int(rng.integers(0, int(PredictedOutcome.NUM_OUTCOMES)))

        action_bias = parent.action_bias.copy()
        mask = rng.random(len(action_bias)) < 0.3
        action_bias[mask] += rng.normal(0, 0.2, size=mask.sum())

        h = Hypothesis(new_conditions, outcome, action_bias)
        return h

    def get_active_rules_description(self) -> list[str]:
        descriptions = []
        for hyp in self.hypotheses:
            if hyp.tests >= 5 and hyp.accuracy > 0.6:
                descriptions.append(hyp.describe())
        descriptions.sort(key=lambda x: x, reverse=True)
        return descriptions

    def get_best_hypotheses(self, min_tests: int = 10, min_accuracy: float = 0.6) -> list[Hypothesis]:
        return [
            h for h in self.hypotheses
            if h.tests >= min_tests and h.accuracy >= min_accuracy
        ]

    def encode_all(self) -> np.ndarray:
        parts = []
        for h in self.hypotheses:
            parts.append(h.encode())
        for _ in range(self.max_hypotheses - len(self.hypotheses)):
            parts.append(np.zeros(HYPOTHESIS_ENCODING_SIZE))
        return np.concatenate(parts)

    def decode_all(self, arr: np.ndarray):
        self.hypotheses = []
        for i in range(self.max_hypotheses):
            start = i * HYPOTHESIS_ENCODING_SIZE
            end = start + HYPOTHESIS_ENCODING_SIZE
            if end <= len(arr):
                self.hypotheses.append(
                    Hypothesis.decode(arr[start:end], self.action_dim)
                )

    @property
    def stats(self) -> dict:
        tested = [h for h in self.hypotheses if h.tests >= 5]
        if not tested:
            return {"n_rules": 0, "mean_accuracy": 0, "best_accuracy": 0, "total_tests": 0}
        return {
            "n_rules": len(tested),
            "mean_accuracy": float(np.mean([h.accuracy for h in tested])),
            "best_accuracy": float(max(h.accuracy for h in tested)),
            "total_tests": sum(h.tests for h in tested),
        }
