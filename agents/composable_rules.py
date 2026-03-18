
import numpy as np
from agents.hypothesis import Feature, Comparator, PredictedOutcome, Hypothesis, Condition


class TemporalBuffer:

    def __init__(self, window_size: int = 20):
        self.window_size = window_size
        self.history: list[np.ndarray] = []

    def record(self, features: np.ndarray):
        self.history.append(features.copy())
        if len(self.history) > self.window_size:
            self.history.pop(0)

    def was_true_n_ago(self, feature: int, comparator: int,
                       threshold: float, n_ticks: int) -> bool:
        if n_ticks >= len(self.history):
            return False
        idx = len(self.history) - 1 - n_ticks
        val = self.history[idx][feature] if feature < len(self.history[idx]) else 0
        if comparator == 0:
            return val > threshold
        return val < threshold

    def count_true_recent(self, feature: int, comparator: int,
                          threshold: float, window: int = 10) -> int:
        count = 0
        start = max(0, len(self.history) - window)
        for i in range(start, len(self.history)):
            val = self.history[i][feature] if feature < len(self.history[i]) else 0
            if comparator == 0:
                is_true = val > threshold
            else:
                is_true = val < threshold
            if is_true:
                count += 1
        return count

    def get_trend(self, feature: int, window: int = 5) -> float:
        if len(self.history) < 2:
            return 0.0
        start = max(0, len(self.history) - window)
        values = []
        for i in range(start, len(self.history)):
            if feature < len(self.history[i]):
                values.append(self.history[i][feature])
        if len(values) < 2:
            return 0.0
        return float(values[-1] - values[0])


class ComposableRule:

    def __init__(self):
        self.conditions: list[Condition] = []
        self.temporal_condition: dict | None = None
        self.count_condition: dict | None = None
        self.trend_condition: dict | None = None
        self.chain_rule_idx: int = -1
        self.outcome: int = 0
        self.action_bias: np.ndarray = np.zeros(8)
        self.tests: int = 0
        self.successes: int = 0
        self.age: int = 0

    @property
    def accuracy(self) -> float:
        if self.tests < 3:
            return 0.5
        return self.successes / self.tests

    @property
    def confidence(self) -> float:
        return 1.0 - 1.0 / (1.0 + self.tests * 0.1)

    @property
    def complexity(self) -> int:
        c = len(self.conditions)
        if self.temporal_condition:
            c += 1
        if self.count_condition:
            c += 1
        if self.trend_condition:
            c += 1
        if self.chain_rule_idx >= 0:
            c += 1
        return c

    def evaluate(self, features: np.ndarray, temporal: TemporalBuffer,
                 other_rules: list = None) -> bool:
        for cond in self.conditions:
            if not cond.evaluate(features):
                return False

        if self.temporal_condition:
            tc = self.temporal_condition
            if not temporal.was_true_n_ago(tc["feature"], tc["comp"],
                                           tc["thresh"], tc["n_ago"]):
                return False

        if self.count_condition:
            cc = self.count_condition
            count = temporal.count_true_recent(cc["feature"], cc["comp"],
                                               cc["thresh"], cc.get("window", 10))
            if count < cc["min_count"]:
                return False

        if self.trend_condition:
            trc = self.trend_condition
            trend = temporal.get_trend(trc["feature"])
            if trc["direction"] > 0 and trend < trc["threshold"]:
                return False
            if trc["direction"] < 0 and trend > -trc["threshold"]:
                return False

        if self.chain_rule_idx >= 0 and other_rules:
            if self.chain_rule_idx < len(other_rules):
                prereq = other_rules[self.chain_rule_idx]
                if not prereq.evaluate(features, temporal):
                    return False

        return True

    def describe(self) -> str:
        parts = []
        for c in self.conditions:
            parts.append(c.describe())
        if self.temporal_condition:
            tc = self.temporal_condition
            fname = Feature(tc["feature"]).name if tc["feature"] < Feature.NUM_FEATURES else "?"
            comp = ">" if tc["comp"] == 0 else "<"
            parts.append(f"{fname} {comp} {tc['thresh']:.2f} [{tc['n_ago']}t ago]")
        if self.count_condition:
            cc = self.count_condition
            fname = Feature(cc["feature"]).name if cc["feature"] < Feature.NUM_FEATURES else "?"
            parts.append(f"COUNT({fname})>={cc['min_count']} in {cc.get('window',10)}t")
        if self.trend_condition:
            trc = self.trend_condition
            fname = Feature(trc["feature"]).name if trc["feature"] < Feature.NUM_FEATURES else "?"
            direction = "rising" if trc["direction"] > 0 else "falling"
            parts.append(f"TREND({fname})={direction}")
        if self.chain_rule_idx >= 0:
            parts.append(f"REQUIRES rule#{self.chain_rule_idx}")

        outcome_name = PredictedOutcome(self.outcome).name if self.outcome < PredictedOutcome.NUM_OUTCOMES else "?"
        cond_str = " AND ".join(parts)
        return (f"IF {cond_str} THEN {outcome_name} "
                f"[acc={self.accuracy:.2f}, n={self.tests}, "
                f"complexity={self.complexity}]")


class ComposableRuleSystem:

    def __init__(self, max_rules: int = 24, action_dim: int = 8,
                 max_complexity: int = 4):
        self.max_rules = max_rules
        self.action_dim = action_dim
        self.max_complexity = max_complexity
        self.rules: list[ComposableRule] = []
        self.temporal = TemporalBuffer(window_size=100)
        self._generation_count = 0

    def init_random(self, rng: np.random.Generator):
        n_feat = int(Feature.NUM_FEATURES)
        n_comp = int(Comparator.NUM_COMPARATORS)
        n_out = int(PredictedOutcome.NUM_OUTCOMES)
        for _ in range(self.max_rules):
            rule = ComposableRule()
            rule.conditions = [Condition(
                int(rng.integers(0, n_feat)),
                int(rng.integers(0, n_comp)),
                float(rng.uniform(0.1, 0.9)),
            )]
            rule.outcome = int(rng.integers(0, n_out))
            rule.action_bias = rng.standard_normal(self.action_dim) * 0.2
            self.rules.append(rule)

    def record_and_evaluate(self, features: np.ndarray) -> np.ndarray:
        self.temporal.record(features)
        total_bias = np.zeros(self.action_dim)

        for rule in self.rules:
            if rule.evaluate(features, self.temporal, self.rules):
                weight = rule.accuracy * rule.confidence
                n = min(len(rule.action_bias), self.action_dim)
                total_bias[:n] += rule.action_bias[:n] * weight

        norm = np.linalg.norm(total_bias)
        if norm > 2.0:
            total_bias *= 2.0 / norm
        return total_bias

    def test_rules(self, features: np.ndarray, outcome: np.ndarray):
        for rule in self.rules:
            if rule.evaluate(features, self.temporal, self.rules):
                rule.tests += 1
                rule.age += 1
                predicted = rule.outcome
                correct = False
                if predicted < len(outcome):
                    if predicted == 0:
                        correct = outcome[0] > 0.01
                    elif predicted == 1:
                        correct = outcome[0] < -0.01
                    elif predicted == 2:
                        correct = outcome[1] > 0.001
                    elif predicted == 3:
                        correct = outcome[1] < -0.001
                    elif predicted == 4:
                        correct = len(outcome) > 2 and outcome[2] > 0.2
                    elif predicted == 5:
                        correct = len(outcome) > 3 and outcome[3] > 0.2
                    elif predicted == 6:
                        correct = len(outcome) > 3 and outcome[3] < 0.1
                    else:
                        correct = False
                if correct:
                    rule.successes += 1

    def evolve(self, rng: np.random.Generator):
        self._generation_count += 1
        if len(self.rules) < 2:
            return

        scored = [(r, r.accuracy * r.confidence + r.complexity * 0.05)
                  for r in self.rules]
        scored.sort(key=lambda x: x[1], reverse=True)

        n_replace = max(1, len(scored) // 3)
        top = [s[0] for s in scored[:max(1, len(scored) // 3)]]

        for i in range(len(scored) - n_replace, len(scored)):
            parent = top[i % len(top)]
            child = self._mutate(parent, rng)
            self.rules[i] = child

    def _mutate(self, parent: ComposableRule, rng: np.random.Generator) -> ComposableRule:
        n_feat = int(Feature.NUM_FEATURES)
        n_comp = int(Comparator.NUM_COMPARATORS)
        n_out = int(PredictedOutcome.NUM_OUTCOMES)
        child = ComposableRule()

        child.conditions = []
        for c in parent.conditions:
            if rng.random() < 0.3:
                child.conditions.append(Condition(
                    c.feature if rng.random() > 0.2 else int(rng.integers(0, n_feat)),
                    c.comparator if rng.random() > 0.3 else int(rng.integers(0, n_comp)),
                    float(np.clip(c.threshold + rng.normal(0, 0.15), 0, 1)),
                ))
            else:
                child.conditions.append(Condition(c.feature, c.comparator, c.threshold))

        if rng.random() < 0.1 and len(child.conditions) < self.max_complexity:
            child.conditions.append(Condition(
                int(rng.integers(0, n_feat)), int(rng.integers(0, n_comp)),
                float(rng.uniform(0.1, 0.9)),
            ))

        if parent.temporal_condition and rng.random() > 0.2:
            tc = parent.temporal_condition.copy()
            if rng.random() < 0.3:
                tc["n_ago"] = max(1, tc["n_ago"] + int(rng.choice([-2, -1, 0, 1, 2])))
                tc["thresh"] = float(np.clip(tc["thresh"] + rng.normal(0, 0.1), 0, 1))
            child.temporal_condition = tc
        elif rng.random() < 0.08:
            child.temporal_condition = {
                "feature": int(rng.integers(0, n_feat)),
                "comp": int(rng.integers(0, n_comp)),
                "thresh": float(rng.uniform(0.1, 0.9)),
                "n_ago": int(rng.integers(1, 15)),
            }

        if parent.count_condition and rng.random() > 0.2:
            cc = parent.count_condition.copy()
            if rng.random() < 0.3:
                cc["min_count"] = max(1, cc["min_count"] + int(rng.choice([-1, 0, 1])))
            child.count_condition = cc
        elif rng.random() < 0.06:
            child.count_condition = {
                "feature": int(rng.integers(0, n_feat)),
                "comp": int(rng.integers(0, n_comp)),
                "thresh": float(rng.uniform(0.1, 0.9)),
                "min_count": int(rng.integers(2, 8)),
                "window": int(rng.integers(5, 40)),
            }

        if parent.trend_condition and rng.random() > 0.2:
            child.trend_condition = parent.trend_condition.copy()
        elif rng.random() < 0.06:
            child.trend_condition = {
                "feature": int(rng.integers(0, n_feat)),
                "direction": int(rng.choice([-1, 1])),
                "threshold": float(rng.uniform(0.01, 0.1)),
            }

        if parent.chain_rule_idx >= 0 and rng.random() > 0.3:
            child.chain_rule_idx = parent.chain_rule_idx
        elif rng.random() < 0.05 and len(self.rules) > 1:
            child.chain_rule_idx = int(rng.integers(0, len(self.rules)))

        child.outcome = parent.outcome if rng.random() > 0.15 else int(rng.integers(0, n_out))
        child.action_bias = parent.action_bias.copy()
        mask = rng.random(len(child.action_bias)) < 0.3
        child.action_bias[mask] += rng.normal(0, 0.2, size=mask.sum())

        return child

    def get_best_rules(self, min_tests: int = 10, min_accuracy: float = 0.6) -> list[ComposableRule]:
        return [r for r in self.rules if r.tests >= min_tests and r.accuracy >= min_accuracy]

    def describe_best(self) -> list[str]:
        best = self.get_best_rules(min_tests=8, min_accuracy=0.55)
        best.sort(key=lambda r: r.accuracy * r.confidence * (1 + r.complexity * 0.1), reverse=True)
        return [r.describe() for r in best[:5]]

    @property
    def stats(self) -> dict:
        tested = [r for r in self.rules if r.tests >= 5]
        complex_rules = [r for r in tested if r.complexity > 1]
        return {
            "n_composable_rules": len(tested),
            "n_complex_rules": len(complex_rules),
            "max_complexity": max((r.complexity for r in self.rules), default=0),
            "mean_accuracy": float(np.mean([r.accuracy for r in tested])) if tested else 0,
        }
