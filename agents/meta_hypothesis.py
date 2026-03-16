
import numpy as np


class MetaRule:
    __slots__ = ['base_rule_idx', 'context_feature', 'context_comparator',
                 'context_threshold', 'accuracy_boost', 'tests', 'successes']

    def __init__(self, base_rule_idx: int, context_feature: int,
                 context_comparator: int, context_threshold: float):
        self.base_rule_idx = base_rule_idx
        self.context_feature = context_feature
        self.context_comparator = context_comparator
        self.context_threshold = context_threshold
        self.accuracy_boost = 0.0
        self.tests = 0
        self.successes = 0

    @property
    def accuracy(self) -> float:
        if self.tests < 3:
            return 0.5
        return self.successes / self.tests

    def context_active(self, features: np.ndarray) -> bool:
        val = features[self.context_feature] if self.context_feature < len(features) else 0
        if self.context_comparator == 0:
            return val > self.context_threshold
        return val < self.context_threshold

    def describe(self, base_rule_desc: str) -> str:
        from agents.hypothesis import Feature
        fname = Feature(self.context_feature).name if self.context_feature < 20 else "?"
        comp = ">" if self.context_comparator == 0 else "<"
        return (f"META: [{base_rule_desc}] works better WHEN "
                f"{fname} {comp} {self.context_threshold:.2f} "
                f"[acc={self.accuracy:.2f}, n={self.tests}]")


class ExperimentPlan:
    __slots__ = ['hypothesis_idx', 'action_to_try', 'expected_outcome',
                 'steps_remaining', 'observations']

    def __init__(self, hypothesis_idx: int, action_to_try: int,
                 expected_outcome: int, duration: int = 10):
        self.hypothesis_idx = hypothesis_idx
        self.action_to_try = action_to_try
        self.expected_outcome = expected_outcome
        self.steps_remaining = duration
        self.observations = []


class MetaHypothesisSystem:

    def __init__(self, max_meta_rules: int = 6, max_experiments: int = 2):
        self.meta_rules: list[MetaRule] = []
        self.max_meta_rules = max_meta_rules
        self.experiments: list[ExperimentPlan] = []
        self.max_experiments = max_experiments
        self._rule_accuracy_history: dict[int, list[float]] = {}

    def update(self, hypotheses, features: np.ndarray, outcome: np.ndarray,
               rng: np.random.Generator):
        for i, hyp in enumerate(hypotheses):
            if hyp.tests > 5:
                if i not in self._rule_accuracy_history:
                    self._rule_accuracy_history[i] = []
                self._rule_accuracy_history[i].append(hyp.accuracy)
                if len(self._rule_accuracy_history[i]) > 20:
                    self._rule_accuracy_history[i] = self._rule_accuracy_history[i][-20:]

        for mr in self.meta_rules:
            if mr.base_rule_idx < len(hypotheses):
                base = hypotheses[mr.base_rule_idx]
                if base.evaluate(features) and base.tests > 5:
                    mr.tests += 1
                    context_match = mr.context_active(features)
                    if context_match and base.accuracy > 0.6:
                        mr.successes += 1
                    elif not context_match and base.accuracy < 0.5:
                        mr.successes += 1

        for exp in self.experiments:
            exp.steps_remaining -= 1
            exp.observations.append(outcome.copy() if len(outcome) > 0 else np.zeros(1))

        self.experiments = [e for e in self.experiments if e.steps_remaining > 0]

        if rng.random() < 0.05:
            self._generate_meta_rule(hypotheses, features, rng)
        if rng.random() < 0.03 and len(self.experiments) < self.max_experiments:
            self._generate_experiment(hypotheses, rng)

        if len(self.meta_rules) > self.max_meta_rules:
            self.meta_rules.sort(key=lambda m: m.accuracy * m.tests, reverse=True)
            self.meta_rules = self.meta_rules[:self.max_meta_rules]

    def _generate_meta_rule(self, hypotheses, features, rng):
        for i, history in self._rule_accuracy_history.items():
            if len(history) < 5 or i >= len(hypotheses):
                continue
            std = np.std(history)
            if std > 0.1:
                context_feat = int(rng.integers(0, min(20, len(features))))
                context_comp = int(rng.integers(0, 2))
                context_thresh = float(features[context_feat]) if context_feat < len(features) else 0.5

                mr = MetaRule(i, context_feat, context_comp, context_thresh)
                self.meta_rules.append(mr)
                break

    def _generate_experiment(self, hypotheses, rng):
        for i, hyp in enumerate(hypotheses):
            if 5 < hyp.tests < 30 and 0.4 < hyp.accuracy < 0.7:
                action = int(rng.integers(0, 8))
                exp = ExperimentPlan(i, action, hyp.outcome, duration=10)
                self.experiments.append(exp)
                break

    def get_experiment_bias(self, action_dim: int) -> np.ndarray:
        bias = np.zeros(action_dim)
        for exp in self.experiments:
            if exp.action_to_try < action_dim:
                bias[exp.action_to_try] += 0.3
        return bias

    def get_context_accuracy_boost(self, hypotheses, features: np.ndarray) -> dict[int, float]:
        boosts = {}
        for mr in self.meta_rules:
            if mr.tests > 3 and mr.accuracy > 0.6:
                if mr.context_active(features):
                    boosts[mr.base_rule_idx] = boosts.get(mr.base_rule_idx, 0) + 0.1
                else:
                    boosts[mr.base_rule_idx] = boosts.get(mr.base_rule_idx, 0) - 0.05
        return boosts

    @property
    def stats(self) -> dict:
        tested = [m for m in self.meta_rules if m.tests >= 3]
        return {
            "n_meta_rules": len(tested),
            "mean_meta_accuracy": float(np.mean([m.accuracy for m in tested])) if tested else 0,
            "active_experiments": len(self.experiments),
        }

    def describe_meta_rules(self, hypotheses) -> list[str]:
        descs = []
        for mr in self.meta_rules:
            if mr.tests >= 5 and mr.accuracy > 0.55 and mr.base_rule_idx < len(hypotheses):
                base_desc = hypotheses[mr.base_rule_idx].describe()
                descs.append(mr.describe(base_desc))
        return descs
