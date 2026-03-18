
import numpy as np


class LineagePool:
    """Shared knowledge pool for a lineage — best hypothesis rules."""

    __slots__ = ['rules', 'max_rules']

    def __init__(self, max_rules: int = 24):
        self.max_rules = max_rules
        self.rules: list[dict] = []  # [{encoded, accuracy, teacher_id, generation, tick, use_count, innovation_score}]

    def contribute(self, encoded_hyp, accuracy: float, teacher_id: int,
                   generation: int, tick: int) -> bool:
        """Add a rule if it beats the worst existing rule."""
        entry = {
            "encoded": encoded_hyp,
            "accuracy": accuracy,
            "teacher_id": teacher_id,
            "generation": generation,
            "tick": tick,
            "use_count": 0,
            "innovation_score": 0.0,
        }
        if len(self.rules) < self.max_rules:
            self.rules.append(entry)
            return True
        # Replace worst (accounting for use_count)
        worst_idx = min(range(len(self.rules)),
                        key=lambda i: self.rules[i]["accuracy"] * (1.0 + 0.05 * self.rules[i].get("use_count", 0)))
        worst_val = self.rules[worst_idx]["accuracy"] * (1.0 + 0.05 * self.rules[worst_idx].get("use_count", 0))
        if accuracy > worst_val:
            self.rules[worst_idx] = entry
            return True
        return False

    @property
    def mean_accuracy(self) -> float:
        if not self.rules:
            return 0.0
        return float(np.mean([r["accuracy"] for r in self.rules]))

    @property
    def cultural_complexity(self) -> float:
        if not self.rules:
            return 0.0
        mean_acc = self.mean_accuracy
        fill = len(self.rules) / self.max_rules
        return mean_acc * fill


class LineageMemorySystem:
    """World-level cultural memory organized by lineage."""

    def __init__(self, max_rules_per_lineage: int = 24):
        self._max_rules = max_rules_per_lineage
        self.pools: dict[int, LineagePool] = {}

    def _get_pool(self, lineage_id: int) -> LineagePool:
        if lineage_id not in self.pools:
            self.pools[lineage_id] = LineagePool(self._max_rules)
        return self.pools[lineage_id]

    def contribute(self, lineage_id: int, encoded_hyp, accuracy: float,
                   teacher_id: int, generation: int, tick: int) -> bool:
        pool = self._get_pool(lineage_id)
        return pool.contribute(encoded_hyp, accuracy, teacher_id, generation, tick)

    def study(self, lineage_id: int, rng: np.random.Generator,
              agent_generation: int = 0) -> dict | None:
        """Draw a random rule from the lineage pool for a young agent to study."""
        if lineage_id not in self.pools:
            return None
        pool = self.pools[lineage_id]
        if not pool.rules:
            return None
        # Weighted by accuracy × use_count_bonus × generational relevance
        weights = np.zeros(len(pool.rules))
        for i, r in enumerate(pool.rules):
            use_bonus = 1.0 + 0.1 * min(r.get("use_count", 0), 10)
            gen_rel = 1.0 / (1.0 + 0.01 * abs(agent_generation - r.get("generation", 0)))
            weights[i] = r["accuracy"] * use_bonus * gen_rel
        total = weights.sum()
        if total <= 0:
            idx = rng.integers(0, len(pool.rules))
        else:
            probs = weights / total
            probs = probs / probs.sum()
            idx = rng.choice(len(pool.rules), p=probs)
        pool.rules[idx]["use_count"] = pool.rules[idx].get("use_count", 0) + 1
        return pool.rules[idx]

    def cleanup(self, active_lineages: set[int]):
        """Remove pools for extinct lineages."""
        dead = [lid for lid in self.pools if lid not in active_lineages]
        for lid in dead:
            del self.pools[lid]

    @property
    def active_pool_count(self) -> int:
        return len(self.pools)

    @property
    def total_rules(self) -> int:
        return sum(len(p.rules) for p in self.pools.values())

    def get_stats(self) -> dict:
        if not self.pools:
            return {"pools": 0, "rules": 0, "mean_accuracy": 0.0}
        accuracies = [p.mean_accuracy for p in self.pools.values() if p.rules]
        return {
            "pools": len(self.pools),
            "rules": self.total_rules,
            "mean_accuracy": float(np.mean(accuracies)) if accuracies else 0.0,
        }
