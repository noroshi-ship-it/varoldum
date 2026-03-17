
import numpy as np


class LineagePool:
    """Shared knowledge pool for a lineage — best hypothesis rules."""

    __slots__ = ['rules', 'max_rules']

    def __init__(self, max_rules: int = 8):
        self.max_rules = max_rules
        self.rules: list[dict] = []  # [{encoded, accuracy, teacher_id, generation, tick}]

    def contribute(self, encoded_hyp, accuracy: float, teacher_id: int,
                   generation: int, tick: int) -> bool:
        """Add a rule if it beats the worst existing rule."""
        entry = {
            "encoded": encoded_hyp,
            "accuracy": accuracy,
            "teacher_id": teacher_id,
            "generation": generation,
            "tick": tick,
        }
        if len(self.rules) < self.max_rules:
            self.rules.append(entry)
            return True
        # Replace worst
        worst_idx = min(range(len(self.rules)), key=lambda i: self.rules[i]["accuracy"])
        if accuracy > self.rules[worst_idx]["accuracy"]:
            self.rules[worst_idx] = entry
            return True
        return False

    @property
    def mean_accuracy(self) -> float:
        if not self.rules:
            return 0.0
        return float(np.mean([r["accuracy"] for r in self.rules]))


class LineageMemorySystem:
    """World-level cultural memory organized by lineage."""

    def __init__(self, max_rules_per_lineage: int = 8):
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

    def study(self, lineage_id: int, rng: np.random.Generator) -> dict | None:
        """Draw a random rule from the lineage pool for a young agent to study."""
        if lineage_id not in self.pools:
            return None
        pool = self.pools[lineage_id]
        if not pool.rules:
            return None
        # Weighted by accuracy
        accuracies = np.array([r["accuracy"] for r in pool.rules])
        total = accuracies.sum()
        if total <= 0:
            idx = rng.integers(0, len(pool.rules))
        else:
            probs = accuracies / total
            probs = probs / probs.sum()  # ensure exact sum to 1.0
            idx = rng.choice(len(pool.rules), p=probs)
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
