
import numpy as np


class ConceptLabeler:
    """Learns human-readable labels for concept dimensions based on
    population-level correlations between concepts and behaviors."""

    def __init__(self, max_dims: int = 32):
        self.max_dims = max_dims
        # Correlation accumulators: concept_dim → behavior → running_stats
        self._behavior_names = [
            "eating", "moving", "signaling", "fighting", "sharing",
            "high_energy", "low_health", "near_danger", "near_food",
        ]
        self._correlations = np.zeros((max_dims, len(self._behavior_names)), dtype=np.float64)
        self._counts = 0
        self._labels: dict[int, str] = {}

    def observe(self, agent):
        """Observe one agent's concept-behavior correlation."""
        concepts = agent.brain.get_concepts()
        n = min(len(concepts), self.max_dims)

        # Compute behavior indicators
        action = getattr(agent, '_action', np.zeros(6))
        behaviors = np.array([
            max(0, float(action[2])) if len(action) > 2 else 0,   # eating (mouth > 0)
            float(np.sqrt(action[0]**2 + action[1]**2)) if len(action) > 1 else 0,  # moving
            max(0, float(action[5])) if len(action) > 5 else 0,   # signaling
            max(0, -float(action[3])) if len(action) > 3 else 0,  # fighting (social < 0)
            max(0, float(action[3])) if len(action) > 3 else 0,   # sharing (social > 0)
            float(agent.body.energy),  # high energy
            float(1.0 - agent.body.health),  # low health
            float(agent.internal.fear),  # near danger proxy
            float(1.0 - agent.internal.hunger),  # near food proxy
        ], dtype=np.float64)

        # Accumulate correlation
        for i in range(n):
            self._correlations[i] += concepts[i] * behaviors
        self._counts += 1

    def update_labels(self):
        """Recompute labels from accumulated correlations."""
        if self._counts < 50:
            return

        avg = self._correlations / self._counts
        self._labels = {}

        for dim in range(min(self.max_dims, avg.shape[0])):
            row = avg[dim]
            best_idx = int(np.argmax(np.abs(row)))
            strength = abs(row[best_idx])
            if strength > 0.05:
                sign = "+" if row[best_idx] > 0 else "-"
                self._labels[dim] = f"{sign}{self._behavior_names[best_idx]}"

    def get_label(self, dim: int) -> str:
        return self._labels.get(dim, f"C{dim}")

    def label_concepts(self, concepts: np.ndarray) -> dict[str, float]:
        """Return labeled concept activations."""
        result = {}
        for i in range(min(len(concepts), self.max_dims)):
            label = self.get_label(i)
            if abs(concepts[i]) > 0.2:
                result[label] = float(concepts[i])
        return result

    def reset(self):
        self._correlations[:] = 0
        self._counts = 0


class CognitionInterpreter:
    """Zero-cost interpretability layer for understanding agent cognition."""

    def __init__(self):
        self.labeler = ConceptLabeler()
        self._observation_count = 0

    def observe_population(self, agents, max_samples: int = 200):
        """Sample agents for concept-behavior correlation analysis."""
        alive = [a for a in agents if a.is_alive]
        n = min(len(alive), max_samples)
        for agent in alive[:n]:
            self.labeler.observe(agent)
        self._observation_count += n

        # Update labels periodically
        if self._observation_count >= 500:
            self.labeler.update_labels()
            self.labeler.reset()
            self._observation_count = 0

    def interpret_agent(self, agent) -> dict:
        """Full interpretable state of an agent."""
        concepts = agent.brain.get_concepts()
        labeled = self.labeler.label_concepts(concepts)

        meta_info = agent.meta_concepts.introspect()
        grammar_roles = agent.grammar.get_role_descriptions()
        tom_beliefs = agent.tom.describe_beliefs()

        return {
            "labeled_concepts": labeled,
            "meta_pattern": meta_info["pattern"],
            "meta_accuracy": meta_info["meta_wm_accuracy"],
            "grammar_roles": grammar_roles,
            "grammar_weight": agent.grammar.grammar_weight,
            "tom_beliefs": tom_beliefs,
        }
