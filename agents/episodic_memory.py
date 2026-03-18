
import numpy as np


class Episode:
    __slots__ = ['concepts', 'position', 'tick', 'emotional_valence',
                 'agents_involved', 'action_taken', 'outcome',
                 'surprise', 'detail_level', 'access_count']

    def __init__(self, concepts, position, tick, emotional_valence,
                 agents_involved, action_taken, outcome, surprise):
        self.concepts = concepts.copy()
        self.position = position.copy()
        self.tick = int(tick)
        self.emotional_valence = float(emotional_valence)
        self.agents_involved = list(agents_involved[:3])
        self.action_taken = action_taken.copy()
        self.outcome = float(outcome)
        self.surprise = float(surprise)
        self.detail_level = 1.0
        self.access_count = 0


class EpisodicMemorySystem:
    """Stores and retrieves discrete episodic memories (events)."""

    def __init__(self, capacity: int, bottleneck_size: int, action_dim: int,
                 surprise_threshold: float, consolidation_rate: float,
                 emotion_weight: float):
        self.capacity = max(2, int(capacity))
        self.bottleneck_size = bottleneck_size
        self.action_dim = action_dim
        self.surprise_threshold = surprise_threshold
        self.consolidation_rate = consolidation_rate
        self.emotion_weight = emotion_weight
        self.episodes: list[Episode] = []
        self._last_stored_tick = -100

    def maybe_store(self, concepts: np.ndarray, position: np.ndarray,
                    tick: int, internal_state: np.ndarray,
                    agents_nearby: list[int], action: np.ndarray,
                    reward: float, surprise: float) -> bool:
        """Store an episode if the event is significant enough."""
        # Compute emotional valence from internal state deviation
        emotional_valence = float(np.sum(np.abs(internal_state - 0.5)))

        # Check triggers
        is_surprising = surprise > self.surprise_threshold
        is_high_reward = abs(reward) > 0.3
        is_emotional = emotional_valence > 3.0
        is_social = len(agents_nearby) > 0 and (
            len(action) > 3 and abs(float(action[3])) > 0.3
        )

        if not (is_surprising or is_high_reward or is_emotional or is_social):
            return False

        # Rate limit: don't store more than once per 5 ticks
        if tick - self._last_stored_tick < 5:
            return False

        episode = Episode(
            concepts=concepts,
            position=position,
            tick=tick,
            emotional_valence=emotional_valence,
            agents_involved=agents_nearby,
            action_taken=action,
            outcome=reward,
            surprise=surprise,
        )

        if len(self.episodes) >= self.capacity:
            # Evict lowest-value episode
            worst_idx = 0
            worst_val = float('inf')
            for i, ep in enumerate(self.episodes):
                val = ep.detail_level * (1.0 + 0.1 * ep.access_count)
                if val < worst_val:
                    worst_val = val
                    worst_idx = i
            self.episodes[worst_idx] = episode
        else:
            self.episodes.append(episode)

        self._last_stored_tick = tick
        return True

    def retrieve(self, query_concepts: np.ndarray, query_emotion: float,
                 current_tick: int, top_k: int = 3) -> list[Episode]:
        """Retrieve most relevant episodes by similarity scoring."""
        if not self.episodes:
            return []

        scores = []
        q_norm = np.linalg.norm(query_concepts)
        if q_norm < 1e-8:
            q_norm = 1.0

        for ep in self.episodes:
            # Concept similarity
            ep_norm = np.linalg.norm(ep.concepts)
            if ep_norm < 1e-8:
                concept_sim = 0.0
            else:
                n = min(len(query_concepts), len(ep.concepts))
                concept_sim = float(np.dot(query_concepts[:n], ep.concepts[:n])) / (q_norm * ep_norm)

            # Emotion match
            emotion_match = 1.0 - min(1.0, abs(query_emotion - ep.emotional_valence) / 5.0)

            # Recency decay
            age = current_tick - ep.tick
            recency = float(np.exp(-0.001 * age))

            score = (0.5 * concept_sim
                     + self.emotion_weight * emotion_match
                     + 0.2 * recency
                     + 0.1 * ep.detail_level)
            scores.append(score)

        # Get top-k indices
        k = min(top_k, len(self.episodes))
        indices = np.argpartition(scores, -k)[-k:]
        indices = sorted(indices, key=lambda i: scores[i], reverse=True)

        result = []
        for i in indices:
            self.episodes[i].access_count += 1
            result.append(self.episodes[i])
        return result

    def replay(self, episode: Episode) -> tuple[np.ndarray, np.ndarray, float]:
        """Return (concepts, action, outcome) scaled by detail_level for think() replay."""
        dl = episode.detail_level
        concepts = episode.concepts * dl
        action = episode.action_taken * dl
        outcome = episode.outcome * dl
        return concepts, action, outcome

    def consolidate(self, current_tick: int):
        """Compress old memories: reduce detail, eventually convert to gist."""
        for ep in self.episodes:
            age = current_tick - ep.tick
            if age < 50:
                continue

            # Frequently accessed episodes consolidate slower
            decay = self.consolidation_rate / (1.0 + 0.1 * ep.access_count)
            ep.detail_level *= (1.0 - decay)

            # Below threshold: convert to gist (lose specific details)
            if ep.detail_level < 0.1:
                ep.action_taken = np.zeros_like(ep.action_taken)
                ep.position = np.zeros_like(ep.position)
                ep.agents_involved = []

    def get_context_vector(self, query_concepts: np.ndarray,
                           query_emotion: float, current_tick: int,
                           max_dim: int = 8) -> np.ndarray:
        """Return fixed-size context vector from top episodes."""
        result = np.zeros(max_dim, dtype=np.float64)
        if not self.episodes:
            return result

        top = self.retrieve(query_concepts, query_emotion, current_tick, top_k=3)
        if not top:
            return result

        # Weighted average of episode concepts
        total_weight = 0.0
        for ep in top:
            w = ep.detail_level * (1.0 + 0.1 * ep.access_count)
            n = min(max_dim, len(ep.concepts))
            result[:n] += ep.concepts[:n] * w
            total_weight += w

        if total_weight > 0:
            result /= total_weight

        return result

    @property
    def param_count(self) -> int:
        return 0

    @property
    def episode_count(self) -> int:
        return len(self.episodes)

    @property
    def mean_detail_level(self) -> float:
        if not self.episodes:
            return 0.0
        return float(np.mean([ep.detail_level for ep in self.episodes]))
