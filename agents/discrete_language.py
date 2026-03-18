
import numpy as np

MAX_VOCAB = 16
MAX_UTTERANCE_LEN = 3


class TokenUtterance:
    __slots__ = ['tokens', 'sender_id', 'sender_pos', 'sender_lineage', 'tick']

    def __init__(self, tokens: np.ndarray, sender_id: int, sender_pos: np.ndarray,
                 sender_lineage: int, tick: int):
        self.tokens = tokens.copy()
        self.sender_id = sender_id
        self.sender_pos = sender_pos.copy()
        self.sender_lineage = sender_lineage
        self.tick = tick


class DiscreteVocab:
    """Per-agent discrete token vocabulary grounded in concept space.
    Each token has a meaning vector that evolves through use."""

    __slots__ = ['vocab_active', 'utterance_length', 'listen_weight',
                 'token_meanings', 'token_reliability', '_bottleneck_size',
                 '_usage_count']

    def __init__(self, bottleneck_size: int, vocab_active: int = 8,
                 utterance_length: int = 2, listen_weight: float = 0.5):
        self.vocab_active = max(2, min(MAX_VOCAB, vocab_active))
        self.utterance_length = max(1, min(MAX_UTTERANCE_LEN, utterance_length))
        self.listen_weight = np.clip(listen_weight, 0.0, 1.0)
        self._bottleneck_size = bottleneck_size

        # Token meaning vectors — each token maps to a point in concept space
        self.token_meanings = np.random.randn(MAX_VOCAB, bottleneck_size).astype(np.float32) * 0.3
        # Normalize
        norms = np.linalg.norm(self.token_meanings, axis=1, keepdims=True)
        self.token_meanings /= np.maximum(norms, 1e-8)

        self.token_reliability = np.zeros(MAX_VOCAB, dtype=np.float32)
        self._usage_count = np.zeros(MAX_VOCAB, dtype=np.int32)

    def encode_utterance(self, concepts: np.ndarray, speak_intent: float) -> np.ndarray | None:
        """Convert current concepts into a token sequence. Returns None if not speaking."""
        if speak_intent <= 0:
            return None

        c = concepts[:self._bottleneck_size].astype(np.float32)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-8:
            return None

        c_unit = c / c_norm

        # Similarity with all active token meanings
        sims = self.token_meanings[:self.vocab_active] @ c_unit
        tokens = np.zeros(self.utterance_length, dtype=np.int32)

        # First token: best match
        tokens[0] = int(np.argmax(sims))
        self._usage_count[tokens[0]] += 1

        if self.utterance_length > 1:
            # Additional tokens: residual encoding
            residual = c_unit - self.token_meanings[tokens[0]]
            for t in range(1, self.utterance_length):
                res_sims = self.token_meanings[:self.vocab_active] @ residual
                # Mask already-used tokens
                for prev in range(t):
                    res_sims[tokens[prev]] = -999.0
                tokens[t] = int(np.argmax(res_sims))
                self._usage_count[tokens[t]] += 1
                residual = residual - self.token_meanings[tokens[t]]

        return tokens

    def decode_utterance(self, tokens: np.ndarray) -> np.ndarray:
        """Decode received tokens back to concept space meaning."""
        result = np.zeros(self._bottleneck_size, dtype=np.float32)
        n_tokens = min(len(tokens), self.utterance_length)
        weight = 1.0
        total_w = 0.0
        for i in range(n_tokens):
            tid = int(tokens[i])
            if 0 <= tid < MAX_VOCAB:
                result += self.token_meanings[tid] * weight
                total_w += weight
                weight *= 0.7  # Later tokens have less weight
        if total_w > 1e-8:
            result /= total_w
        return result * self.listen_weight

    def ground_speaker(self, token_id: int, concepts: np.ndarray, lr: float = 0.01):
        """Speaker grounding: token meaning drifts toward what was experienced."""
        if token_id < 0 or token_id >= MAX_VOCAB:
            return
        c = concepts[:self._bottleneck_size].astype(np.float32)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-8:
            return
        c_unit = c / c_norm
        self.token_meanings[token_id] = (1.0 - lr) * self.token_meanings[token_id] + lr * c_unit
        norm = np.linalg.norm(self.token_meanings[token_id])
        if norm > 1e-8:
            self.token_meanings[token_id] /= norm

    def ground_listener(self, token_id: int, concepts: np.ndarray, outcome_good: bool,
                        lr: float = 0.005):
        """Listener grounding: if hearing token T leads to good outcome,
        listener's meaning for T drifts toward current experience."""
        if token_id < 0 or token_id >= MAX_VOCAB:
            return
        if outcome_good:
            c = concepts[:self._bottleneck_size].astype(np.float32)
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-8:
                return
            c_unit = c / c_norm
            self.token_meanings[token_id] = (
                (1.0 - lr) * self.token_meanings[token_id] + lr * c_unit
            )
            norm = np.linalg.norm(self.token_meanings[token_id])
            if norm > 1e-8:
                self.token_meanings[token_id] /= norm
            self.token_reliability[token_id] = min(
                1.0, self.token_reliability[token_id] + 0.02)
        else:
            self.token_reliability[token_id] = max(
                0.0, self.token_reliability[token_id] - 0.01)

    @property
    def stats(self) -> dict:
        active = int(np.sum(self._usage_count[:self.vocab_active] > 0))
        total = int(np.sum(self._usage_count))
        mean_rel = float(np.mean(self.token_reliability[:self.vocab_active]))
        return {
            "vocab_used": active,
            "vocab_active": self.vocab_active,
            "total_tokens_sent": total,
            "mean_reliability": mean_rel,
        }


# Re-export StructuredUtterance for convenience
# (StructuredUtterance is a subclass of TokenUtterance defined in grammar.py)
# Import it here so callers can check isinstance without importing grammar.py
def _is_structured(utterance) -> bool:
    """Check if an utterance has grammatical role annotations."""
    return hasattr(utterance, 'roles')


class DiscreteLanguageSystem:
    """World-level system that manages token-based communication between agents."""

    def __init__(self, world_width: int, world_height: int, hear_radius: int = 8):
        self.w = world_width
        self.h = world_height
        self.hear_radius = hear_radius
        self._utterances: list[TokenUtterance] = []
        self._tick_stats = {"tokens_sent": 0, "tokens_heard": 0, "unique_speakers": 0}

    def broadcast(self, agent_id: int, agent_pos: np.ndarray, agent_lineage: int,
                  tokens: np.ndarray, tick: int):
        self._utterances.append(TokenUtterance(
            tokens=tokens, sender_id=agent_id, sender_pos=agent_pos,
            sender_lineage=agent_lineage, tick=tick,
        ))
        self._tick_stats["tokens_sent"] += len(tokens)

    def get_nearest_utterance(self, agent_id: int, agent_pos: np.ndarray,
                              tick: int) -> TokenUtterance | None:
        ax, ay = agent_pos[0], agent_pos[1]
        best = None
        best_dist = float('inf')
        for u in self._utterances:
            if u.sender_id == agent_id:
                continue
            if u.tick < tick - 3:
                continue
            dx = abs(u.sender_pos[0] - ax)
            dy = abs(u.sender_pos[1] - ay)
            dx = min(dx, self.w - dx)
            dy = min(dy, self.h - dy)
            d = dx + dy
            if d <= self.hear_radius and d < best_dist:
                best_dist = d
                best = u
        if best is not None:
            self._tick_stats["tokens_heard"] += len(best.tokens)
        return best

    def cleanup(self, tick: int):
        self._utterances = [u for u in self._utterances if u.tick >= tick - 3]

    @property
    def recent_stats(self) -> dict:
        stats = self._tick_stats.copy()
        stats["unique_speakers"] = len(set(u.sender_id for u in self._utterances))
        self._tick_stats = {"tokens_sent": 0, "tokens_heard": 0, "unique_speakers": 0}
        return stats
