
import numpy as np
from agents.discrete_language import TokenUtterance, DiscreteVocab, MAX_VOCAB


MAX_GRAMMAR_SLOTS = 5


class StructuredUtterance(TokenUtterance):
    """A token utterance with grammatical role annotations.

    Each token occupies a slot with a specific grammatical role,
    enabling compositional meaning where word order matters.
    """
    __slots__ = ['roles']

    def __init__(self, tokens: np.ndarray, roles: np.ndarray,
                 sender_id: int, sender_pos: np.ndarray,
                 sender_lineage: int, tick: int):
        super().__init__(tokens, sender_id, sender_pos, sender_lineage, tick)
        self.roles = roles.copy()


class GrammarSystem:
    """Phase 7: Compositional grammar with slot-based roles.

    Each slot has a learned role embedding that gates which concept
    dimensions are expressed there. Different slots naturally specialize
    to encode different aspects (spatial, emotional, intentional).

    Word order matters: swapping tokens between slots changes meaning
    because each slot filters different concept dimensions.
    """

    __slots__ = [
        'bottleneck_size', 'n_slots', 'grammar_lr', 'grammar_weight',
        'role_embeddings', '_role_usage',
    ]

    def __init__(self, bottleneck_size: int, n_slots: int,
                 grammar_lr: float, grammar_weight: float):
        self.bottleneck_size = bottleneck_size
        self.n_slots = max(2, min(MAX_GRAMMAR_SLOTS, n_slots))
        self.grammar_lr = np.clip(grammar_lr, 0.001, 0.05)
        self.grammar_weight = np.clip(grammar_weight, 0.0, 1.0)

        # Role embeddings: each role gates specific concept dimensions
        # Initialized with random orthogonal-ish vectors
        self.role_embeddings = np.random.randn(
            self.n_slots, bottleneck_size
        ).astype(np.float32) * 0.5
        # Normalize each role to unit vector
        for i in range(self.n_slots):
            norm = np.linalg.norm(self.role_embeddings[i])
            if norm > 1e-8:
                self.role_embeddings[i] /= norm

        self._role_usage = np.zeros(self.n_slots, dtype=np.int32)

    def encode_structured(self, concepts: np.ndarray, speak_intent: float,
                          vocab: DiscreteVocab,
                          referent_token: int | None = None,
                          referential_weight: float = 0.0) -> 'StructuredUtterance | None':
        """Encode concepts into a structured utterance with grammatical roles.
        If referent_token is provided and referential_weight > 0.3,
        the last slot encodes the referent instead of gated concepts."""
        if speak_intent <= 0:
            return None

        c = concepts[:self.bottleneck_size].astype(np.float32)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-8:
            return None

        tokens = np.zeros(self.n_slots, dtype=np.int32)
        roles = np.arange(self.n_slots, dtype=np.int32)

        # Determine if last slot is a referent slot
        use_referent = (referent_token is not None and referential_weight > 0.3
                        and self.n_slots >= 3)
        concept_slots = self.n_slots - 1 if use_referent else self.n_slots

        for slot in range(concept_slots):
            # Gate concepts through this slot's role embedding
            gated = c * self.role_embeddings[slot]  # element-wise
            gated_norm = np.linalg.norm(gated)
            if gated_norm < 1e-8:
                tokens[slot] = 0
                continue

            gated_unit = gated / gated_norm

            # Find best matching token from vocab
            sims = vocab.token_meanings[:vocab.vocab_active] @ gated_unit
            # Mask already-used tokens
            for prev in range(slot):
                if tokens[prev] < vocab.vocab_active:
                    sims[tokens[prev]] = -999.0
            tokens[slot] = int(np.argmax(sims))
            self._role_usage[slot] += 1

        # Referent slot: encode the named entity's token
        if use_referent:
            tokens[self.n_slots - 1] = referent_token
            self._role_usage[self.n_slots - 1] += 1

        return StructuredUtterance(
            tokens=tokens, roles=roles,
            sender_id=0, sender_pos=np.zeros(2),
            sender_lineage=0, tick=0,
        )

    def decode_structured(self, utterance, vocab: DiscreteVocab) -> np.ndarray:
        """Decode a structured utterance back to concept space."""
        result = np.zeros(self.bottleneck_size, dtype=np.float32)

        if not hasattr(utterance, 'roles'):
            # Fallback for plain TokenUtterance
            return vocab.decode_utterance(utterance.tokens)

        n_tokens = min(len(utterance.tokens), self.n_slots)
        total_weight = 0.0

        for i in range(n_tokens):
            tid = int(utterance.tokens[i])
            if tid < 0 or tid >= MAX_VOCAB:
                continue
            role_idx = int(utterance.roles[i]) if i < len(utterance.roles) else i
            if role_idx >= self.n_slots:
                role_idx = i % self.n_slots

            # Token meaning un-gated through role embedding
            token_meaning = vocab.token_meanings[tid]
            unmasked = token_meaning * self.role_embeddings[role_idx]
            weight = 1.0 / (1.0 + i * 0.3)  # slight decay for later slots
            result += unmasked * weight
            total_weight += weight

        if total_weight > 1e-8:
            result /= total_weight

        return result * vocab.listen_weight

    def ground_speaker_roles(self, slot_idx: int, concepts: np.ndarray, lr: float = None):
        """Speaker grounding: role embedding drifts toward gating pattern that worked."""
        if slot_idx < 0 or slot_idx >= self.n_slots:
            return
        if lr is None:
            lr = self.grammar_lr

        c = concepts[:self.bottleneck_size].astype(np.float32)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-8:
            return

        # The ideal role embedding maximizes the gated information for this slot
        # Drift toward the absolute value pattern of concepts (captures which dims are active)
        target = np.abs(c) / c_norm
        self.role_embeddings[slot_idx] = (
            (1.0 - lr) * self.role_embeddings[slot_idx] + lr * target
        )
        # Re-normalize
        norm = np.linalg.norm(self.role_embeddings[slot_idx])
        if norm > 1e-8:
            self.role_embeddings[slot_idx] /= norm

    def ground_listener_roles(self, slot_idx: int, concepts: np.ndarray,
                               outcome_good: bool, lr: float = None):
        """Listener grounding: if outcome was good, align role embedding."""
        if not outcome_good:
            return
        if slot_idx < 0 or slot_idx >= self.n_slots:
            return
        if lr is None:
            lr = self.grammar_lr * 0.5  # listener learns slower

        c = concepts[:self.bottleneck_size].astype(np.float32)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-8:
            return

        target = np.abs(c) / c_norm
        self.role_embeddings[slot_idx] = (
            (1.0 - lr) * self.role_embeddings[slot_idx] + lr * target
        )
        norm = np.linalg.norm(self.role_embeddings[slot_idx])
        if norm > 1e-8:
            self.role_embeddings[slot_idx] /= norm

    @property
    def param_count(self) -> int:
        return self.role_embeddings.size

    def get_role_descriptions(self) -> list[dict]:
        """Return interpretable description of each grammatical role."""
        descriptions = []
        for i in range(self.n_slots):
            role = self.role_embeddings[i]
            # Find top-3 most weighted dimensions
            top_dims = np.argsort(np.abs(role))[-3:][::-1]
            specialization = float(np.max(np.abs(role)) / (np.mean(np.abs(role)) + 1e-8))
            descriptions.append({
                "slot": i,
                "top_dims": top_dims.tolist(),
                "top_weights": [float(role[d]) for d in top_dims],
                "specialization": specialization,
                "usage": int(self._role_usage[i]),
            })
        return descriptions

    @property
    def stats(self) -> dict:
        # Measure role differentiation: how different are roles from each other?
        if self.n_slots < 2:
            differentiation = 0.0
        else:
            cos_sims = []
            for i in range(self.n_slots):
                for j in range(i + 1, self.n_slots):
                    dot = float(self.role_embeddings[i] @ self.role_embeddings[j])
                    cos_sims.append(abs(dot))
            differentiation = 1.0 - float(np.mean(cos_sims)) if cos_sims else 0.0

        return {
            "n_slots": self.n_slots,
            "grammar_weight": self.grammar_weight,
            "role_differentiation": differentiation,
            "total_usage": int(np.sum(self._role_usage)),
        }
