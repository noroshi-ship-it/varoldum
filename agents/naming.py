
import numpy as np


class NameBinding:
    __slots__ = ['token_id', 'entity_type', 'entity_id', 'concept_signature',
                 'last_seen_pos', 'last_seen_tick', 'strength', 'shared_count']

    def __init__(self, token_id: int, entity_type: str, entity_id: int,
                 concept_signature: np.ndarray, position: np.ndarray, tick: int):
        self.token_id = token_id
        self.entity_type = entity_type
        self.entity_id = entity_id
        self.concept_signature = concept_signature.copy()
        self.last_seen_pos = position.copy()
        self.last_seen_tick = tick
        self.strength = 0.5
        self.shared_count = 0


class NamingSystem:
    """Agents learn to associate internal 'name tokens' with specific entities.
    Supports abstract (non-physical) referents for shared fiction."""

    def __init__(self, capacity: int, bottleneck_size: int,
                 learning_rate: float, referential_weight: float):
        self.capacity = max(1, int(capacity))
        self.bottleneck_size = bottleneck_size
        self.learning_rate = learning_rate
        self.referential_weight = referential_weight
        self.abstract_capacity = 0  # set externally from gene
        # token_id -> NameBinding
        self.name_registry: dict[int, NameBinding] = {}
        # (entity_type, entity_id) -> token_id
        self.entity_to_name: dict[tuple[str, int], int] = {}

    def assign_name(self, token_id: int, entity_type: str, entity_id: int,
                    concept_sig: np.ndarray, position: np.ndarray,
                    tick: int) -> bool:
        """Bind a token to an entity. Evicts weakest if at capacity."""
        key = (entity_type, entity_id)

        # Already named with this token
        if key in self.entity_to_name and self.entity_to_name[key] == token_id:
            binding = self.name_registry[token_id]
            binding.strength = min(1.0, binding.strength + 0.1)
            return True

        # At capacity — evict weakest
        if len(self.name_registry) >= self.capacity and token_id not in self.name_registry:
            weakest_tid = min(self.name_registry, key=lambda t: self.name_registry[t].strength)
            weakest = self.name_registry[weakest_tid]
            old_key = (weakest.entity_type, weakest.entity_id)
            self.entity_to_name.pop(old_key, None)
            del self.name_registry[weakest_tid]

        # If this token was bound to something else, unbind it
        if token_id in self.name_registry:
            old = self.name_registry[token_id]
            old_key = (old.entity_type, old.entity_id)
            self.entity_to_name.pop(old_key, None)

        # If this entity had a different name, unbind it
        if key in self.entity_to_name:
            old_tid = self.entity_to_name[key]
            self.name_registry.pop(old_tid, None)

        binding = NameBinding(token_id, entity_type, entity_id,
                              concept_sig, position, tick)
        self.name_registry[token_id] = binding
        self.entity_to_name[key] = token_id
        return True

    def get_name_for_entity(self, entity_type: str, entity_id: int) -> int | None:
        """Lookup: do I have a name for this entity?"""
        return self.entity_to_name.get((entity_type, entity_id))

    def get_entity_for_name(self, token_id: int) -> NameBinding | None:
        """Reverse lookup: what does this name refer to?"""
        return self.name_registry.get(token_id)

    def update_sighting(self, entity_type: str, entity_id: int,
                        position: np.ndarray, concept_sig: np.ndarray,
                        tick: int):
        """Update a named entity when seen. Strengthen binding."""
        key = (entity_type, entity_id)
        tid = self.entity_to_name.get(key)
        if tid is None:
            return

        binding = self.name_registry[tid]
        binding.last_seen_pos = position.copy()
        binding.last_seen_tick = tick
        binding.strength = min(1.0, binding.strength + 0.02)

        # Drift concept signature toward current observation
        n = min(len(binding.concept_signature), len(concept_sig))
        binding.concept_signature[:n] += self.learning_rate * (
            concept_sig[:n] - binding.concept_signature[:n]
        )

    def attempt_grounding(self, heard_token_id: int, attended_entity_id: int,
                          entity_type: str, concept_sig: np.ndarray,
                          position: np.ndarray, tick: int):
        """Ground a heard token to an attended entity (shared naming)."""
        existing = self.name_registry.get(heard_token_id)

        if existing is not None:
            if existing.entity_id == attended_entity_id and existing.entity_type == entity_type:
                # Already matches — strengthen
                existing.strength = min(1.0, existing.strength + 0.1)
                existing.shared_count += 1
            else:
                # Conflict — weaken existing
                existing.strength -= 0.05
                if existing.strength < 0.1:
                    # Rebind to new entity
                    self.assign_name(heard_token_id, entity_type, attended_entity_id,
                                     concept_sig, position, tick)
        else:
            # Unbound token — bind to attended entity
            self.assign_name(heard_token_id, entity_type, attended_entity_id,
                             concept_sig, position, tick)

    def decay(self, current_tick: int):
        """Unused names fade over time."""
        to_remove = []
        for tid, binding in self.name_registry.items():
            age = current_tick - binding.last_seen_tick
            if age > 100:
                binding.strength *= 0.998
            if binding.strength < 0.05:
                to_remove.append(tid)

        for tid in to_remove:
            binding = self.name_registry[tid]
            key = (binding.entity_type, binding.entity_id)
            self.entity_to_name.pop(key, None)
            del self.name_registry[tid]

    def get_referent_context(self, attended_entity_type: str | None,
                             attended_entity_id: int | None,
                             max_dim: int = 4) -> np.ndarray:
        """Return concept signature of named attended entity as context."""
        result = np.zeros(max_dim, dtype=np.float64)
        if attended_entity_type is None or attended_entity_id is None:
            return result

        key = (attended_entity_type, attended_entity_id)
        tid = self.entity_to_name.get(key)
        if tid is None:
            return result

        binding = self.name_registry[tid]
        n = min(max_dim, len(binding.concept_signature))
        result[:n] = binding.concept_signature[:n]
        return result

    def encode_referent(self, entity_type: str, entity_id: int) -> int | None:
        """Get token ID for a named entity (for structured utterances)."""
        return self.entity_to_name.get((entity_type, entity_id))

    def mint_abstract_name(self, token_id: int, concept_signature: np.ndarray,
                           tick: int) -> bool:
        """Bind a token to an abstract concept (no physical entity).
        Only if abstract_capacity > 0 and current abstracts < capacity."""
        if self.abstract_capacity <= 0:
            return False
        # Count current abstract bindings
        abstract_count = sum(1 for b in self.name_registry.values()
                             if b.entity_type == "abstract")
        if abstract_count >= self.abstract_capacity:
            return False

        abstract_id = int(np.sum(np.abs(concept_signature[:8]) * 1000)) % 100000
        return self.assign_name(
            token_id, "abstract", abstract_id,
            concept_signature, np.zeros(2), tick
        )

    def should_mint_abstract(self, concepts: np.ndarray,
                             emotional_salience: float,
                             min_salience: float = 0.3) -> bool:
        """Check if agent should try to name an abstract concept."""
        if self.abstract_capacity <= 0:
            return False
        if emotional_salience < min_salience:
            return False
        c_norm = np.linalg.norm(concepts)
        if c_norm < 1e-8:
            return False
        c_unit = concepts / c_norm
        for binding in self.name_registry.values():
            b_norm = np.linalg.norm(binding.concept_signature)
            if b_norm < 1e-8:
                continue
            plen = min(len(c_unit), len(binding.concept_signature))
            sim = float(np.dot(c_unit[:plen],
                               binding.concept_signature[:plen] / b_norm))
            if sim > 0.7:
                return False
        return True

    @property
    def param_count(self) -> int:
        return 0

    @property
    def named_count(self) -> int:
        return len(self.name_registry)

    @property
    def mean_strength(self) -> float:
        if not self.name_registry:
            return 0.0
        return float(np.mean([b.strength for b in self.name_registry.values()]))
