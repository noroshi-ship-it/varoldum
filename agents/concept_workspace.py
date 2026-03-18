
import numpy as np


class ConceptWorkspace:
    """Gene-controlled scratchpad: N vector slots the brain can read/write.

    Enables multi-step concept manipulation during think().
    Zero slots = no workspace = zero cost = backward compatible.
    """

    __slots__ = [
        'n_slots', 'bottleneck_size', 'gate_strength',
        'slots', '_read_weights', '_write_count',
    ]

    def __init__(self, n_slots: int, bottleneck_size: int,
                 gate_strength: float):
        self.n_slots = max(0, int(n_slots))
        self.bottleneck_size = bottleneck_size
        self.gate_strength = float(np.clip(gate_strength, 0.0, 1.0))
        self.slots = np.zeros((max(1, self.n_slots), bottleneck_size),
                              dtype=np.float32)
        self._read_weights = np.zeros(max(1, self.n_slots), dtype=np.float32)
        self._write_count = 0

    def read(self, query: np.ndarray) -> np.ndarray:
        """Soft attention read: query against all slots, return weighted sum."""
        if self.n_slots == 0:
            return np.zeros(self.bottleneck_size, dtype=np.float32)

        q_norm = np.linalg.norm(query)
        if q_norm < 1e-8:
            return np.zeros(self.bottleneck_size, dtype=np.float32)

        q_unit = query[:self.bottleneck_size].astype(np.float32) / q_norm
        sims = self.slots @ q_unit
        sims_exp = np.exp(np.clip(sims * 3.0, -10, 10))
        weights = sims_exp / (np.sum(sims_exp) + 1e-8)
        self._read_weights = weights

        result = weights @ self.slots
        return result * self.gate_strength

    def write(self, slot_query: np.ndarray, content: np.ndarray,
              write_gate: float):
        """Gated write: find best-matching slot and blend in new content."""
        if self.n_slots == 0:
            return
        effective_gate = write_gate * self.gate_strength
        if effective_gate < 0.1:
            return

        q = slot_query[:self.bottleneck_size].astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            target_slot = self._write_count % self.n_slots
        else:
            sims = self.slots @ (q / q_norm)
            target_slot = int(np.argmax(sims))

        c = content[:self.bottleneck_size].astype(np.float32)
        self.slots[target_slot] = (
            (1 - effective_gate) * self.slots[target_slot]
            + effective_gate * np.tanh(c)
        )
        self._write_count += 1

    def get_context_vector(self, query: np.ndarray, max_dim: int = 6) -> np.ndarray:
        """Return fixed-size context from workspace read for brain input."""
        result = np.zeros(max_dim, dtype=np.float64)
        if self.n_slots == 0:
            return result
        read_vec = self.read(query)
        n = min(max_dim, len(read_vec))
        result[:n] = read_vec[:n]
        return result

    def clear(self):
        self.slots[:] = 0.0
        self._write_count = 0

    @property
    def stats(self) -> dict:
        if self.n_slots > 0:
            slot_norms = [float(np.linalg.norm(self.slots[i]))
                          for i in range(self.n_slots)]
        else:
            slot_norms = []
        return {
            "n_slots": self.n_slots,
            "gate_strength": float(self.gate_strength),
            "write_count": self._write_count,
            "active_slots": sum(1 for n in slot_norms if n > 0.1),
            "mean_slot_norm": float(np.mean(slot_norms)) if slot_norms else 0.0,
        }
