
import numpy as np


class KeyValueMemory:
    """Addressable external memory bank.

    Keys and values are bottleneck-sized vectors.
    Read: soft attention over keys using a query vector.
    Write: gated update to the slot with highest key similarity, or LRU eviction.
    Capacity is gene-controlled.
    """

    def __init__(self, capacity: int, key_dim: int, value_dim: int,
                 read_strength: float, write_strength: float):
        self.capacity = max(2, min(32, int(capacity)))
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.read_strength = float(np.clip(read_strength, 0.1, 2.0))
        self.write_strength = float(np.clip(write_strength, 0.1, 1.0))

        self.keys = np.zeros((self.capacity, key_dim), dtype=np.float32)
        self.values = np.zeros((self.capacity, value_dim), dtype=np.float32)
        self.usage = np.zeros(self.capacity, dtype=np.float32)
        self._last_read_weights = np.zeros(self.capacity, dtype=np.float32)

    def read(self, query: np.ndarray) -> np.ndarray:
        """Attention-weighted read from memory."""
        q = np.zeros(self.key_dim, dtype=np.float32)
        n = min(len(query), self.key_dim)
        q[:n] = query[:n]
        q_norm = np.linalg.norm(q)
        if q_norm < 1e-8:
            return np.zeros(self.value_dim, dtype=np.float32)
        q_unit = q / q_norm

        key_norms = np.linalg.norm(self.keys, axis=1, keepdims=True)
        key_norms = np.maximum(key_norms, 1e-8)
        sims = (self.keys / key_norms) @ q_unit

        scaled = sims * self.read_strength * 3.0
        scaled -= np.max(scaled)
        weights = np.exp(np.clip(scaled, -10, 10))
        weights /= (np.sum(weights) + 1e-8)
        self._last_read_weights = weights

        result = weights @ self.values

        self.usage *= 0.99
        self.usage += weights

        return result

    def write(self, key: np.ndarray, value: np.ndarray):
        """Write to memory: find best-matching slot or LRU slot."""
        k = np.zeros(self.key_dim, dtype=np.float32)
        nk = min(len(key), self.key_dim)
        k[:nk] = key[:nk]

        v = np.zeros(self.value_dim, dtype=np.float32)
        nv = min(len(value), self.value_dim)
        v[:nv] = value[:nv]

        k_norm = np.linalg.norm(k)
        if k_norm < 1e-8:
            target = int(np.argmin(self.usage))
        else:
            k_unit = k / k_norm
            key_norms = np.linalg.norm(self.keys, axis=1, keepdims=True)
            key_norms = np.maximum(key_norms, 1e-8)
            sims = (self.keys / key_norms) @ k_unit
            max_sim = float(np.max(sims))

            if max_sim > 0.7:
                target = int(np.argmax(sims))
            else:
                target = int(np.argmin(self.usage))

        gate = self.write_strength
        self.keys[target] = (1 - gate) * self.keys[target] + gate * np.tanh(k)
        self.values[target] = (1 - gate) * self.values[target] + gate * np.tanh(v)
        self.usage[target] = 1.0

    def get_context_vector(self, query: np.ndarray, max_dim: int = 8) -> np.ndarray:
        """Fixed-size context vector for brain input."""
        result = np.zeros(max_dim, dtype=np.float64)
        read_val = self.read(query)
        n = min(max_dim, len(read_val))
        result[:n] = read_val[:n]
        return result

    @property
    def occupancy(self) -> float:
        return float(np.mean(self.usage > 0.01))
