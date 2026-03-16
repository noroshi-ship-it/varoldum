
import numpy as np


class ShortTermMemory:

    def __init__(self, capacity: int, entry_dim: int):
        self.capacity = capacity
        self.buffer = np.zeros((capacity, entry_dim))
        self.index = 0
        self.count = 0

    def store(self, entry: np.ndarray):
        trimmed = np.zeros(self.buffer.shape[1])
        n = min(len(entry), len(trimmed))
        trimmed[:n] = entry[:n]
        self.buffer[self.index % self.capacity] = trimmed
        self.index += 1
        self.count = min(self.count + 1, self.capacity)

    def summary(self) -> np.ndarray:
        if self.count == 0:
            return np.zeros(self.buffer.shape[1])
        active = self.buffer[:self.count]
        weights = np.arange(1, self.count + 1, dtype=np.float64)
        weights /= weights.sum()
        return (active.T @ weights)


class LongTermMemory:

    def __init__(self, capacity: int, key_dim: int, value_dim: int):
        self.capacity = capacity
        self.keys = np.zeros((capacity, key_dim))
        self.values = np.zeros((capacity, value_dim))
        self.access_time = np.zeros(capacity)
        self.count = 0
        self._time = 0

    def store(self, key: np.ndarray, value: np.ndarray):
        self._time += 1

        if self.count < self.capacity:
            idx = self.count
            self.count += 1
        else:
            idx = int(np.argmin(self.access_time[:self.count]))

        k = np.zeros(self.keys.shape[1])
        k[:min(len(key), len(k))] = key[:min(len(key), len(k))]
        v = np.zeros(self.values.shape[1])
        v[:min(len(value), len(v))] = value[:min(len(value), len(v))]

        self.keys[idx] = k
        self.values[idx] = v
        self.access_time[idx] = self._time

    def retrieve(self, query: np.ndarray, top_k: int = 3) -> np.ndarray:
        if self.count == 0:
            return np.zeros(self.values.shape[1])

        q = np.zeros(self.keys.shape[1])
        q[:min(len(query), len(q))] = query[:min(len(query), len(q))]

        active_keys = self.keys[:self.count]
        norms = np.linalg.norm(active_keys, axis=1) * np.linalg.norm(q)
        norms = np.maximum(norms, 1e-8)
        similarities = (active_keys @ q) / norms

        k = min(top_k, self.count)
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_sims = similarities[top_indices]

        self._time += 1
        self.access_time[top_indices] = self._time

        top_sims = top_sims - top_sims.max()
        weights = np.exp(top_sims)
        weights /= weights.sum() + 1e-8

        return (self.values[top_indices].T @ weights)


class MemorySystem:

    def __init__(self, stm_capacity: int, ltm_capacity: int, entry_dim: int, output_dim: int):
        self.stm = ShortTermMemory(stm_capacity, entry_dim)
        self.ltm = LongTermMemory(ltm_capacity, key_dim=entry_dim, value_dim=entry_dim)
        self.output_dim = output_dim
        self._entry_dim = entry_dim

    def store_experience(self, state: np.ndarray, reward: float):
        self.stm.store(state)
        if abs(reward) > 0.05:
            value = np.zeros(self._entry_dim)
            value[0] = reward
            self.ltm.store(state, value)

    def get_summary(self, current_state: np.ndarray) -> np.ndarray:
        stm_summary = self.stm.summary()
        ltm_retrieval = self.ltm.retrieve(current_state)

        combined = np.concatenate([stm_summary, ltm_retrieval])
        output = np.zeros(self.output_dim)
        n = min(len(combined), self.output_dim)
        output[:n] = combined[:n]
        return output
