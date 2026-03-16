
import numpy as np


class SymbolCodebook:
    """Vector Quantization codebook that discretizes continuous bottleneck concepts
    into composable symbols. Codebook learns via EMA updates."""

    __slots__ = ['n_symbols', 'n_slots', 'lr', 'codebook', 'usage_count',
                 'active_symbols', 'symbol_strengths', '_bottleneck_size']

    def __init__(self, bottleneck_size: int, n_symbols: int = 16,
                 n_slots: int = 2, lr: float = 0.01):
        self.n_symbols = max(4, min(32, n_symbols))
        self.n_slots = max(1, min(4, n_slots))
        self.lr = lr
        self._bottleneck_size = bottleneck_size

        # Codebook: each row is a prototype vector
        self.codebook = np.random.randn(self.n_symbols, bottleneck_size).astype(np.float32) * 0.5
        # Normalize rows
        norms = np.linalg.norm(self.codebook, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        self.codebook /= norms

        self.usage_count = np.zeros(self.n_symbols, dtype=np.int32)
        self.active_symbols = np.zeros(self.n_slots, dtype=np.int32)
        self.symbol_strengths = np.zeros(self.n_slots, dtype=np.float32)

    def quantize(self, concepts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Quantize continuous concept vector into discrete symbols.
        Returns (active_symbol_ids, strengths)."""
        c = concepts[:self._bottleneck_size].astype(np.float32)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-8:
            self.active_symbols[:] = 0
            self.symbol_strengths[:] = 0.0
            return self.active_symbols.copy(), self.symbol_strengths.copy()

        c_unit = c / c_norm

        # Cosine similarity with all codebook entries
        sims = self.codebook @ c_unit
        # Top-k selection (soft winner-take-all)
        top_k = min(self.n_slots, self.n_symbols)
        top_indices = np.argpartition(sims, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(sims[top_indices])[::-1]]

        self.active_symbols[:] = 0
        self.symbol_strengths[:] = 0.0
        for i in range(top_k):
            self.active_symbols[i] = top_indices[i]
            self.symbol_strengths[i] = max(0.0, sims[top_indices[i]])
            self.usage_count[top_indices[i]] += 1

        return self.active_symbols.copy(), self.symbol_strengths.copy()

    def learn(self, concepts: np.ndarray):
        """EMA update: winning codebook entries drift toward the input."""
        c = concepts[:self._bottleneck_size].astype(np.float32)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-8:
            return

        c_unit = c / c_norm
        for i in range(self.n_slots):
            if self.symbol_strengths[i] < 0.01:
                continue
            idx = self.active_symbols[i]
            # EMA: codebook[idx] = (1-lr) * codebook[idx] + lr * c_unit
            weight = self.lr * self.symbol_strengths[i]
            self.codebook[idx] = (1.0 - weight) * self.codebook[idx] + weight * c_unit
            # Re-normalize
            norm = np.linalg.norm(self.codebook[idx])
            if norm > 1e-8:
                self.codebook[idx] /= norm

    def get_symbol_embedding(self, symbol_id: int) -> np.ndarray:
        """Get the meaning vector for a symbol."""
        if 0 <= symbol_id < self.n_symbols:
            return self.codebook[symbol_id].copy()
        return np.zeros(self._bottleneck_size, dtype=np.float32)

    def get_composite_embedding(self) -> np.ndarray:
        """Get weighted sum of active symbol embeddings."""
        result = np.zeros(self._bottleneck_size, dtype=np.float32)
        total_w = 0.0
        for i in range(self.n_slots):
            if self.symbol_strengths[i] > 0.01:
                result += self.codebook[self.active_symbols[i]] * self.symbol_strengths[i]
                total_w += self.symbol_strengths[i]
        if total_w > 1e-8:
            result /= total_w
        return result

    def has_symbol(self, symbol_id: int) -> bool:
        """Check if a symbol is currently active."""
        for i in range(self.n_slots):
            if self.active_symbols[i] == symbol_id and self.symbol_strengths[i] > 0.1:
                return True
        return False

    @property
    def stats(self) -> dict:
        active_count = int(np.sum(self.usage_count > 0))
        total_usage = int(np.sum(self.usage_count))
        entropy = 0.0
        if total_usage > 0:
            probs = self.usage_count / max(1, total_usage)
            probs = probs[probs > 0]
            entropy = -float(np.sum(probs * np.log2(probs + 1e-10)))
        return {
            "active_symbols": active_count,
            "total_usage": total_usage,
            "symbol_entropy": entropy,
            "mean_strength": float(np.mean(self.symbol_strengths)),
        }
