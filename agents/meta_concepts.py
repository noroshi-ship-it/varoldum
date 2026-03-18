
import numpy as np
from neural.layers import DenseLayer


MAX_META_WINDOW = 16
MAX_META_BOTTLENECK = 16


class MetaConceptSystem:
    """Phase 6: Meta-concept layer — concepts about concepts.

    Compresses a sliding window of bottleneck vectors into a smaller
    meta-concept space, enabling agents to observe patterns in their
    own thinking (e.g. oscillation, trends, stability).
    """

    __slots__ = [
        'meta_window', 'meta_bottleneck_size', 'meta_wm_lr', 'action_dim',
        'bottleneck_size', '_buffer', '_buf_idx', '_buf_count',
        '_encoder', '_wm_hidden', '_wm_output',
        '_last_meta', '_prev_meta', '_last_predicted',
        '_meta_prediction_error', '_cumulative_accuracy', '_update_count',
    ]

    def __init__(self, bottleneck_size: int, meta_window: int,
                 meta_bottleneck_size: int, meta_wm_lr: float, action_dim: int):
        self.bottleneck_size = bottleneck_size
        self.meta_window = max(4, min(MAX_META_WINDOW, meta_window))
        self.meta_bottleneck_size = max(2, min(MAX_META_BOTTLENECK, meta_bottleneck_size))
        self.meta_wm_lr = np.clip(meta_wm_lr, 0.001, 0.05)
        self.action_dim = action_dim

        # Circular buffer for recent bottleneck vectors
        self._buffer = np.zeros((self.meta_window, bottleneck_size), dtype=np.float32)
        self._buf_idx = 0
        self._buf_count = 0

        # Meta-encoder: flattened window → meta-concepts
        input_dim = self.meta_window * bottleneck_size
        self._encoder = DenseLayer(input_dim, self.meta_bottleneck_size, "tanh")

        # Meta world model: [meta_concepts + action] → predicted next meta
        wm_input = self.meta_bottleneck_size + action_dim
        wm_hidden_size = max(8, self.meta_bottleneck_size * 2)
        self._wm_hidden = DenseLayer(wm_input, wm_hidden_size, "tanh")
        self._wm_output = DenseLayer(wm_hidden_size, self.meta_bottleneck_size, "linear")

        # State
        self._last_meta = np.zeros(self.meta_bottleneck_size, dtype=np.float32)
        self._prev_meta = np.zeros(self.meta_bottleneck_size, dtype=np.float32)
        self._last_predicted = np.zeros(self.meta_bottleneck_size, dtype=np.float32)
        self._meta_prediction_error = 0.0
        self._cumulative_accuracy = 0.0
        self._update_count = 0

    def record_concepts(self, concepts: np.ndarray):
        """Push current bottleneck vector into circular buffer."""
        n = min(len(concepts), self.bottleneck_size)
        self._buffer[self._buf_idx, :n] = concepts[:n]
        self._buf_idx = (self._buf_idx + 1) % self.meta_window
        self._buf_count = min(self._buf_count + 1, self.meta_window)

    def encode_meta(self) -> np.ndarray:
        """Compress concept trajectory into meta-concepts."""
        if self._buf_count < 2:
            return self._last_meta

        # Build ordered sequence from circular buffer
        if self._buf_count >= self.meta_window:
            # Full buffer — reorder oldest-first
            ordered = np.roll(self._buffer, -self._buf_idx, axis=0)
        else:
            ordered = self._buffer[:self._buf_count]
            # Pad to full window
            pad = np.zeros((self.meta_window - self._buf_count, self.bottleneck_size),
                           dtype=np.float32)
            ordered = np.concatenate([pad, ordered], axis=0)

        flat = ordered.flatten().astype(np.float32)
        self._prev_meta = self._last_meta.copy()
        self._last_meta = self._encoder.forward(flat)
        return self._last_meta

    def predict_next_meta(self, action: np.ndarray) -> np.ndarray:
        """Predict next meta-concept state given current action."""
        act = np.zeros(self.action_dim, dtype=np.float32)
        n = min(len(action), self.action_dim)
        act[:n] = action[:n]

        wm_input = np.concatenate([self._last_meta, act])
        hidden = self._wm_hidden.forward(wm_input)
        predicted = np.tanh(self._wm_output.forward(hidden))
        self._last_predicted = predicted
        return predicted

    def learn_meta_wm(self, actual_meta: np.ndarray, action: np.ndarray = None):
        """Train meta world model on prediction error (online backprop)."""
        if self._update_count == 0:
            self._last_predicted = actual_meta.copy()
            self._update_count += 1
            return

        error = self._last_predicted - actual_meta[:self.meta_bottleneck_size]
        mse = float(np.mean(error ** 2))
        self._meta_prediction_error = mse

        # EMA accuracy
        self._update_count += 1
        alpha = min(0.01, 1.0 / self._update_count)
        acc = max(0.0, 1.0 - mse)
        self._cumulative_accuracy = (1 - alpha) * self._cumulative_accuracy + alpha * acc

        lr = self.meta_wm_lr

        # Backprop through output layer — use actual action if provided
        act = np.zeros(self.action_dim, dtype=np.float32)
        if action is not None:
            n = min(len(action), self.action_dim)
            act[:n] = action[:n]
        wm_input = np.concatenate([self._prev_meta, act])
        hidden_out = self._wm_hidden.forward(wm_input)

        d_out = 2.0 * error / max(1, len(error))
        # Gradient for output layer
        grad_W_out = np.outer(hidden_out, d_out)
        grad_b_out = d_out

        # Backprop through hidden
        d_hidden = d_out @ self._wm_output.W.T
        tanh_deriv = 1.0 - hidden_out ** 2
        d_hidden = d_hidden * tanh_deriv

        grad_W_hid = np.outer(wm_input, d_hidden)
        grad_b_hid = d_hidden

        # Clip and apply
        np.clip(grad_W_out, -1.0, 1.0, out=grad_W_out)
        np.clip(grad_b_out, -1.0, 1.0, out=grad_b_out)
        np.clip(grad_W_hid, -1.0, 1.0, out=grad_W_hid)
        np.clip(grad_b_hid, -1.0, 1.0, out=grad_b_hid)

        self._wm_output.W -= lr * grad_W_out
        self._wm_output.b -= lr * grad_b_out
        self._wm_hidden.W -= lr * grad_W_hid
        self._wm_hidden.b -= lr * grad_b_hid

    def get_meta_prediction_error(self) -> float:
        return self._meta_prediction_error

    @property
    def cumulative_accuracy(self) -> float:
        return self._cumulative_accuracy

    @property
    def param_count(self) -> int:
        enc = self._encoder.W.size + self._encoder.b.size
        wm = (self._wm_hidden.W.size + self._wm_hidden.b.size +
              self._wm_output.W.size + self._wm_output.b.size)
        return enc + wm

    def get_meta_summary(self, max_dim: int = 16) -> np.ndarray:
        """Return meta-concepts padded/truncated to fixed size for context."""
        result = np.zeros(max_dim, dtype=np.float32)
        n = min(self.meta_bottleneck_size, max_dim)
        result[:n] = self._last_meta[:n]
        return result

    def introspect(self) -> dict:
        """Return interpretable summary of meta-cognitive state."""
        # Detect temporal pattern type
        if self._buf_count < 4:
            pattern = "insufficient_data"
        else:
            # Compute variance of recent meta-concepts
            meta_var = float(np.var(self._last_meta))
            # Compute trend: difference between current and previous
            trend = float(np.mean(self._last_meta - self._prev_meta))

            if meta_var < 0.05:
                pattern = "stable"
            elif abs(trend) > 0.1:
                pattern = "trending_up" if trend > 0 else "trending_down"
            else:
                pattern = "oscillating"

        return {
            "meta_concepts": self._last_meta.copy(),
            "pattern": pattern,
            "meta_wm_accuracy": self._cumulative_accuracy,
            "meta_prediction_error": self._meta_prediction_error,
            "buffer_fill": self._buf_count / self.meta_window,
            "active_dims": int(np.sum(np.abs(self._last_meta) > 0.3)),
        }
