
import numpy as np
from neural.activations import sigmoid, tanh


class DenseLayer:

    def __init__(self, in_dim: int, out_dim: int, activation: str = "tanh"):
        scale = np.sqrt(2.0 / (in_dim + out_dim))
        self.W = np.random.randn(in_dim, out_dim) * scale
        self.b = np.zeros(out_dim)
        self.activation = activation
        self._last_input = None
        self._last_pre_act = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._last_input = x
        z = x @ self.W + self.b
        self._last_pre_act = z
        if self.activation == "tanh":
            return np.tanh(z)
        elif self.activation == "sigmoid":
            return sigmoid(z)
        elif self.activation == "relu":
            return np.maximum(0, z)
        elif self.activation == "linear":
            return z
        return np.tanh(z)

    @property
    def param_count(self) -> int:
        return self.W.size + self.b.size

    def get_params(self) -> np.ndarray:
        return np.concatenate([self.W.ravel(), self.b.ravel()])

    def set_params(self, flat: np.ndarray):
        w_size = self.W.size
        self.W = flat[:w_size].reshape(self.W.shape)
        self.b = flat[w_size:w_size + self.b.size]


class GRULayer:

    def __init__(self, in_dim: int, hidden_dim: int):
        self.hidden_dim = hidden_dim
        scale = np.sqrt(2.0 / (in_dim + hidden_dim))

        self.W_r = np.random.randn(in_dim + hidden_dim, hidden_dim) * scale
        self.b_r = np.zeros(hidden_dim)
        self.W_z = np.random.randn(in_dim + hidden_dim, hidden_dim) * scale
        self.b_z = np.zeros(hidden_dim)
        self.W_h = np.random.randn(in_dim + hidden_dim, hidden_dim) * scale
        self.b_h = np.zeros(hidden_dim)

        self.h = np.zeros(hidden_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        combined = np.concatenate([x, self.h])
        r = sigmoid(combined @ self.W_r + self.b_r)
        z = sigmoid(combined @ self.W_z + self.b_z)
        combined_r = np.concatenate([x, r * self.h])
        h_candidate = np.tanh(combined_r @ self.W_h + self.b_h)
        self.h = (1 - z) * self.h + z * h_candidate
        return self.h.copy()

    def reset_state(self):
        self.h = np.zeros(self.hidden_dim)

    @property
    def param_count(self) -> int:
        return (self.W_r.size + self.b_r.size +
                self.W_z.size + self.b_z.size +
                self.W_h.size + self.b_h.size)

    def get_params(self) -> np.ndarray:
        return np.concatenate([
            self.W_r.ravel(), self.b_r.ravel(),
            self.W_z.ravel(), self.b_z.ravel(),
            self.W_h.ravel(), self.b_h.ravel(),
        ])

    def set_params(self, flat: np.ndarray):
        idx = 0
        for attr in ['W_r', 'b_r', 'W_z', 'b_z', 'W_h', 'b_h']:
            arr = getattr(self, attr)
            size = arr.size
            setattr(self, attr, flat[idx:idx + size].reshape(arr.shape))
            idx += size
