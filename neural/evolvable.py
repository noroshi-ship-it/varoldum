
import numpy as np
from neural.layers import DenseLayer, GRULayer


ARCH_GENE_COUNT = 6
ARCH_GENE_OFFSET = 0

MIN_LAYER_SIZE = 8
MAX_LAYER_SIZE = 128
MIN_GRU_SIZE = 4
MAX_GRU_SIZE = 64
MIN_LAYERS = 1
MAX_LAYERS = 4


def decode_architecture(arch_genes: np.ndarray) -> dict:
    n_layers = int(np.clip(round(arch_genes[0]), MIN_LAYERS, MAX_LAYERS))
    layer_sizes = []
    for i in range(n_layers):
        size = int(np.clip(round(arch_genes[1 + i]), MIN_LAYER_SIZE, MAX_LAYER_SIZE))
        size = max(MIN_LAYER_SIZE, (size // 4) * 4)
        layer_sizes.append(size)
    gru_size = int(np.clip(round(arch_genes[5]), MIN_GRU_SIZE, MAX_GRU_SIZE))
    gru_size = max(MIN_GRU_SIZE, (gru_size // 4) * 4)
    return {"n_layers": n_layers, "layer_sizes": layer_sizes, "gru_size": gru_size}


class EvolvableBrain:

    def __init__(self, input_dim: int, action_dim: int, arch_genes: np.ndarray):
        self.input_dim = input_dim
        self.action_dim = action_dim
        arch = decode_architecture(arch_genes)
        self.arch = arch

        self.layers: list[DenseLayer | GRULayer] = []
        prev_dim = input_dim

        for size in arch["layer_sizes"]:
            self.layers.append(DenseLayer(prev_dim, size, "tanh"))
            prev_dim = size

        self.gru_size = arch["gru_size"]
        self.gru = GRULayer(prev_dim, self.gru_size)
        self.layers.append(self.gru)
        prev_dim = self.gru_size

        self.output_layer = DenseLayer(prev_dim, action_dim, "tanh")
        self.layers.append(self.output_layer)

        self.value_layer = DenseLayer(self.gru_size, 1, "linear")

        self._last_hidden = np.zeros(self.gru_size)
        self._last_value = 0.0

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, float]:
        inp = np.zeros(self.input_dim)
        n = min(len(x), self.input_dim)
        inp[:n] = x[:n]

        h = inp
        for layer in self.layers:
            h = layer.forward(h)

        self._last_hidden = self.gru.h.copy()
        value = self.value_layer.forward(self._last_hidden)
        self._last_value = float(value[0])

        return h, self._last_value

    @property
    def hidden_state(self) -> np.ndarray:
        return self._last_hidden.copy()

    @property
    def param_count(self) -> int:
        total = sum(l.param_count for l in self.layers)
        total += self.value_layer.param_count
        return total

    @property
    def policy_param_count(self) -> int:
        return sum(l.param_count for l in self.layers)

    def get_policy_params(self) -> np.ndarray:
        parts = [l.get_params() for l in self.layers]
        return np.concatenate(parts)

    def set_policy_params(self, flat: np.ndarray):
        idx = 0
        for layer in self.layers:
            size = layer.param_count
            if idx + size <= len(flat):
                layer.set_params(flat[idx:idx + size])
            idx += size

    def get_value_params(self) -> np.ndarray:
        return self.value_layer.get_params()

    def set_value_params(self, flat: np.ndarray):
        self.value_layer.set_params(flat)

    def reset_state(self):
        self.gru.reset_state()

    def description(self) -> str:
        sizes = [str(s) for s in self.arch["layer_sizes"]]
        return f"Dense[{','.join(sizes)}]->GRU[{self.gru_size}]->Out[{self.action_dim}] ({self.param_count} params)"
