
import numpy as np
from neural.layers import DenseLayer, GRULayer


class Network:

    def __init__(self, layer_specs: list[dict]):
        self.layers = []
        for spec in layer_specs:
            if spec["type"] == "dense":
                self.layers.append(
                    DenseLayer(spec["in"], spec["out"], spec.get("act", "tanh"))
                )
            elif spec["type"] == "gru":
                self.layers.append(GRULayer(spec["in"], spec["hidden"]))

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    @property
    def param_count(self) -> int:
        return sum(l.param_count for l in self.layers)

    def get_params(self) -> np.ndarray:
        return np.concatenate([l.get_params() for l in self.layers])

    def set_params(self, flat: np.ndarray):
        idx = 0
        for layer in self.layers:
            size = layer.param_count
            layer.set_params(flat[idx:idx + size])
            idx += size

    def get_hidden_state(self) -> np.ndarray:
        states = []
        for layer in self.layers:
            if isinstance(layer, GRULayer):
                states.append(layer.h.copy())
        if states:
            return np.concatenate(states)
        return np.zeros(1)

    def reset_state(self):
        for layer in self.layers:
            if isinstance(layer, GRULayer):
                layer.reset_state()
