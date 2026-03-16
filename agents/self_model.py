
import numpy as np
from neural.network import Network


class SelfModel:

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.network = Network([
            {"type": "dense", "in": input_dim, "out": hidden_dim, "act": "tanh"},
            {"type": "dense", "in": hidden_dim, "out": output_dim, "act": "linear"},
        ])
        self._last_prediction = np.zeros(output_dim)
        self._last_actual = np.zeros(output_dim)
        self.prediction_error = 0.0
        self.cumulative_accuracy = 0.0
        self._update_count = 0

    def predict(self, current_state: np.ndarray) -> np.ndarray:
        x = np.zeros(self.input_dim)
        n = min(len(current_state), self.input_dim)
        x[:n] = current_state[:n]
        self._last_prediction = self.network.forward(x)
        return self._last_prediction.copy()

    def observe_actual(self, actual_state: np.ndarray, learning_rate: float = 0.01):
        actual = np.zeros(self.output_dim)
        n = min(len(actual_state), self.output_dim)
        actual[:n] = actual_state[:n]
        self._last_actual = actual

        error = self._last_prediction - actual
        self.prediction_error = float(np.mean(error ** 2))

        self._update_count += 1
        alpha = min(0.01, 1.0 / self._update_count)
        self.cumulative_accuracy = (
            (1 - alpha) * self.cumulative_accuracy + alpha * (1.0 - min(1.0, self.prediction_error))
        )

        params = self.network.get_params()
        grad = np.zeros_like(params)
        out_layer = self.network.layers[-1]
        if hasattr(out_layer, '_last_input') and out_layer._last_input is not None:
            d_out = 2 * error
            grad_W = np.outer(out_layer._last_input, d_out).ravel()
            grad_b = d_out

            n_out_params = out_layer.param_count
            out_params = out_layer.get_params()
            out_grad = np.concatenate([grad_W, grad_b])
            out_grad = np.clip(out_grad, -1.0, 1.0)
            out_params -= learning_rate * out_grad
            out_layer.set_params(out_params)

    @property
    def surprise(self) -> float:
        return float(np.clip(self.prediction_error * 10, 0, 1))

    @property
    def param_count(self) -> int:
        return self.network.param_count

    def get_params(self) -> np.ndarray:
        return self.network.get_params()

    def set_params(self, flat: np.ndarray):
        self.network.set_params(flat)
