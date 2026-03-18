
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

        out_layer = self.network.layers[-1]
        if hasattr(out_layer, '_last_input') and out_layer._last_input is not None:
            d_out = 2 * error
            grad_w_out = np.outer(out_layer._last_input, d_out).ravel()
            grad_b_out = d_out

            out_params = out_layer.get_params()
            out_grad = np.concatenate([grad_w_out, grad_b_out])
            out_grad = np.clip(out_grad, -1.0, 1.0)
            out_params -= learning_rate * out_grad
            out_layer.set_params(out_params)

            # Backpropagate through hidden layer
            if len(self.network.layers) >= 2:
                hid_layer = self.network.layers[-2]
                if hasattr(hid_layer, '_last_input') and hid_layer._last_input is not None:
                    d_hidden = d_out @ out_layer.W.T
                    tanh_deriv = 1.0 - out_layer._last_input ** 2
                    d_hidden = d_hidden * tanh_deriv
                    grad_w_hid = np.outer(hid_layer._last_input, d_hidden).ravel()
                    grad_b_hid = d_hidden
                    hid_params = hid_layer.get_params()
                    hid_grad = np.concatenate([grad_w_hid, grad_b_hid])
                    hid_grad = np.clip(hid_grad, -1.0, 1.0)
                    hid_params -= learning_rate * hid_grad
                    hid_layer.set_params(hid_params)

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


class MortalitySelfModel:
    """Extends self-awareness with mortality prediction.
    NOT hardcoded — survival estimates emerge from the agent's own concepts
    and observed deaths. Uses concept-space similarity to process death
    observations rather than fixed multipliers."""

    def __init__(self, bottleneck_size: int, mortality_sensitivity: float = 1.0):
        self.mortality_sensitivity = np.clip(mortality_sensitivity, 0.0, 2.0)
        self._bottleneck_size = bottleneck_size

        # Learned survival predictor: maps [concepts, body_features] -> [survival_prob, time_to_death]
        input_dim = bottleneck_size + 6  # concepts + [energy, health, age_ratio, energy_trend, health_trend, recent_damage]
        self._survival_net = Network([
            {"type": "dense", "in": input_dim, "out": 8, "act": "tanh"},
            {"type": "dense", "in": 8, "out": 2, "act": "linear"},
        ])

        self.survival_prob = 0.5
        self.time_to_death = 1.0  # normalized [0,1] where 1 = far away
        self._death_observations = []  # recent (concept_similarity, tick) pairs
        self._last_prediction = np.zeros(2)
        self._update_count = 0
        self._energy_history = []
        self._health_history = []
        self._recent_damage = 0.0

    def update_trends(self, energy: float, health: float, damage: float):
        """Track energy/health trends for survival prediction input."""
        self._energy_history.append(energy)
        self._health_history.append(health)
        if len(self._energy_history) > 20:
            self._energy_history.pop(0)
        if len(self._health_history) > 20:
            self._health_history.pop(0)
        # EMA for recent damage
        self._recent_damage = 0.9 * self._recent_damage + 0.1 * damage

    def predict_survival(self, concepts: np.ndarray, energy: float, health: float,
                         age_ratio: float) -> tuple[float, float]:
        """Predict survival probability and time-to-death from own concepts.
        The prediction is the agent's own model of its mortality — personal and learned."""
        energy_trend = 0.0
        health_trend = 0.0
        if len(self._energy_history) >= 5:
            energy_trend = self._energy_history[-1] - self._energy_history[-5]
        if len(self._health_history) >= 5:
            health_trend = self._health_history[-1] - self._health_history[-5]

        c = np.zeros(self._bottleneck_size)
        n = min(len(concepts), self._bottleneck_size)
        c[:n] = concepts[:n]

        features = np.concatenate([
            c,
            [energy, health, age_ratio, energy_trend, health_trend, self._recent_damage]
        ])

        raw = self._survival_net.forward(features)
        self._last_prediction = raw.copy()

        # Sigmoid for probability, sigmoid for time
        self.survival_prob = float(1.0 / (1.0 + np.exp(-raw[0])))
        self.time_to_death = float(1.0 / (1.0 + np.exp(-raw[1])))

        return self.survival_prob, self.time_to_death

    def observe_nearby_death(self, my_concepts: np.ndarray, dead_concepts: np.ndarray,
                             tick: int):
        """Observe another agent dying. Effect is NOT hardcoded — it depends on
        how similar the dead agent's conceptual state is to ours, modulated by
        the genome-controlled mortality_sensitivity. This means:
        - An agent who has developed rich concept representations will have
          more nuanced death-awareness
        - The similarity is in CONCEPT space — not raw features — so it reflects
          the agent's own abstract understanding
        - mortality_sensitivity controls how much attention the agent pays to deaths"""
        # Align to minimum dimension — agents can have different bottleneck sizes
        dim = min(self._bottleneck_size, len(my_concepts), len(dead_concepts))
        my_c = my_concepts[:dim]
        dead_c = dead_concepts[:dim]

        my_norm = np.linalg.norm(my_c)
        dead_norm = np.linalg.norm(dead_c)
        if my_norm < 1e-8 or dead_norm < 1e-8:
            return

        # Concept-space similarity: how much do I relate to the dead agent?
        similarity = float(np.dot(my_c, dead_c) / (my_norm * dead_norm))
        similarity = max(0.0, similarity)  # only positive similarity matters

        # Store observation — agent accumulates death experiences
        self._death_observations.append((similarity, tick))
        # Keep only recent observations
        if len(self._death_observations) > 10:
            self._death_observations.pop(0)

    def get_death_awareness(self) -> float:
        """How aware is this agent of death risk from observations?
        This is a learned, personal metric — not a fixed formula."""
        if not self._death_observations:
            return 0.0
        # Weight recent observations more
        total = 0.0
        weight_sum = 0.0
        for i, (sim, _tick) in enumerate(self._death_observations):
            w = 0.5 + 0.5 * (i / max(1, len(self._death_observations) - 1))
            total += sim * w * self.mortality_sensitivity
            weight_sum += w
        if weight_sum < 1e-8:
            return 0.0
        return float(np.clip(total / weight_sum, 0.0, 1.0))

    def learn_from_survival(self, survived: bool, learning_rate: float = 0.01):
        """After each tick: the agent survived. Train survival predictor."""
        self._update_count += 1
        # Target: [survival_prob_target, time_to_death_target]
        if survived:
            target = np.array([1.0, 0.8])  # alive = high survival, moderate time
        else:
            target = np.array([-1.0, -1.0])  # dead = low everything

        error = self._last_prediction - target
        out_layer = self._survival_net.layers[-1]
        if hasattr(out_layer, '_last_input') and out_layer._last_input is not None:
            d_out = 2 * error
            grad_W = np.outer(out_layer._last_input, d_out).ravel()
            grad_b = d_out
            out_params = out_layer.get_params()
            out_grad = np.concatenate([grad_W, grad_b])
            out_grad = np.clip(out_grad, -1.0, 1.0)
            out_params -= learning_rate * out_grad
            out_layer.set_params(out_params)

    @property
    def param_count(self) -> int:
        return self._survival_net.param_count
