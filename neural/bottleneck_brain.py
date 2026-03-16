
import numpy as np
from neural.layers import DenseLayer, GRULayer


MIN_BOTTLENECK = 2
MAX_BOTTLENECK = 16


def decode_bottleneck_architecture(arch_genes: np.ndarray, concept_genes: np.ndarray) -> dict:
    n_layers = int(np.clip(round(arch_genes[0]), 1, 4))
    layer_sizes = []
    for i in range(n_layers):
        size = int(np.clip(round(arch_genes[1 + i]), 8, 128))
        size = max(8, (size // 4) * 4)
        layer_sizes.append(size)
    gru_size = int(np.clip(round(arch_genes[5]), 4, 64))
    gru_size = max(4, (gru_size // 4) * 4)

    bottleneck_size = int(np.clip(round(concept_genes[0]), MIN_BOTTLENECK, MAX_BOTTLENECK))
    think_steps = int(np.clip(round(concept_genes[1]), 0, 8))
    world_model_lr = float(np.clip(concept_genes[2], 0.001, 0.05))

    return {
        "n_layers": n_layers,
        "layer_sizes": layer_sizes,
        "gru_size": gru_size,
        "bottleneck_size": bottleneck_size,
        "think_steps": think_steps,
        "world_model_lr": world_model_lr,
    }


class BottleneckBrain:

    def __init__(self, raw_input_dim: int, context_dim: int, action_dim: int,
                 arch_genes: np.ndarray, concept_genes: np.ndarray):
        self.raw_input_dim = raw_input_dim
        self.context_dim = context_dim
        self.action_dim = action_dim

        arch = decode_bottleneck_architecture(arch_genes, concept_genes)
        self.arch = arch
        self.bottleneck_size = arch["bottleneck_size"]
        self.think_steps = arch["think_steps"]
        self.world_model_lr = arch["world_model_lr"]

        self.encoder_layers: list[DenseLayer] = []
        prev_dim = raw_input_dim
        for size in arch["layer_sizes"]:
            self.encoder_layers.append(DenseLayer(prev_dim, size, "tanh"))
            prev_dim = size
        self.bottleneck_layer = DenseLayer(prev_dim, self.bottleneck_size, "tanh")

        policy_input_dim = self.bottleneck_size + context_dim
        self.gru_size = arch["gru_size"]
        self.gru = GRULayer(policy_input_dim, self.gru_size)

        self.output_layer = DenseLayer(self.gru_size, action_dim, "tanh")

        self.value_layer = DenseLayer(self.gru_size, 1, "linear")

        wm_input = self.bottleneck_size + action_dim
        wm_hidden = max(8, self.bottleneck_size * 2)
        self.wm_hidden_layer = DenseLayer(wm_input, wm_hidden, "tanh")
        self.wm_output_layer = DenseLayer(wm_hidden, self.bottleneck_size, "linear")

        self._last_hidden = np.zeros(self.gru_size)
        self._last_value = 0.0
        self._last_bottleneck = np.zeros(self.bottleneck_size)
        self._last_action = np.zeros(action_dim)
        self._last_wm_prediction = np.zeros(self.bottleneck_size)
        self.world_prediction_error = 0.0
        self._wm_update_count = 0
        self.cumulative_wm_accuracy = 0.0
        self._pre_encoded = False
        self._thought_depth = 0
        self._thought_value_delta = 0.0

    def get_concepts(self):
        return self._last_bottleneck

    def set_pre_encoded(self, concepts):
        self._last_bottleneck = concepts
        self._pre_encoded = True

    def encode(self, raw_input: np.ndarray) -> np.ndarray:
        h = np.zeros(self.raw_input_dim)
        n = min(len(raw_input), self.raw_input_dim)
        h[:n] = raw_input[:n]

        for layer in self.encoder_layers:
            h = layer.forward(h)
        self._last_bottleneck = self.bottleneck_layer.forward(h)
        return self._last_bottleneck.copy()

    def forward(self, raw_input: np.ndarray, context: np.ndarray) -> tuple[np.ndarray, float]:
        if self._pre_encoded:
            concepts = self._last_bottleneck
            self._pre_encoded = False
        else:
            concepts = self.encode(raw_input)

        ctx = np.zeros(self.context_dim)
        n = min(len(context), self.context_dim)
        ctx[:n] = context[:n]

        policy_input = np.concatenate([concepts, ctx])
        gru_out = self.gru.forward(policy_input)
        self._last_hidden = self.gru.h.copy()

        actions = self.output_layer.forward(gru_out)
        value = self.value_layer.forward(self._last_hidden)
        self._last_value = float(value[0])
        self._last_action = actions.copy()

        return actions, self._last_value

    def imagine_step(self, concepts: np.ndarray, action: np.ndarray) -> np.ndarray:
        wm_input = np.concatenate([concepts, action])
        h = self.wm_hidden_layer.forward(wm_input)
        predicted_next = self.wm_output_layer.forward(h)
        return np.tanh(predicted_next)

    def think(self, raw_input: np.ndarray, context: np.ndarray,
              depth_gate_threshold: float = 0.0) -> tuple[np.ndarray, float, int]:
        actions, value = self.forward(raw_input, context)
        steps_used = 0

        if self.think_steps <= 0:
            return actions, value, steps_used

        best_action = actions.copy()
        best_value = value
        concepts = self._last_bottleneck.copy()

        # Save GRU state — thinking happens in a sandbox
        saved_hidden = self.gru.h.copy()

        ctx = np.zeros(self.context_dim)
        n = min(len(context), self.context_dim)
        ctx[:n] = context[:n]

        for step in range(self.think_steps):
            noise_scale = 0.3 / (1 + step)
            candidate = best_action + np.random.randn(self.action_dim) * noise_scale
            candidate = np.clip(candidate, -1, 1)

            # Imagine next concepts from current concepts + candidate action
            imagined_next = self.imagine_step(concepts, candidate)

            # RECURSIVE: pass imagined concepts through GRU (hidden updates)
            imagined_input = np.concatenate([imagined_next, ctx])
            gru_out = self.gru.forward(imagined_input)

            # Evaluate value from the recursively updated hidden state
            imagined_value = float(self.value_layer.forward(gru_out)[0])

            confidence_bonus = 0.1 * (1.0 - min(1.0, self.world_prediction_error))
            imagined_value += confidence_bonus

            improvement = imagined_value - best_value
            if imagined_value > best_value:
                best_value = imagined_value
                best_action = candidate
                # Use imagined concepts for next depth level
                concepts = imagined_next
            steps_used += 1

            # Depth gate: stop if improvement is below threshold and not first step
            if step > 0 and improvement < depth_gate_threshold:
                break

        # Restore GRU state — thinking was a sandbox
        self.gru.h = saved_hidden
        self._last_hidden = saved_hidden

        # Store thought metadata for self-model
        self._thought_depth = steps_used
        self._thought_value_delta = best_value - value

        return best_action, best_value, steps_used

    def learn_world_model(self, new_bottleneck: np.ndarray):
        if self._wm_update_count == 0 and np.all(self._last_wm_prediction == 0):
            wm_input = np.concatenate([self._last_bottleneck, self._last_action])
            h = self.wm_hidden_layer.forward(wm_input)
            self._last_wm_prediction = self.wm_output_layer.forward(h)
            self._wm_update_count += 1
            return

        error = self._last_wm_prediction - new_bottleneck
        self.world_prediction_error = float(np.mean(error ** 2))

        self._wm_update_count += 1
        alpha = min(0.01, 1.0 / self._wm_update_count)
        self.cumulative_wm_accuracy = (
            (1 - alpha) * self.cumulative_wm_accuracy
            + alpha * (1.0 - min(1.0, self.world_prediction_error))
        )

        lr = self.world_model_lr
        out_layer = self.wm_output_layer
        if hasattr(out_layer, '_last_input') and out_layer._last_input is not None:
            d_out = 2 * error
            grad_W = np.outer(out_layer._last_input, d_out).ravel()
            grad_b = d_out
            out_params = out_layer.get_params()
            out_grad = np.concatenate([grad_W, grad_b])
            out_grad = np.clip(out_grad, -1.0, 1.0)
            out_params -= lr * out_grad
            out_layer.set_params(out_params)

        hid_layer = self.wm_hidden_layer
        if hasattr(hid_layer, '_last_input') and hid_layer._last_input is not None:
            d_hidden = (d_out @ out_layer.W.T)
            if hid_layer._last_pre_act is not None:
                d_hidden *= (1.0 - np.tanh(hid_layer._last_pre_act) ** 2)
            grad_W_h = np.outer(hid_layer._last_input, d_hidden).ravel()
            grad_b_h = d_hidden
            hid_params = hid_layer.get_params()
            hid_grad = np.concatenate([grad_W_h, grad_b_h])
            hid_grad = np.clip(hid_grad, -1.0, 1.0)
            hid_params -= lr * 0.5 * hid_grad
            hid_layer.set_params(hid_params)

        wm_input = np.concatenate([new_bottleneck, self._last_action])
        h = self.wm_hidden_layer.forward(wm_input)
        self._last_wm_prediction = self.wm_output_layer.forward(h)

    @property
    def concepts(self) -> np.ndarray:
        return self._last_bottleneck.copy()

    @property
    def hidden_state(self) -> np.ndarray:
        return self._last_hidden.copy()

    @property
    def param_count(self) -> int:
        total = sum(l.param_count for l in self.encoder_layers)
        total += self.bottleneck_layer.param_count
        total += self.gru.param_count
        total += self.output_layer.param_count
        total += self.value_layer.param_count
        total += self.wm_hidden_layer.param_count
        total += self.wm_output_layer.param_count
        return total

    @property
    def policy_param_count(self) -> int:
        total = sum(l.param_count for l in self.encoder_layers)
        total += self.bottleneck_layer.param_count
        total += self.gru.param_count
        total += self.output_layer.param_count
        return total

    def get_policy_params(self) -> np.ndarray:
        parts = []
        for l in self.encoder_layers:
            parts.append(l.get_params())
        parts.append(self.bottleneck_layer.get_params())
        parts.append(self.gru.get_params())
        parts.append(self.output_layer.get_params())
        return np.concatenate(parts)

    def set_policy_params(self, flat: np.ndarray):
        idx = 0
        for layer in self.encoder_layers:
            size = layer.param_count
            if idx + size <= len(flat):
                layer.set_params(flat[idx:idx + size])
            idx += size
        size = self.bottleneck_layer.param_count
        if idx + size <= len(flat):
            self.bottleneck_layer.set_params(flat[idx:idx + size])
        idx += size
        size = self.gru.param_count
        if idx + size <= len(flat):
            self.gru.set_params(flat[idx:idx + size])
        idx += size
        size = self.output_layer.param_count
        if idx + size <= len(flat):
            self.output_layer.set_params(flat[idx:idx + size])

    def get_value_params(self) -> np.ndarray:
        return self.value_layer.get_params()

    def set_value_params(self, flat: np.ndarray):
        self.value_layer.set_params(flat)

    def reset_state(self):
        self.gru.reset_state()

    def description(self) -> str:
        enc_sizes = [str(s) for s in self.arch["layer_sizes"]]
        return (
            f"Enc[{','.join(enc_sizes)}]->BN[{self.bottleneck_size}]"
            f"->GRU[{self.gru_size}]->Out[{self.action_dim}] "
            f"think={self.think_steps} wm_acc={self.cumulative_wm_accuracy:.2f} "
            f"({self.param_count} params)"
        )
