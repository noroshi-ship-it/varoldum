
import numpy as np
from neural.layers import DenseLayer, GRULayer


MIN_BOTTLENECK = 2
MAX_BOTTLENECK = 128


def decode_bottleneck_architecture(arch_genes: np.ndarray, concept_genes: np.ndarray) -> dict:
    n_layers = int(np.clip(round(arch_genes[0]), 1, 4))
    layer_sizes = []
    for i in range(n_layers):
        size = int(np.clip(round(arch_genes[1 + i]), 8, 1024))  # Phase 16: up from 256
        size = max(8, (size // 4) * 4)
        layer_sizes.append(size)
    gru_size = int(np.clip(round(arch_genes[5]), 4, 768))  # Phase 16: up from 128
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
        self._prev_bottleneck = np.zeros(self.bottleneck_size)
        self._last_action = np.zeros(action_dim)
        self._last_wm_prediction = np.zeros(self.bottleneck_size)
        self.world_prediction_error = 0.0
        self._wm_update_count = 0
        self.cumulative_wm_accuracy = 0.0
        self._pre_encoded = False
        self._thought_depth = 0
        self._thought_value_delta = 0.0
        self._last_context = None

        # Daydream / inspiration system
        self._concept_history = np.zeros((16, self.bottleneck_size), dtype=np.float32)
        self._concept_history_idx = 0
        self._concept_history_count = 0
        self._daydream_value = 0.0      # value of last daydream
        self._daydream_novelty = 0.0    # how novel was last daydream
        self._inspiration_count = 0     # times daydream led to discovery

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
        # Save previous bottleneck before encoding overwrites it
        self._prev_bottleneck = self._last_bottleneck.copy()

        if self._pre_encoded:
            concepts = self._last_bottleneck
            self._pre_encoded = False
        else:
            concepts = self.encode(raw_input)

        ctx = np.zeros(self.context_dim)
        n = min(len(context), self.context_dim)
        ctx[:n] = context[:n]
        self._last_context = ctx.copy()

        policy_input = np.concatenate([concepts, ctx])
        gru_out = self.gru.forward(policy_input)
        self._last_hidden = self.gru.h.copy()

        actions = self.output_layer.forward(gru_out)
        value = self.value_layer.forward(self._last_hidden)
        self._last_value = float(value[0])
        self._last_action = actions.copy()

        return actions, self._last_value

    def daydream(self, rng: np.random.Generator) -> tuple[float, float]:
        """Aimless imagination — explore concept space without sensory input.

        Generates a random concept vector, runs it through the world model
        with random actions, and evaluates novelty + value.
        Returns (daydream_value_delta, novelty_score).

        Most daydreams are garbage. But occasionally one discovers a valuable
        region of concept space the agent hasn't visited — inspiration.
        """
        # Generate random starting point in concept space
        # Mix: 70% random noise + 30% current concepts (grounded fantasy)
        random_concepts = rng.standard_normal(self.bottleneck_size).astype(np.float32) * 0.8
        dream_concepts = np.tanh(0.3 * self._last_bottleneck + 0.7 * random_concepts)

        # Random action — "what if I did something completely different?"
        dream_action = rng.standard_normal(self.action_dim).astype(np.float32) * 0.5
        dream_action = np.clip(dream_action, -1, 1)

        # Imagine what would happen
        imagined_next = self.imagine_step(dream_concepts, dream_action)

        # Chain: imagine 2 steps deep (deeper fantasy)
        dream_action2 = rng.standard_normal(self.action_dim).astype(np.float32) * 0.5
        dream_action2 = np.clip(dream_action2, -1, 1)
        imagined_deep = self.imagine_step(imagined_next, dream_action2)

        # Evaluate: is this imagined state valuable?
        saved_h = self.gru.h.copy()
        dream_ctx = self._last_context if self._last_context is not None else np.zeros(self.context_dim)
        gru_input = np.concatenate([imagined_deep, dream_ctx])
        gru_out = self.gru.forward(gru_input)
        dream_value = float(self.value_layer.forward(gru_out)[0])
        self.gru.h = saved_h  # restore — daydream is sandbox

        # Value delta: is the dream better than reality?
        self._daydream_value = dream_value - self._last_value

        # Novelty: how different is this from anything seen before?
        novelty = 0.0
        if self._concept_history_count > 0:
            n = min(self._concept_history_count, 16)
            diffs = self._concept_history[:n] - imagined_deep[np.newaxis, :]
            min_dist = float(np.min(np.sqrt(np.sum(diffs ** 2, axis=1))))
            novelty = min(1.0, min_dist / max(0.1, self.bottleneck_size * 0.3))
        else:
            novelty = 1.0  # first daydream is always novel
        self._daydream_novelty = novelty

        # Record in concept history
        self._concept_history[self._concept_history_idx] = imagined_deep
        self._concept_history_idx = (self._concept_history_idx + 1) % 16
        self._concept_history_count = min(self._concept_history_count + 1, 16)

        # Inspiration: daydream was both novel AND valuable
        if novelty > 0.5 and self._daydream_value > 0.1:
            self._inspiration_count += 1

        return self._daydream_value, novelty

    def imagine_step(self, concepts: np.ndarray, action: np.ndarray) -> np.ndarray:
        wm_input = np.concatenate([concepts, action])
        h = self.wm_hidden_layer.forward(wm_input)
        predicted_next = self.wm_output_layer.forward(h)
        return np.tanh(predicted_next)

    def think(self, raw_input: np.ndarray, context: np.ndarray,
              depth_gate_threshold: float = 0.0,
              branch_count: int = 1,
              workspace=None) -> tuple[np.ndarray, float, int]:
        """Phase 16: True sequential rollout planning.
        Each branch rolls out a full trajectory — GRU state chains forward,
        step N's output feeds step N+1. Only the first action is returned."""
        actions, value = self.forward(raw_input, context)
        steps_used = 0

        if self.think_steps <= 0:
            return actions, value, steps_used

        best_first_action = actions.copy()
        best_trajectory_value = value
        concepts = self._last_bottleneck.copy()
        branch_count = max(1, int(branch_count))

        saved_hidden = self.gru.h.copy()

        ctx = np.zeros(self.context_dim)
        n = min(len(context), self.context_dim)
        ctx[:n] = context[:n]
        discount = 0.9

        # Compute value gradient w.r.t. action to guide search
        action_grad = np.zeros(self.action_dim)
        if self.think_steps > 0:
            base_value = value
            eps = 0.1
            for d in range(self.action_dim):
                perturbed = actions.copy()
                perturbed[d] += eps
                perturbed = np.clip(perturbed, -1, 1)
                imagined = self.imagine_step(concepts, perturbed)
                gru_tmp = self.gru.forward(np.concatenate([imagined, ctx]))
                v_plus = float(self.value_layer.forward(gru_tmp)[0])
                self.gru.h = saved_hidden.copy()
                action_grad[d] = (v_plus - base_value) / eps
            grad_norm = np.linalg.norm(action_grad) + 1e-8
            action_grad_dir = action_grad / grad_norm

        for _branch in range(branch_count):
            # Each branch: roll out a full trajectory
            rollout_concepts = concepts.copy()
            self.gru.h = saved_hidden.copy()
            trajectory_value = 0.0
            first_action = None
            current_actions = actions.copy()

            for step in range(self.think_steps):
                noise_scale = 0.3 / (1 + step)
                # Mix gradient direction with exploration noise
                candidate = current_actions + noise_scale * (
                    action_grad_dir * 0.5 + np.random.randn(self.action_dim) * 0.5
                )
                candidate = np.clip(candidate, -1, 1)

                if step == 0:
                    first_action = candidate.copy()

                # Predict next state
                imagined_next = self.imagine_step(rollout_concepts, candidate)

                # GRU state carries forward within branch — true sequential planning
                gru_out = self.gru.forward(np.concatenate([imagined_next, ctx]))
                step_value = float(self.value_layer.forward(gru_out)[0])

                # Accumulate discounted value
                trajectory_value += (discount ** step) * step_value

                # Chain: use predicted state + policy output for next step
                rollout_concepts = imagined_next
                current_actions = self.output_layer.forward(gru_out)

                steps_used = max(steps_used, step + 1)

            # Normalize and add confidence bonus
            trajectory_value /= max(1, self.think_steps)
            confidence_bonus = 0.1 * (1.0 - min(1.0, self.world_prediction_error))
            trajectory_value += confidence_bonus

            if trajectory_value > best_trajectory_value:
                best_trajectory_value = trajectory_value
                best_first_action = first_action

            if workspace is not None and workspace.n_slots > 0:
                workspace.write(concepts, rollout_concepts, 0.5)

        # Restore GRU state — thinking was a sandbox
        self.gru.h = saved_hidden
        self._last_hidden = saved_hidden

        self._thought_depth = steps_used
        self._thought_value_delta = best_trajectory_value - value

        return best_first_action, best_trajectory_value, steps_used

    def learn_world_model(self, new_bottleneck: np.ndarray):
        if self._wm_update_count == 0 and np.all(self._last_wm_prediction == 0):
            # First call: use previous bottleneck as input to make initial prediction
            wm_input = np.concatenate([self._prev_bottleneck, self._last_action])
            h = self.wm_hidden_layer.forward(wm_input)
            self._last_wm_prediction = self.wm_output_layer.forward(h)
            self._wm_update_count += 1
            return

        error = self._last_wm_prediction - new_bottleneck
        self.world_prediction_error = float(np.mean(error ** 2))

        self._wm_update_count += 1
        alpha = 0.05 / (1.0 + 0.01 * self._wm_update_count)
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

        # Predict next state from PREVIOUS bottleneck + action (temporal offset fix)
        wm_input = np.concatenate([self._prev_bottleneck, self._last_action])
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
