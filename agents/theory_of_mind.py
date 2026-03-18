
import numpy as np
from neural.layers import DenseLayer


# Observation vector: [rel_pos(2), action(6), signal_strength(1), decoded_utterance(4)] = 13
OBS_DIM = 13
MAX_TOM_TRACKED = 6
MAX_TOM_WINDOW = 12


class TrackedAgent:
    """State for a single other agent being modeled."""
    __slots__ = [
        'agent_id', 'obs_buffer', 'obs_idx', 'obs_count',
        'predicted_concepts', 'accuracy', 'priority',
        'interaction_count', 'last_seen_tick',
        'predicted_action', 'has_prediction',
    ]

    def __init__(self, agent_id: int, tom_window: int, bottleneck_size: int,
                 action_dim: int = 6):
        self.agent_id = agent_id
        self.obs_buffer = np.zeros((tom_window, OBS_DIM), dtype=np.float32)
        self.obs_idx = 0
        self.obs_count = 0
        self.predicted_concepts = np.zeros(bottleneck_size, dtype=np.float32)
        self.accuracy = 0.0
        self.priority = 0.0
        self.interaction_count = 0
        self.last_seen_tick = 0
        self.predicted_action = np.zeros(action_dim, dtype=np.float32)
        self.has_prediction = False

    def record_observation(self, obs: np.ndarray, tick: int):
        n = min(len(obs), OBS_DIM)
        self.obs_buffer[self.obs_idx, :n] = obs[:n]
        self.obs_idx = (self.obs_idx + 1) % len(self.obs_buffer)
        self.obs_count = min(self.obs_count + 1, len(self.obs_buffer))
        self.interaction_count += 1
        self.last_seen_tick = tick

    def get_flat_observations(self) -> np.ndarray:
        """Return flattened observation history, oldest-first."""
        window = len(self.obs_buffer)
        if self.obs_count >= window:
            ordered = np.roll(self.obs_buffer, -self.obs_idx, axis=0)
        else:
            ordered = self.obs_buffer[:self.obs_count]
            pad = np.zeros((window - self.obs_count, OBS_DIM), dtype=np.float32)
            ordered = np.concatenate([pad, ordered], axis=0)
        return ordered.flatten()


class TheoryOfMindSystem:
    """Phase 8: Theory of Mind — model other agents' internal states.

    Uses a shared-weight MLP to predict what other agents are 'thinking'
    (in concept space) from their observable behavior. Self-supervised:
    predict concepts → derive expected action via own world model →
    compare with observed action.
    """

    __slots__ = [
        'bottleneck_size', 'tom_max_tracked', 'tom_window', 'tom_lr',
        'tom_weight', 'action_dim',
        '_tracked', '_hidden_layer', '_output_layer', '_action_layer',
        '_cumulative_accuracy', '_update_count',
    ]

    def __init__(self, bottleneck_size: int, tom_max_tracked: int,
                 tom_window: int, tom_lr: float, tom_weight: float,
                 action_dim: int):
        self.bottleneck_size = bottleneck_size
        self.tom_max_tracked = max(2, min(MAX_TOM_TRACKED, tom_max_tracked))
        self.tom_window = max(4, min(MAX_TOM_WINDOW, tom_window))
        self.tom_lr = np.clip(tom_lr, 0.001, 0.05)
        self.tom_weight = np.clip(tom_weight, 0.0, 1.0)
        self.action_dim = action_dim

        # Tracked agents dict: agent_id → TrackedAgent
        self._tracked: dict[int, TrackedAgent] = {}

        # Shared MLP: flattened obs history → predicted concepts / predicted action
        input_dim = self.tom_window * OBS_DIM
        hidden_size = max(16, min(32, bottleneck_size * 2))
        self._hidden_layer = DenseLayer(input_dim, hidden_size, "tanh")
        self._output_layer = DenseLayer(hidden_size, bottleneck_size, "tanh")
        # Action prediction head: hidden → predicted next action (tanh)
        self._action_layer = DenseLayer(hidden_size, action_dim, "tanh")

        self._cumulative_accuracy = 0.0
        self._update_count = 0

    def observe_other(self, other_id: int, rel_pos: np.ndarray,
                      action: np.ndarray, signal_strength: float,
                      utterance_decoded: np.ndarray, tick: int):
        """Record an observation of another agent's behavior.

        Before recording, runs the MLP on existing history to predict this
        agent's action, so that verify_prediction() can later compare the
        prediction against the real action.
        """
        # Build observation vector
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0:2] = rel_pos[:2] if len(rel_pos) >= 2 else 0.0
        n_act = min(len(action), 6)
        obs[2:2 + n_act] = action[:n_act]
        obs[8] = signal_strength
        n_utt = min(len(utterance_decoded), 4)
        obs[9:9 + n_utt] = utterance_decoded[:n_utt]

        if other_id in self._tracked:
            ta = self._tracked[other_id]
            # Predict action BEFORE recording the new observation
            if ta.obs_count >= 2:
                flat = ta.get_flat_observations()
                hidden = self._hidden_layer.forward(flat)
                ta.predicted_action = self._action_layer.forward(hidden).copy()
                ta.has_prediction = True
            ta.record_observation(obs, tick)
        elif len(self._tracked) < self.tom_max_tracked:
            # New slot available
            ta = TrackedAgent(other_id, self.tom_window, self.bottleneck_size,
                              self.action_dim)
            ta.record_observation(obs, tick)
            self._tracked[other_id] = ta
        else:
            # Replace lowest priority tracked agent
            worst_id = min(self._tracked, key=lambda k: self._tracked[k].priority)
            if self._tracked[worst_id].priority < 0.1:
                del self._tracked[worst_id]
                ta = TrackedAgent(other_id, self.tom_window, self.bottleneck_size,
                                  self.action_dim)
                ta.record_observation(obs, tick)
                self._tracked[other_id] = ta

    def predict_other(self, other_id: int) -> np.ndarray:
        """Predict another agent's concept state from their observed behavior."""
        if other_id not in self._tracked:
            return np.zeros(self.bottleneck_size, dtype=np.float32)

        ta = self._tracked[other_id]
        if ta.obs_count < 2:
            return np.zeros(self.bottleneck_size, dtype=np.float32)

        flat = ta.get_flat_observations()
        hidden = self._hidden_layer.forward(flat)
        predicted = self._output_layer.forward(hidden)  # tanh activation
        ta.predicted_concepts = predicted.copy()
        return predicted

    def verify_prediction(self, other_id: int, observed_action: np.ndarray) -> float:
        """Verify ToM prediction against the other agent's actual action.

        Compares the previously predicted action (made during observe_other)
        with the now-observed action. Returns accuracy in [0, 1] where 1 is
        a perfect prediction.
        """
        if other_id not in self._tracked:
            return 0.0

        ta = self._tracked[other_id]
        if not ta.has_prediction:
            return 0.0

        # Truncate observed action to match action_dim
        actual = np.zeros(self.action_dim, dtype=np.float32)
        n = min(len(observed_action), self.action_dim)
        actual[:n] = observed_action[:n]

        # MSE between predicted and actual action
        error_vec = ta.predicted_action - actual
        mse = float(np.mean(error_vec ** 2))

        # Convert MSE to accuracy in [0, 1]. With tanh outputs and actions
        # roughly in [-1, 1], max MSE ~4. Use exp(-mse) for smooth mapping.
        accuracy = float(np.exp(-mse))

        ta.accuracy = ta.accuracy * 0.9 + accuracy * 0.1

        # Backprop prediction error to improve the action prediction head
        # and shared hidden layer
        self._learn_from_verification(ta, error_vec, accuracy)

        ta.has_prediction = False
        return accuracy

    def _learn_from_verification(self, ta: TrackedAgent,
                                 error_vec: np.ndarray, accuracy: float):
        """Backprop action prediction error through action head and hidden layer."""
        self._update_count += 1
        alpha = min(0.01, 1.0 / self._update_count)
        self._cumulative_accuracy = (1 - alpha) * self._cumulative_accuracy + alpha * accuracy

        if accuracy > 0.9:
            return  # Already a good prediction, skip update

        # Re-run forward pass to get hidden activations (needed for backprop)
        flat = ta.get_flat_observations()
        hidden = self._hidden_layer.forward(flat)

        lr = self.tom_lr

        # --- Backprop through action head ---
        # d_loss/d_output = 2 * error / action_dim  (MSE gradient)
        d_act_out = 2.0 * error_vec / max(1, len(error_vec))
        # tanh derivative at action output
        d_act_out = d_act_out * (1.0 - ta.predicted_action ** 2)

        grad_W_act = np.outer(hidden, d_act_out)
        grad_b_act = d_act_out

        # --- Backprop through shared hidden layer (from action head only) ---
        d_hidden = d_act_out @ self._action_layer.W.T
        tanh_deriv = 1.0 - hidden ** 2
        d_hidden = d_hidden * tanh_deriv

        grad_W_hid = np.outer(flat, d_hidden)
        grad_b_hid = d_hidden

        # Gradient clipping
        np.clip(grad_W_act, -1.0, 1.0, out=grad_W_act)
        np.clip(grad_b_act, -1.0, 1.0, out=grad_b_act)
        np.clip(grad_W_hid, -1.0, 1.0, out=grad_W_hid)
        np.clip(grad_b_hid, -1.0, 1.0, out=grad_b_hid)

        # Apply updates
        self._action_layer.W -= lr * grad_W_act
        self._action_layer.b -= lr * grad_b_act
        self._hidden_layer.W -= lr * grad_W_hid
        self._hidden_layer.b -= lr * grad_b_hid

    def update_attention(self, bonds: dict, proximity_map: dict, current_tick: int):
        """Update priority for each tracked agent based on bonds, proximity, recency."""
        stale = []
        for aid, ta in self._tracked.items():
            bond_strength = bonds.get(aid, 0.0)
            proximity = proximity_map.get(aid, 0.0)
            recency = max(0.0, 1.0 - (current_tick - ta.last_seen_tick) / 100.0)
            freq = min(1.0, ta.interaction_count / 50.0)

            ta.priority = (bond_strength * 0.4 + proximity * 0.3 +
                          freq * 0.2 + recency * 0.1)

            # Mark stale entries
            if current_tick - ta.last_seen_tick > 200:
                stale.append(aid)

        for aid in stale:
            del self._tracked[aid]

    def get_most_attended_prediction(self) -> np.ndarray:
        """Return predicted concepts of the highest-priority tracked agent."""
        if not self._tracked:
            return np.zeros(self.bottleneck_size, dtype=np.float32)

        best_id = max(self._tracked, key=lambda k: self._tracked[k].priority)
        ta = self._tracked[best_id]
        if ta.obs_count >= 2:
            self.predict_other(best_id)
        return ta.predicted_concepts.copy()

    def get_tom_summary(self, max_dim: int = 4) -> np.ndarray:
        """Return first max_dim dims of most-attended prediction for context."""
        pred = self.get_most_attended_prediction()
        result = np.zeros(max_dim, dtype=np.float32)
        n = min(len(pred), max_dim)
        result[:n] = pred[:n]
        return result

    @property
    def param_count(self) -> int:
        return (self._hidden_layer.W.size + self._hidden_layer.b.size +
                self._output_layer.W.size + self._output_layer.b.size +
                self._action_layer.W.size + self._action_layer.b.size)

    @property
    def cumulative_accuracy(self) -> float:
        return self._cumulative_accuracy

    def describe_beliefs(self) -> dict:
        """Return interpretable summary of ToM beliefs about others."""
        beliefs = {}
        for aid, ta in self._tracked.items():
            beliefs[aid] = {
                "predicted_concepts": ta.predicted_concepts.copy(),
                "accuracy": ta.accuracy,
                "priority": ta.priority,
                "observations": ta.obs_count,
                "interaction_count": ta.interaction_count,
            }
        return {
            "tracked_count": len(self._tracked),
            "mean_accuracy": self._cumulative_accuracy,
            "beliefs": beliefs,
        }
