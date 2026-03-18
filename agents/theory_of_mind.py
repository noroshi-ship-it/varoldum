
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
    ]

    def __init__(self, agent_id: int, tom_window: int, bottleneck_size: int):
        self.agent_id = agent_id
        self.obs_buffer = np.zeros((tom_window, OBS_DIM), dtype=np.float32)
        self.obs_idx = 0
        self.obs_count = 0
        self.predicted_concepts = np.zeros(bottleneck_size, dtype=np.float32)
        self.accuracy = 0.0
        self.priority = 0.0
        self.interaction_count = 0
        self.last_seen_tick = 0

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
        '_tracked', '_hidden_layer', '_output_layer',
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

        # Shared MLP: flattened obs history → predicted concepts
        input_dim = self.tom_window * OBS_DIM
        hidden_size = max(16, min(32, bottleneck_size * 2))
        self._hidden_layer = DenseLayer(input_dim, hidden_size, "tanh")
        self._output_layer = DenseLayer(hidden_size, bottleneck_size, "tanh")

        self._cumulative_accuracy = 0.0
        self._update_count = 0

    def observe_other(self, other_id: int, rel_pos: np.ndarray,
                      action: np.ndarray, signal_strength: float,
                      utterance_decoded: np.ndarray, tick: int):
        """Record an observation of another agent's behavior."""
        # Build observation vector
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[0:2] = rel_pos[:2] if len(rel_pos) >= 2 else 0.0
        n_act = min(len(action), 6)
        obs[2:2 + n_act] = action[:n_act]
        obs[8] = signal_strength
        n_utt = min(len(utterance_decoded), 4)
        obs[9:9 + n_utt] = utterance_decoded[:n_utt]

        if other_id in self._tracked:
            self._tracked[other_id].record_observation(obs, tick)
        elif len(self._tracked) < self.tom_max_tracked:
            # New slot available
            ta = TrackedAgent(other_id, self.tom_window, self.bottleneck_size)
            ta.record_observation(obs, tick)
            self._tracked[other_id] = ta
        else:
            # Replace lowest priority tracked agent
            worst_id = min(self._tracked, key=lambda k: self._tracked[k].priority)
            if self._tracked[worst_id].priority < 0.1:
                del self._tracked[worst_id]
                ta = TrackedAgent(other_id, self.tom_window, self.bottleneck_size)
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
        """Verify ToM prediction by checking if predicted concepts are consistent
        with the observed action. Returns accuracy in [0, 1]."""
        if other_id not in self._tracked:
            return 0.0

        ta = self._tracked[other_id]
        if ta.obs_count < 3:
            return 0.0

        # Simple verification: did the predicted concepts change in a way
        # consistent with the observed action direction?
        # Use cosine similarity between predicted concept-change and action
        prev_obs = ta.obs_buffer[(ta.obs_idx - 2) % len(ta.obs_buffer)]
        prev_action = prev_obs[2:8]
        predicted = ta.predicted_concepts

        # The prediction is "good" if the agent's action is consistent
        # with what we'd expect from someone in that concept state
        # Simple heuristic: correlation between action magnitude and concept magnitude
        act_energy = float(np.sum(observed_action[:min(6, len(observed_action))] ** 2))
        concept_energy = float(np.sum(predicted ** 2))

        # Both should be correlated (active agent = active concepts)
        if act_energy < 1e-8 and concept_energy < 1e-8:
            accuracy = 0.5
        else:
            # Normalized correlation
            accuracy = max(0.0, min(1.0,
                0.5 + 0.5 * (1.0 - abs(act_energy - concept_energy) /
                              max(act_energy, concept_energy, 0.01))
            ))

        ta.accuracy = ta.accuracy * 0.9 + accuracy * 0.1

        # Online learning: update network to better predict
        self._learn_from_verification(ta, accuracy)

        return accuracy

    def _learn_from_verification(self, ta: TrackedAgent, accuracy: float):
        """Update ToM network based on prediction quality."""
        self._update_count += 1
        alpha = min(0.01, 1.0 / self._update_count)
        self._cumulative_accuracy = (1 - alpha) * self._cumulative_accuracy + alpha * accuracy

        if accuracy > 0.7:
            return  # Good prediction, no update needed

        # The prediction was bad — nudge toward observed pattern
        flat = ta.get_flat_observations()
        hidden = self._hidden_layer.forward(flat)

        # Target: use last observed action as a rough concept proxy
        # (Not perfect, but provides gradient signal)
        last_obs = ta.obs_buffer[(ta.obs_idx - 1) % len(ta.obs_buffer)]
        target = np.zeros(self.bottleneck_size, dtype=np.float32)
        # Fill from action + signal as proxy for concept state
        n = min(6, self.bottleneck_size)
        target[:n] = last_obs[2:2 + n]  # action as concept proxy
        if self.bottleneck_size > 6:
            target[6:min(10, self.bottleneck_size)] = last_obs[9:9 + min(4, self.bottleneck_size - 6)]

        error = ta.predicted_concepts - target
        lr = self.tom_lr

        # Backprop output
        d_out = 2.0 * error / max(1, len(error))
        # tanh derivative
        d_out = d_out * (1.0 - ta.predicted_concepts ** 2)

        grad_W_out = np.outer(hidden, d_out)
        grad_b_out = d_out

        # Backprop hidden
        d_hidden = d_out @ self._output_layer.W.T
        tanh_deriv = 1.0 - hidden ** 2
        d_hidden = d_hidden * tanh_deriv

        grad_W_hid = np.outer(flat, d_hidden)
        grad_b_hid = d_hidden

        np.clip(grad_W_out, -1.0, 1.0, out=grad_W_out)
        np.clip(grad_b_out, -1.0, 1.0, out=grad_b_out)
        np.clip(grad_W_hid, -1.0, 1.0, out=grad_W_hid)
        np.clip(grad_b_hid, -1.0, 1.0, out=grad_b_hid)

        self._output_layer.W -= lr * grad_W_out
        self._output_layer.b -= lr * grad_b_out
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
                self._output_layer.W.size + self._output_layer.b.size)

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
