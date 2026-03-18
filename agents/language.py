
import numpy as np

SIGNAL_DIM = 4


class Utterance:
    __slots__ = ['signal', 'sender_id', 'sender_pos', 'sender_lineage', 'tick']

    def __init__(self, signal: np.ndarray, sender_id: int, sender_pos: np.ndarray,
                 sender_lineage: int, tick: int):
        self.signal = signal.copy()
        self.sender_id = sender_id
        self.sender_pos = sender_pos.copy()
        self.sender_lineage = sender_lineage
        self.tick = tick


class ListenerModel:

    def __init__(self, signal_dim: int = SIGNAL_DIM):
        self.signal_dim = signal_dim
        self.W = np.random.randn(signal_dim, 3) * 0.1
        self.b = np.zeros(3)
        self._last_input = None
        self._last_prediction = np.zeros(3)
        self.accuracy = 0.0
        self._n_updates = 0

    def predict(self, signal: np.ndarray) -> np.ndarray:
        x = np.zeros(self.signal_dim)
        n = min(len(signal), self.signal_dim)
        x[:n] = signal[:n]
        self._last_input = x
        self._last_prediction = np.tanh(x @ self.W + self.b)
        return self._last_prediction.copy()

    def learn(self, actual_outcome: np.ndarray, lr: float = 0.01):
        if self._last_input is None:
            return
        actual = np.zeros(3)
        n = min(len(actual_outcome), 3)
        actual[:n] = actual_outcome[:n]

        error = self._last_prediction - actual
        mse = float(np.mean(error ** 2))

        self._n_updates += 1
        alpha = min(0.01, 1.0 / self._n_updates)
        self.accuracy = (1 - alpha) * self.accuracy + alpha * (1.0 - min(1.0, mse))

        grad_W = np.outer(self._last_input, error)
        grad_b = error
        grad_W = np.clip(grad_W, -1, 1)
        grad_b = np.clip(grad_b, -1, 1)
        self.W -= lr * grad_W
        self.b -= lr * grad_b


class ProtoLanguage:

    def __init__(self, world_width: int, world_height: int, hear_radius: int = 8):
        self.w = world_width
        self.h = world_height
        self.hear_radius = hear_radius
        self._utterances: list[Utterance] = []
        self._tick_stats = {"signals_sent": 0, "signals_heard": 0,
                            "unique_senders": 0}

    def broadcast(self, agent, signal: np.ndarray, tick: int):
        strength = np.linalg.norm(signal)
        if strength < 0.3:
            return
        self._utterances.append(Utterance(
            signal=np.clip(signal, -1, 1),
            sender_id=agent.id,
            sender_pos=agent.body.position,
            sender_lineage=agent.lineage_id,
            tick=tick,
        ))
        self._tick_stats["signals_sent"] += 1

    PROPAGATION_SPEED = 4.0  # cells per tick — finite speed of sound

    def get_heard_signals(self, agent, tick: int) -> list[Utterance]:
        heard = []
        ax, ay = agent.body.position[0], agent.body.position[1]
        for u in self._utterances:
            if u.sender_id == agent.id:
                continue
            max_age = self.hear_radius / self.PROPAGATION_SPEED + 1
            if u.tick < tick - max_age:
                continue
            dx = abs(u.sender_pos[0] - ax)
            dy = abs(u.sender_pos[1] - ay)
            dx = min(dx, self.w - dx)
            dy = min(dy, self.h - dy)
            d = dx + dy
            if d > self.hear_radius:
                continue
            # Causal delay: signal needs time to travel
            travel_time = d / self.PROPAGATION_SPEED
            if tick < u.tick + travel_time:
                continue
            heard.append(u)
        self._tick_stats["signals_heard"] += len(heard)
        return heard

    def get_strongest_signal(self, agent, tick: int) -> np.ndarray:
        heard = self.get_heard_signals(agent, tick)
        if not heard:
            return np.zeros(SIGNAL_DIM)
        ax, ay = agent.body.position
        best = None
        best_dist = float('inf')
        for u in heard:
            dx = abs(u.sender_pos[0] - ax)
            dy = abs(u.sender_pos[1] - ay)
            dx = min(dx, self.w - dx)
            dy = min(dy, self.h - dy)
            d = dx + dy
            if d < best_dist:
                best_dist = d
                best = u
        if best is None:
            return np.zeros(SIGNAL_DIM)
        atten = 1.0 / (1.0 + best_dist * 0.2)
        return best.signal * atten

    def cleanup(self, tick: int):
        max_age = self.hear_radius / self.PROPAGATION_SPEED + 2
        self._utterances = [u for u in self._utterances if u.tick >= tick - max_age]

    @property
    def recent_stats(self):
        stats = self._tick_stats.copy()
        stats["unique_senders"] = len(set(u.sender_id for u in self._utterances))
        self._tick_stats = {"signals_sent": 0, "signals_heard": 0, "unique_senders": 0}
        return stats
