
"""Spacetime: local time, variable clock rates, causal event system.

The universe is an interaction graph U = (V, E).
Each node (grid cell or agent) has:
  - tau_i: local proper time (accumulated)
  - r_i: clock rate (how fast local time flows per meta-step)
  - delta_i: update threshold (how much tau must accumulate before node updates)

Global tick is just a meta-step index n = 0, 1, 2, ...
Not physical time — just a scheduling counter.

Agents additionally have:
  - theta_i: cognitive time (subjective experience of time)
  - c_i: cognitive rate (processing intensity, modulated by emotions)
"""

import numpy as np


class LocalTimeField:
    """Per-cell local time and clock rates for the spatial grid.

    Clock rate r_i = f(substance_density, temperature, stability)
    High mass/density regions have slower clock rates (GR analogy).
    High temperature regions have faster chemistry clocks.
    """

    def __init__(self, width: int, height: int):
        self.w = width
        self.h = height

        # Local proper time per cell
        self.tau = np.zeros((width, height), dtype=np.float64)

        # Clock rate per cell — 1.0 = normal, <1 = time dilation, >1 = faster
        self.clock_rate = np.ones((width, height), dtype=np.float32)

        # Update threshold — how much tau must accumulate for a cell to "tick"
        self.update_threshold = np.ones((width, height), dtype=np.float32)

        # Accumulated since last update
        self.tau_since_update = np.zeros((width, height), dtype=np.float32)

        # Coherence: how predictable is this cell's state (0=chaotic, 1=stable)
        self.coherence = np.ones((width, height), dtype=np.float32)

        # Causal influence: how many events originated from this cell
        self._event_count = np.zeros((width, height), dtype=np.int32)

    def compute_clock_rates(self, substance_density, temperature, stability_field=None):
        """Recompute clock rates from physical conditions.

        Clock rate = base * temperature_factor / mass_factor
        - High substance density -> slower (gravitational time dilation analogy)
        - High temperature -> faster (thermal agitation)
        - Low stability -> noisier updates
        """
        # Mass factor: total substance concentration slows time
        mass = np.clip(substance_density, 0.0, 10.0)
        mass_factor = 1.0 + mass * 0.1  # denser = slower

        # Temperature factor: warmer = faster local physics
        temp_factor = 0.5 + temperature  # range ~0.5 to 1.5

        self.clock_rate = temp_factor / mass_factor
        np.clip(self.clock_rate, 0.2, 3.0, out=self.clock_rate)

        # Coherence from stability
        if stability_field is not None:
            self.coherence = 0.3 + 0.7 * np.clip(stability_field, 0.0, 1.0)

    def advance(self, dt: float = 1.0):
        """Advance local times by clock_rate * dt. Returns mask of cells that should update."""
        increment = self.clock_rate * dt
        self.tau += increment
        self.tau_since_update += increment

        # Cells that crossed their update threshold
        should_update = self.tau_since_update >= self.update_threshold
        self.tau_since_update[should_update] -= self.update_threshold[should_update]

        return should_update

    def get_time_dilation(self, x, y):
        """How dilated is time at this position? <1 = slower, >1 = faster."""
        return float(self.clock_rate[x % self.w, y % self.h])

    def get_local_time(self, x, y):
        """Get accumulated local proper time at position."""
        return float(self.tau[x % self.w, y % self.h])


class CausalEvent:
    """A discrete event in the causal graph."""
    __slots__ = ['source_id', 'source_pos', 'tau_emitted', 'payload',
                 'propagation_speed', 'radius', 'event_type']

    def __init__(self, source_id: int, source_pos: np.ndarray,
                 tau_emitted: float, payload: np.ndarray,
                 propagation_speed: float = 1.0, radius: int = 8,
                 event_type: str = "signal"):
        self.source_id = source_id
        self.source_pos = source_pos.copy()
        self.tau_emitted = tau_emitted
        self.payload = payload
        self.propagation_speed = propagation_speed
        self.radius = radius
        self.event_type = event_type

    def can_reach(self, target_pos: np.ndarray, target_tau: float,
                  world_w: int, world_h: int) -> bool:
        """Check if this event can causally reach the target (light cone check)."""
        dx = abs(target_pos[0] - self.source_pos[0])
        dy = abs(target_pos[1] - self.source_pos[1])
        dx = min(dx, world_w - dx)
        dy = min(dy, world_h - dy)
        distance = dx + dy

        if distance > self.radius:
            return False

        # Causal check: enough local time must have passed for signal to arrive
        travel_time = distance / max(0.1, self.propagation_speed)
        return target_tau >= self.tau_emitted + travel_time


class CausalEventSystem:
    """Manages causal events with propagation delay.

    Events don't arrive instantly — they propagate at finite speed.
    This implements: m_{j->i} arrives only when tau_i >= tau_j + d_{ij}
    """

    def __init__(self, world_width: int, world_height: int,
                 max_propagation_speed: float = 2.0):
        self.w = world_width
        self.h = world_height
        self.max_speed = max_propagation_speed
        self._events: list[CausalEvent] = []
        self._stats = {"emitted": 0, "received": 0, "expired": 0}

    def emit(self, event: CausalEvent):
        """Emit a causal event into the universe."""
        self._events.append(event)
        self._stats["emitted"] += 1

    def receive(self, receiver_id: int, receiver_pos: np.ndarray,
                receiver_tau: float, event_type: str = None) -> list[CausalEvent]:
        """Get all events that have causally reached this receiver."""
        reached = []
        for e in self._events:
            if e.source_id == receiver_id:
                continue
            if event_type is not None and e.event_type != event_type:
                continue
            if e.can_reach(receiver_pos, receiver_tau, self.w, self.h):
                reached.append(e)
                self._stats["received"] += 1
        return reached

    def cleanup(self, min_tau: float):
        """Remove events that are too old to reach anyone."""
        before = len(self._events)
        # An event can't reach anyone if even the fastest-propagating version
        # would have passed the entire world
        max_travel = (self.w + self.h) / max(0.1, self.max_speed)
        self._events = [e for e in self._events
                        if min_tau - e.tau_emitted <= max_travel + 5]
        self._stats["expired"] += before - len(self._events)

    @property
    def stats(self):
        s = self._stats.copy()
        s["active_events"] = len(self._events)
        self._stats = {"emitted": 0, "received": 0, "expired": 0}
        return s


class AgentSpacetime:
    """Per-agent local time tracking: physical time and cognitive time.

    Physical time (tau): flows based on local spacetime clock rate.
    Cognitive time (theta): flows based on processing intensity.

    Cognitive rate c_i = f(dopamine_proxy, stress, focus, fatigue)
    - High fear -> high cognitive rate (time slows subjectively)
    - High curiosity + low fear -> moderate cognitive rate (flow state)
    - Low energy (fatigue) -> low cognitive rate (time drags)
    """

    __slots__ = ['tau', 'theta', 'clock_rate', 'cognitive_rate',
                 '_last_tau_delta', '_last_theta_delta',
                 'time_perception', '_accumulated_tau']

    def __init__(self):
        self.tau = 0.0           # physical proper time
        self.theta = 0.0         # cognitive/subjective time
        self.clock_rate = 1.0    # physical clock rate (from local spacetime)
        self.cognitive_rate = 1.0  # cognitive processing rate
        self._last_tau_delta = 1.0
        self._last_theta_delta = 1.0
        self.time_perception = 1.0  # ratio: how fast does time FEEL? >1 = feels fast
        self._accumulated_tau = 0.0  # accumulator for variable tick rate

    def update_clock_rate(self, local_time_field: LocalTimeField, x: int, y: int):
        """Set physical clock rate from the local spacetime field."""
        self.clock_rate = local_time_field.get_time_dilation(x, y)

    def update_cognitive_rate(self, fear: float, curiosity: float, energy: float,
                               social_need: float, nostalgia: float):
        """Compute cognitive processing rate from emotional state.

        fear high -> hypervigilance -> high cognitive rate (time "slows")
        curiosity high + safe -> flow state -> moderate-high cognitive rate
        energy low -> fatigue -> low cognitive rate
        nostalgia high -> reflective mode -> slightly lower cognitive rate
        """
        # Base processing
        base = 0.5

        # Fear: adrenaline response — massively increases processing
        fear_boost = fear * 1.5

        # Curiosity: dopamine-driven engagement
        safety = max(0, 1.0 - fear)
        curiosity_boost = curiosity * safety * 0.8

        # Energy: fatigue reduces processing capacity
        fatigue_penalty = max(0, 0.5 - energy) * 0.6

        # Nostalgia: reflective, slightly slower processing
        nostalgia_penalty = nostalgia * 0.2

        self.cognitive_rate = np.clip(
            base + fear_boost + curiosity_boost - fatigue_penalty - nostalgia_penalty,
            0.1, 3.0
        )

    def advance(self, dt: float = 1.0):
        """Advance both physical and cognitive time."""
        self._last_tau_delta = self.clock_rate * dt
        self._last_theta_delta = self.cognitive_rate * dt

        self.tau += self._last_tau_delta
        self.theta += self._last_theta_delta
        self._accumulated_tau += self._last_tau_delta

        # Time perception: ratio of cognitive to physical flow
        if self._last_tau_delta > 1e-8:
            self.time_perception = self._last_theta_delta / self._last_tau_delta
        else:
            self.time_perception = 1.0

    def should_update(self) -> int:
        """How many updates this agent gets this meta-tick.
        clock_rate 1.0 -> 1. clock_rate 0.5 -> 0 or 1. clock_rate 2.0 -> 1 or 2.
        Returns 0 if not enough time accumulated."""
        count = 0
        while self._accumulated_tau >= 1.0:
            self._accumulated_tau -= 1.0
            count += 1
            if count >= 3:  # cap at 3 updates per meta-tick
                break
        return count

    def get_context(self) -> np.ndarray:
        """Return spacetime context for brain input. 4D vector."""
        return np.array([
            min(self.clock_rate, 3.0) / 3.0,       # physical time dilation [0-1]
            min(self.cognitive_rate, 3.0) / 3.0,    # cognitive speed [0-1]
            min(self.time_perception, 3.0) / 3.0,   # subjective time feel [0-1]
            min(self.tau, 10000.0) / 10000.0,       # age in proper time [0-1]
        ], dtype=np.float64)
