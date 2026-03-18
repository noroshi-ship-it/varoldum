
import numpy as np


class Goal:
    __slots__ = ['target_concept', 'priority', 'progress', 'ticks_active',
                 'parent_goal_idx', 'status', 'origin', 'frustration',
                 'initial_similarity']

    def __init__(self, target_concept: np.ndarray, priority: float,
                 origin: str, parent_goal_idx: int = -1):
        self.target_concept = target_concept.copy()
        self.priority = float(priority)
        self.progress = 0.0
        self.ticks_active = 0
        self.parent_goal_idx = parent_goal_idx
        self.status = "active"
        self.origin = origin
        self.frustration = 0.0
        self.initial_similarity = 0.0


class GoalSystem:
    """Persistent multi-step goal system with commitment and frustration."""

    def __init__(self, max_depth: int, commitment_strength: float,
                 patience: float, horizon: float,
                 communication_weight: float,
                 bottleneck_size: int, action_dim: int):
        self.max_depth = max(0, int(max_depth))
        self.commitment_strength = commitment_strength
        self.patience = max(5.0, patience)
        self.horizon = horizon
        self.communication_weight = communication_weight
        self.bottleneck_size = bottleneck_size
        self.action_dim = action_dim
        self.goal_stack: list[Goal] = []
        self._recently_completed = False
        self._goal_selection_cooldown = 0

    def select_goal(self, internal_state: np.ndarray, concepts: np.ndarray,
                    episodic_targets: list[np.ndarray] | None = None,
                    social_request: np.ndarray | None = None) -> Goal | None:
        """Select a new goal based on internal needs and context."""
        if self.max_depth <= 0:
            return None

        # Don't reselect too often
        if self._goal_selection_cooldown > 0:
            self._goal_selection_cooldown -= 1
            return self.current_goal

        # If we have an active goal with low frustration, keep it
        if self.goal_stack and self.goal_stack[-1].status == "active":
            if self.goal_stack[-1].frustration < 0.7:
                return self.goal_stack[-1]

        # Clear abandoned/completed goals
        while self.goal_stack and self.goal_stack[-1].status != "active":
            self.goal_stack.pop()

        # Priority 1: Survival needs
        hunger = float(internal_state[0]) if len(internal_state) > 0 else 0.5
        fear = float(internal_state[1]) if len(internal_state) > 1 else 0.0

        target = None
        priority = 0.0
        origin = "need"

        if hunger > 0.6:
            # Find food — use episodic memory if available
            if episodic_targets and len(episodic_targets) > 0:
                target = episodic_targets[0].copy()
            else:
                target = self._need_target(concepts, 0, internal_state)
            priority = min(1.0, hunger)
            origin = "need"

        elif fear > 0.5:
            # Seek safety — inverse of current fearful state
            target = concepts.copy()
            if len(target) > 1:
                target[1] = -abs(target[1])  # reduce fear dimension
            priority = min(1.0, fear)
            origin = "need"

        # Priority 2: Social request
        elif social_request is not None and np.linalg.norm(social_request) > 0.3:
            target = social_request.copy()
            priority = 0.5
            origin = "social"

        # Priority 3: Highest unmet need
        elif np.max(internal_state[:4]) > 0.5:
            need_idx = int(np.argmax(internal_state[:4]))
            target = self._need_target(concepts, need_idx, internal_state)
            priority = float(internal_state[need_idx])
            origin = "need"

        # Priority 4: Curiosity (horizon gene biases toward this)
        elif self.horizon > 0.5:
            target = concepts + np.random.randn(len(concepts)) * 0.3
            priority = 0.3 * self.horizon
            origin = "curiosity"

        # Priority 5: Replay past positive outcome
        elif episodic_targets and len(episodic_targets) > 0 and self.horizon > 0.3:
            target = episodic_targets[0].copy()
            priority = 0.3
            origin = "episodic"

        if target is None:
            return None

        # Ensure target is bottleneck-sized
        if len(target) < self.bottleneck_size:
            padded = np.zeros(self.bottleneck_size)
            padded[:len(target)] = target
            target = padded
        elif len(target) > self.bottleneck_size:
            target = target[:self.bottleneck_size]

        goal = Goal(target, priority, origin)

        # Compute initial similarity so we can measure progress
        n = min(len(concepts), len(target))
        c_norm = np.linalg.norm(concepts[:n])
        t_norm = np.linalg.norm(target[:n])
        if c_norm > 1e-8 and t_norm > 1e-8:
            goal.initial_similarity = float(np.dot(concepts[:n], target[:n])) / (c_norm * t_norm)

        # Replace current goal
        if self.goal_stack:
            self.goal_stack[-1] = goal
        else:
            self.goal_stack.append(goal)

        self._goal_selection_cooldown = 20
        return goal

    def _need_target(self, concepts: np.ndarray, need_idx: int,
                     internal_state: np.ndarray) -> np.ndarray:
        """Generate target concept for a need — perturb current concepts away from need."""
        target = concepts.copy()
        # Modulate concept dims associated with the need
        if need_idx < len(target):
            target[need_idx] = -target[need_idx]  # flip the need-associated dimension
        return target

    def get_action_bias(self, current_concepts: np.ndarray) -> np.ndarray:
        """Return action-space bias toward current goal."""
        bias = np.zeros(self.action_dim, dtype=np.float64)
        goal = self.current_goal
        if goal is None:
            return bias

        # Direction in concept space
        n = min(len(current_concepts), len(goal.target_concept))
        direction = goal.target_concept[:n] - current_concepts[:n]

        # Project to action space: use first action_dim dims of direction
        m = min(self.action_dim, len(direction))
        bias[:m] = direction[:m] * self.commitment_strength * 0.3

        return np.clip(bias, -0.5, 0.5)

    def update_progress(self, current_concepts: np.ndarray, reward: float, tick: int):
        """Update goal progress and frustration."""
        goal = self.current_goal
        if goal is None:
            return

        goal.ticks_active += 1

        # Measure similarity to target
        n = min(len(current_concepts), len(goal.target_concept))
        c_norm = np.linalg.norm(current_concepts[:n])
        t_norm = np.linalg.norm(goal.target_concept[:n])
        if c_norm > 1e-8 and t_norm > 1e-8:
            similarity = float(np.dot(current_concepts[:n], goal.target_concept[:n])) / (c_norm * t_norm)
        else:
            similarity = 0.0

        # Progress is improvement over initial
        if goal.initial_similarity < 0.99:
            goal.progress = max(0.0, (similarity - goal.initial_similarity) / (1.0 - goal.initial_similarity + 1e-8))
        else:
            goal.progress = 1.0

        # Frustration management
        if reward > 0.1 or goal.progress > goal.frustration * 0.5:
            goal.frustration = max(0.0, goal.frustration - 0.05)
        else:
            goal.frustration += 1.0 / self.patience

        # Goal completion
        if goal.progress > 0.8 or similarity > 0.9:
            goal.status = "completed"
            self._recently_completed = True

        # Goal abandonment
        if goal.frustration > 1.0:
            goal.status = "abandoned"

    def get_goal_context(self, max_dim: int = 6) -> np.ndarray:
        """Return fixed-size context vector from current goal."""
        result = np.zeros(max_dim, dtype=np.float64)
        goal = self.current_goal
        if goal is None:
            return result

        # First dims: target concept excerpt
        n = min(max_dim - 3, len(goal.target_concept))
        if n > 0:
            result[:n] = goal.target_concept[:n]

        # Last 3 dims: priority, progress, frustration
        result[max_dim - 3] = goal.priority
        result[max_dim - 2] = goal.progress
        result[max_dim - 1] = goal.frustration
        return result

    def get_goal_signal(self) -> np.ndarray:
        """Encode current goal as concept vector for communication."""
        result = np.zeros(self.bottleneck_size, dtype=np.float64)
        goal = self.current_goal
        if goal is None or self.communication_weight < 0.1:
            return result

        n = min(self.bottleneck_size, len(goal.target_concept))
        result[:n] = goal.target_concept[:n] * self.communication_weight
        return result

    def abandon_all(self):
        """Panic mode: abandon all goals."""
        for g in self.goal_stack:
            g.status = "abandoned"
        self.goal_stack.clear()

    def check_and_clear_completion(self) -> bool:
        """Check if a goal was recently completed and clear flag."""
        if self._recently_completed:
            self._recently_completed = False
            return True
        return False

    @property
    def current_goal(self) -> Goal | None:
        if not self.goal_stack:
            return None
        top = self.goal_stack[-1]
        if top.status == "active":
            return top
        return None

    @property
    def has_active_goal(self) -> bool:
        return self.current_goal is not None

    @property
    def param_count(self) -> int:
        return 0
