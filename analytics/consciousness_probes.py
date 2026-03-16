
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.agent import Agent


def self_model_accuracy(agent) -> float:
    return agent.self_model.cumulative_accuracy


def information_integration_proxy(agent) -> float:
    hidden = agent.brain.hidden_state
    sensor = agent._raw_input

    if len(hidden) == 0 or len(sensor) == 0:
        return 0.0

    mid = len(sensor) // 2
    s1, s2 = sensor[:mid], sensor[mid:]

    def _corr(a, b):
        if len(a) == 0 or len(b) == 0:
            return 0.0
        a_flat = a[:min(len(a), len(b))]
        b_flat = b[:min(len(a), len(b))]
        if np.std(a_flat) < 1e-8 or np.std(b_flat) < 1e-8:
            return 0.0
        return float(abs(np.corrcoef(a_flat, b_flat)[0, 1]))

    h_trimmed = hidden[:min(len(hidden), max(len(s1), len(s2)))]
    c1 = _corr(s1, h_trimmed[:len(s1)])
    c2 = _corr(s2, h_trimmed[:len(s2)])

    return float(np.sqrt(max(0, c1 * c2)))


def counterfactual_sensitivity(agent, perturbation_scale: float = 0.1) -> float:
    original_action = agent._action.copy()

    original_internal = agent.internal.as_vector().copy()
    perturbed = original_internal + np.random.normal(0, perturbation_scale, size=original_internal.shape)

    if np.std(original_action) < 1e-8:
        return 0.0

    internal_expanded = np.zeros(len(original_action))
    n = min(3, len(original_action))
    internal_expanded[:n] = original_internal[:n]

    correlation = abs(np.corrcoef(
        original_action[:max(2, len(original_action))],
        internal_expanded[:max(2, len(original_action))]
    )[0, 1]) if len(original_action) > 1 else 0.0

    return float(correlation) if not np.isnan(correlation) else 0.0


def behavioral_sequence_complexity(action_history: list[np.ndarray]) -> float:
    if len(action_history) < 2:
        return 0.0

    symbols = []
    for action in action_history:
        s = tuple(int(np.sign(a)) for a in action)
        symbols.append(s)

    n = len(symbols)
    complexity = 1
    i = 0
    while i < n:
        best_len = 0
        for j in range(i):
            match_len = 0
            while (i + match_len < n and j + match_len < i and
                   symbols[j + match_len] == symbols[i + match_len]):
                match_len += 1
            best_len = max(best_len, match_len)
        complexity += 1
        i += max(1, best_len + 1)

    max_complexity = n / max(1, np.log2(n + 1))
    return float(min(1.0, complexity / max(1, max_complexity)))


def probe_all(agent) -> dict:
    return {
        "self_model_accuracy": self_model_accuracy(agent),
        "information_integration": information_integration_proxy(agent),
        "counterfactual_sensitivity": counterfactual_sensitivity(agent),
    }
