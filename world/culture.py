
import numpy as np


class CulturalEvent:
    __slots__ = ['type', 'teacher_id', 'student_id', 'rule_desc', 'tick']

    def __init__(self, etype: str, teacher: int, student: int, desc: str, tick: int):
        self.type = etype
        self.teacher_id = teacher
        self.student_id = student
        self.rule_desc = desc
        self.tick = tick


class CultureSystem:

    def __init__(self):
        self.events: list[CulturalEvent] = []
        self._tick_events: list[CulturalEvent] = []
        self.total_teachings = 0
        self.total_imitations = 0

    def process_teaching(self, agents: list, tick: int, rng: np.random.Generator):
        self._tick_events = []

        for teacher in agents:
            if not teacher.is_alive:
                continue

            best_rules = teacher.hypotheses.get_best_hypotheses(
                min_tests=15, min_accuracy=0.7
            )
            if not best_rules:
                continue

            for student in agents:
                if student.id == teacher.id or not student.is_alive:
                    continue

                dx = abs(teacher.body.position[0] - student.body.position[0])
                dy = abs(teacher.body.position[1] - student.body.position[1])
                if dx > 2 or dy > 2:
                    continue

                if rng.random() > 0.02:
                    continue

                rule = best_rules[0]

                student_hyps = student.hypotheses.hypotheses
                if student_hyps:
                    worst_idx = min(range(len(student_hyps)),
                                   key=lambda i: student_hyps[i].accuracy * student_hyps[i].confidence
                                   if student_hyps[i].tests > 3 else 1.0)

                    from agents.hypothesis import Hypothesis, Condition
                    new_conditions = [
                        Condition(c.feature, c.comparator, c.threshold)
                        for c in rule.conditions
                    ]
                    new_rule = Hypothesis(
                        new_conditions, rule.outcome,
                        rule.action_bias.copy()
                    )
                    new_rule.tests = 3
                    new_rule.successes = 2

                    student.hypotheses.hypotheses[worst_idx] = new_rule

                    self.total_teachings += 1
                    event = CulturalEvent(
                        "teach", teacher.id, student.id,
                        rule.describe(), tick
                    )
                    self._tick_events.append(event)
                    self.events.append(event)

                    break

    def process_imitation(self, agents: list, tick: int, rng: np.random.Generator):
        for agent in agents:
            if not agent.is_alive or rng.random() > 0.01:
                continue

            best_neighbor = None
            best_reward = agent.total_reward

            for other in agents:
                if other.id == agent.id or not other.is_alive:
                    continue
                dx = abs(agent.body.position[0] - other.body.position[0])
                dy = abs(agent.body.position[1] - other.body.position[1])
                if dx > 3 or dy > 3:
                    continue
                if other.total_reward > best_reward:
                    best_neighbor = other
                    best_reward = other.total_reward

            if best_neighbor is not None:
                if hasattr(agent, 'composable') and hasattr(best_neighbor, 'composable'):
                    for my_rule, their_rule in zip(agent.composable.rules,
                                                    best_neighbor.composable.rules):
                        n = min(len(my_rule.action_bias), len(their_rule.action_bias))
                        my_rule.action_bias[:n] += 0.05 * (their_rule.action_bias[:n] - my_rule.action_bias[:n])

                self.total_imitations += 1
                self._tick_events.append(CulturalEvent(
                    "imitate", best_neighbor.id, agent.id, "", tick
                ))

    def cleanup(self):
        if len(self.events) > 5000:
            self.events = self.events[-2500:]

    @property
    def recent_stats(self) -> dict:
        return {
            "teachings": sum(1 for e in self._tick_events if e.type == "teach"),
            "imitations": sum(1 for e in self._tick_events if e.type == "imitate"),
            "total_teachings": self.total_teachings,
            "total_imitations": self.total_imitations,
        }
