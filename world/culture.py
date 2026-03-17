
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
        self.total_lineage_studies = 0

    def process_teaching(self, agents: list, tick: int, rng: np.random.Generator,
                         lineage_memory=None):
        self._tick_events = []

        # Build nearby-agents lookup for efficiency
        alive = [a for a in agents if a.is_alive]
        if not alive:
            return

        for teacher in alive:
            best_rules = teacher.hypotheses.get_best_hypotheses(
                min_tests=15, min_accuracy=0.7
            )
            if not best_rules:
                continue

            # Fitness rank for this teacher (higher = better teacher)
            teacher_fitness = teacher.total_reward

            for student in alive:
                if student.id == teacher.id:
                    continue

                dx = abs(teacher.body.position[0] - student.body.position[0])
                dy = abs(teacher.body.position[1] - student.body.position[1])
                if dx > 2 or dy > 2:
                    continue

                # Fitness-weighted probability instead of flat 2%
                base_prob = 0.005
                fitness_bonus = min(2.0, max(0.1, teacher_fitness / max(1, abs(student.total_reward) + 1)))

                # Lineage trust bonus
                lineage_bonus = 1.0
                if teacher.lineage_id == student.lineage_id and teacher.lineage_id != -1:
                    lineage_bonus = 3.0

                # Student receptivity from genome
                receptivity = 1.0
                if hasattr(student, '_teaching_receptivity'):
                    receptivity = student._teaching_receptivity

                prob = base_prob * fitness_bonus * lineage_bonus * receptivity
                if rng.random() > prob:
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

                    # Contribute to lineage memory
                    if lineage_memory is not None:
                        encoded = teacher.hypotheses.encode_best(min_accuracy=0.7)
                        if encoded is not None:
                            lineage_memory.contribute(
                                teacher.lineage_id, encoded,
                                rule.accuracy, teacher.id,
                                teacher.generation, tick,
                            )

                    # Teacher social satisfaction boost
                    if hasattr(teacher, '_pending_social_reward'):
                        teacher._pending_social_reward += 0.2

                    break

    def process_lineage_study(self, agents: list, lineage_memory, tick: int,
                              rng: np.random.Generator):
        """Young agents study from their lineage's cultural memory pool."""
        if lineage_memory is None:
            return

        for agent in agents:
            if not agent.is_alive:
                continue
            # Only young agents study (age < 200 ticks)
            if agent.body.age > 200:
                continue
            # Cooldown check
            if hasattr(agent, '_lineage_study_cooldown') and agent._lineage_study_cooldown > 0:
                agent._lineage_study_cooldown -= 1
                continue

            # Study probability
            if rng.random() > 0.01:
                continue

            rule_data = lineage_memory.study(agent.lineage_id, rng)
            if rule_data is None:
                continue

            # Try to absorb the rule
            encoded = rule_data.get("encoded")
            if encoded is not None and hasattr(agent.hypotheses, 'decode_and_replace_worst'):
                agent.hypotheses.decode_and_replace_worst(encoded)
                self.total_lineage_studies += 1
                self._tick_events.append(CulturalEvent(
                    "lineage_study", rule_data.get("teacher_id", -1),
                    agent.id, "", tick
                ))

            if hasattr(agent, '_lineage_study_cooldown'):
                agent._lineage_study_cooldown = 50

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
                # Trust-weighted imitation: higher trust = stronger learning
                trust_weight = 1.0
                if hasattr(agent, 'trust_memory'):
                    trust_weight = 0.5 + agent.trust_memory.get_trust(best_neighbor.id)

                if hasattr(agent, 'composable') and hasattr(best_neighbor, 'composable'):
                    lr = 0.05 * trust_weight
                    for my_rule, their_rule in zip(agent.composable.rules,
                                                    best_neighbor.composable.rules):
                        n = min(len(my_rule.action_bias), len(their_rule.action_bias))
                        my_rule.action_bias[:n] += lr * (their_rule.action_bias[:n] - my_rule.action_bias[:n])

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
            "lineage_studies": sum(1 for e in self._tick_events if e.type == "lineage_study"),
            "total_teachings": self.total_teachings,
            "total_imitations": self.total_imitations,
            "total_lineage_studies": self.total_lineage_studies,
        }
