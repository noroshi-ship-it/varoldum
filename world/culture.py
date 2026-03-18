
import numpy as np
from agents.genome import get_trait


class CulturalEvent:
    __slots__ = ['type', 'teacher_id', 'student_id', 'rule_desc', 'tick']

    def __init__(self, etype: str, teacher: int, student: int, desc: str, tick: int):
        self.type = etype
        self.teacher_id = teacher
        self.student_id = student
        self.rule_desc = desc
        self.tick = tick


class CultureSystem:
    """
    Brain-driven cultural transmission.
    No hardcoded probabilities — the agent's signal output drives teaching.
    No age cutoffs — youth factor is continuous.
    No lineage bonuses — brain sees lineage as input, decides itself.
    """

    def __init__(self):
        self.events: list[CulturalEvent] = []
        self._tick_events: list[CulturalEvent] = []
        self.total_teachings = 0
        self.total_imitations = 0
        self.total_lineage_studies = 0

    def process_teaching(self, agents: list, tick: int, rng: np.random.Generator,
                         lineage_memory=None, actions: list = None):
        self._tick_events = []

        alive = [a for a in agents if a.is_alive]
        if not alive:
            return

        # Build action lookup for brain-driven teaching
        action_map = {}
        if actions is not None:
            for agent, action in zip(agents, actions):
                action_map[agent.id] = action

        for teacher in alive:
            # Teacher's signal output drives teaching intent
            teacher_action = action_map.get(teacher.id, {})
            teacher_signal = teacher_action.get("signal", 0)
            if teacher_signal <= 0:
                continue  # brain says don't broadcast

            best_rules = teacher.hypotheses.get_best_hypotheses(
                min_tests=5, min_accuracy=0.5
            )
            if not best_rules:
                continue

            for student in alive:
                if student.id == teacher.id:
                    continue

                dx = abs(teacher.body.position[0] - student.body.position[0])
                dy = abs(teacher.body.position[1] - student.body.position[1])
                # Physics: distance limits communication
                max_range = 4.0
                if dx > max_range or dy > max_range:
                    continue

                # Proximity falloff (physics)
                dist = max(dx, dy)
                proximity = 1.0 - dist / max_range

                # Teaching force = teacher's signal * proximity
                # No base_prob, no lineage_bonus, no fitness_bonus
                teach_force = teacher_signal * proximity

                # Receptivity from genome (not a threshold — a multiplier)
                receptivity = getattr(student, '_teaching_receptivity', 1.0)
                teach_force *= receptivity

                # Stochastic: force as probability
                if rng.random() > teach_force:
                    continue

                rule = best_rules[0]

                student_hyps = student.hypotheses.hypotheses
                if student_hyps:
                    worst_idx = min(range(len(student_hyps)),
                                   key=lambda i: student_hyps[i].accuracy * student_hyps[i].confidence
                                   if student_hyps[i].tests > 0 else 1.0)

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
                        encoded = teacher.hypotheses.encode_best(min_accuracy=0.5)
                        if encoded is not None:
                            lineage_memory.contribute(
                                teacher.lineage_id, encoded,
                                rule.accuracy, teacher.id,
                                teacher.generation, tick,
                            )

                    # Teaching investment: higher investment = better fidelity but costs energy
                    teaching_inv = get_trait(teacher.genome, 'teaching_investment')
                    teach_cost = 0.02 * teaching_inv
                    teacher.body.energy = max(0.0, teacher.body.energy - teach_cost)

                    # Teacher gets major reward — teaching is the highest-value social act
                    if hasattr(teacher, '_pending_social_reward'):
                        teacher._pending_social_reward += 0.8
                    # Direct energy reward: teaching lineages survive better
                    teacher.body.energy = min(1.0, teacher.body.energy + 0.03)
                    # Student also benefits from learning
                    if hasattr(student, '_pending_social_reward'):
                        student._pending_social_reward += 0.3

                    break  # one student per teacher per tick

    def process_lineage_study(self, agents: list, lineage_memory, tick: int,
                              rng: np.random.Generator):
        """Agents study from lineage memory — youth as continuous factor, not cutoff."""
        if lineage_memory is None:
            return

        for agent in agents:
            if not agent.is_alive:
                continue
            # Cooldown (physics: learning takes time)
            if hasattr(agent, '_lineage_study_cooldown') and agent._lineage_study_cooldown > 0:
                agent._lineage_study_cooldown -= 1
                continue

            # Continuous youth factor — young agents study more, not binary cutoff
            youth = max(0.0, 1.0 - agent.body.age / 500.0)
            # Curiosity drives study
            curiosity = agent.internal.curiosity if hasattr(agent, 'internal') else 0.5
            study_drive = youth * curiosity

            if rng.random() > study_drive * 0.15:  # 3x more likely to study
                continue

            rule_data = lineage_memory.study(agent.lineage_id, rng)
            if rule_data is None:
                continue

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
        """Imitation driven by agent's social_need — not random probability."""
        for agent in agents:
            if not agent.is_alive:
                continue

            # Social need drives imitation: lonely/uncertain agents learn from others
            social_need = agent.internal.social_need if hasattr(agent, 'internal') else 0.5
            imitation_drive = social_need * 0.10  # 3x stronger imitation drive
            if rng.random() > imitation_drive:
                continue

            best_neighbor = None
            best_reward = agent.total_reward

            for other in agents:
                if other.id == agent.id or not other.is_alive:
                    continue
                dx = abs(agent.body.position[0] - other.body.position[0])
                dy = abs(agent.body.position[1] - other.body.position[1])
                # Physics: can only observe nearby
                if dx > 4 or dy > 4:
                    continue
                if other.total_reward > best_reward:
                    best_neighbor = other
                    best_reward = other.total_reward

            if best_neighbor is not None:
                if hasattr(agent, 'composable') and hasattr(best_neighbor, 'composable'):
                    lr = 0.05
                    for my_rule, their_rule in zip(agent.composable.rules,
                                                    best_neighbor.composable.rules):
                        n = min(len(my_rule.action_bias), len(their_rule.action_bias))
                        my_rule.action_bias[:n] += lr * (their_rule.action_bias[:n] - my_rule.action_bias[:n])

                self.total_imitations += 1
                self._tick_events.append(CulturalEvent(
                    "imitate", best_neighbor.id, agent.id, "", tick
                ))

    def process_structure_learning(self, agents: list, structure_manager,
                                   tick: int, rng: np.random.Generator):
        """Agents at MARKER/STORAGE structures can absorb inscribed rules."""
        from world.structures import StructureType

        for agent in agents:
            if not agent.is_alive:
                continue

            cultural_receptivity = get_trait(agent.genome, 'cultural_receptivity')
            if cultural_receptivity < 0.1:
                continue

            # Only check every 10 ticks to save performance
            if (agent.id + tick) % 10 != 0:
                continue

            ax, ay = int(agent.x), int(agent.y)
            s = structure_manager.get_at(ax, ay)
            if s is None:
                continue
            if s.stype not in (StructureType.MARKER, StructureType.STORAGE):
                continue
            if not s.inscriptions:
                continue

            # Try to learn from a random inscription with rule_data
            inscriptions_with_rules = [insc for insc in s.inscriptions if insc.rule_data is not None]
            if not inscriptions_with_rules:
                continue

            insc = rng.choice(inscriptions_with_rules)
            success_chance = cultural_receptivity * (insc.use_count * 0.05 + 0.5) * 0.1

            if rng.random() < success_chance:
                if hasattr(agent.hypotheses, 'decode_and_replace_worst'):
                    agent.hypotheses.decode_and_replace_worst(insc.rule_data)
                    insc.use_count += 1
                    self._tick_events.append(CulturalEvent(
                        "structure_learn", insc.author_lineage,
                        agent.id, "", tick
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
