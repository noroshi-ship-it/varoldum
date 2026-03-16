
import json
import os
import numpy as np
from collections import defaultdict


class EventType:
    FIRST_CONCEPT_RULE = "first_concept_rule"
    CONCEPT_BREAKTHROUGH = "concept_breakthrough"
    FIRST_SIGNAL = "first_signal"
    SIGNAL_CONVERGENCE = "signal_convergence"
    MASS_EXTINCTION = "mass_extinction"
    POPULATION_BOOM = "population_boom"
    LINEAGE_DOMINANCE = "lineage_dominance"
    LINEAGE_EXTINCTION = "lineage_extinction"
    WORLD_MODEL_MILESTONE = "world_model_milestone"
    THINK_EVOLUTION = "think_evolution"
    STRUCTURE_CITY = "structure_city"
    NEW_GENERATION = "new_generation"
    FIRST_COMPLEX_RULE = "first_complex_rule"


SEVERITY = {
    EventType.FIRST_CONCEPT_RULE: "milestone",
    EventType.CONCEPT_BREAKTHROUGH: "milestone",
    EventType.FIRST_SIGNAL: "milestone",
    EventType.SIGNAL_CONVERGENCE: "milestone",
    EventType.MASS_EXTINCTION: "crisis",
    EventType.POPULATION_BOOM: "growth",
    EventType.LINEAGE_DOMINANCE: "social",
    EventType.LINEAGE_EXTINCTION: "crisis",
    EventType.WORLD_MODEL_MILESTONE: "milestone",
    EventType.THINK_EVOLUTION: "evolution",
    EventType.STRUCTURE_CITY: "social",
    EventType.NEW_GENERATION: "evolution",
    EventType.FIRST_COMPLEX_RULE: "milestone",
}

ICONS = {
    EventType.FIRST_CONCEPT_RULE: "brain",
    EventType.CONCEPT_BREAKTHROUGH: "star",
    EventType.FIRST_SIGNAL: "speech",
    EventType.SIGNAL_CONVERGENCE: "language",
    EventType.MASS_EXTINCTION: "skull",
    EventType.POPULATION_BOOM: "rocket",
    EventType.LINEAGE_DOMINANCE: "crown",
    EventType.LINEAGE_EXTINCTION: "tombstone",
    EventType.WORLD_MODEL_MILESTONE: "eye",
    EventType.THINK_EVOLUTION: "thought",
    EventType.STRUCTURE_CITY: "city",
    EventType.NEW_GENERATION: "dna",
    EventType.FIRST_COMPLEX_RULE: "puzzle",
}


class Event:
    def __init__(self, tick: int, event_type: str, title: str, description: str,
                 data: dict = None):
        self.tick = tick
        self.event_type = event_type
        self.title = title
        self.description = description
        self.severity = SEVERITY.get(event_type, "info")
        self.icon = ICONS.get(event_type, "info")
        self.data = data or {}

    def to_dict(self):
        return {
            "tick": self.tick,
            "type": self.event_type,
            "severity": self.severity,
            "icon": self.icon,
            "title": self.title,
            "description": self.description,
            "data": self.data,
        }


class EventDetector:

    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        self.events: list[Event] = []
        self._events_file = os.path.join(output_dir, "events.json")

        self._prev_pop = 0
        self._prev_think_mean = 0.0
        self._max_generation = 0
        self._seen_first_concept = False
        self._seen_first_signal = False
        self._seen_first_complex = False
        self._seen_wm_90 = False
        self._known_lineages = set()
        self._lineage_counts = defaultdict(int)
        self._prev_lineage_counts = defaultdict(int)
        self._concept_breakthroughs = set()

        if os.path.exists(self._events_file):
            try:
                with open(self._events_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for e in data.get("events", []):
                        self.events.append(Event(
                            e["tick"], e["type"], e["title"],
                            e["description"], e.get("data", {})
                        ))
                    state = data.get("state", {})
                    self._seen_first_concept = state.get("seen_first_concept", False)
                    self._seen_first_signal = state.get("seen_first_signal", False)
                    self._seen_first_complex = state.get("seen_first_complex", False)
                    self._seen_wm_90 = state.get("seen_wm_90", False)
                    self._max_generation = state.get("max_generation", 0)
            except (json.JSONDecodeError, KeyError):
                pass

    def check(self, tick: int, agents, structures=None, lang_stats=None):
        if not agents:
            return

        n = len(agents)

        if self._prev_pop > 0:
            if n < self._prev_pop * 0.5:
                self._add(Event(tick, EventType.MASS_EXTINCTION,
                    "Mass Extinction",
                    f"Population crashed from {self._prev_pop} to {n} ({(1-n/self._prev_pop)*100:.0f}% loss)",
                    {"from": self._prev_pop, "to": n}))
            elif n > self._prev_pop * 2 and self._prev_pop > 10:
                self._add(Event(tick, EventType.POPULATION_BOOM,
                    "Population Boom",
                    f"Population doubled from {self._prev_pop} to {n}",
                    {"from": self._prev_pop, "to": n}))
        self._prev_pop = n

        max_gen = max(a.generation for a in agents)
        if max_gen > self._max_generation and max_gen > 0:
            self._max_generation = max_gen
            if max_gen in (1, 5, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000):
                self._add(Event(tick, EventType.NEW_GENERATION,
                    f"Generation {max_gen} Reached",
                    f"Evolution has produced generation {max_gen}",
                    {"generation": max_gen}))

        for a in agents:
            for h in a.concept_hyp.hypotheses:
                if not self._seen_first_concept and h.tests >= 10 and h.accuracy > 0.8:
                    self._seen_first_concept = True
                    desc = h.describe(a.bottleneck_size)
                    self._add(Event(tick, EventType.FIRST_CONCEPT_RULE,
                        "First Concept Rule Discovered",
                        f"Agent #{a.id} (gen {a.generation}) discovered: {desc}",
                        {"agent_id": a.id, "rule": desc, "accuracy": h.accuracy}))

                if h.tests >= 50 and h.accuracy > 0.95:
                    key = (a.lineage_id, h.outcome)
                    if key not in self._concept_breakthroughs:
                        self._concept_breakthroughs.add(key)
                        if len(self._concept_breakthroughs) <= 3 or len(self._concept_breakthroughs) in (10, 25, 50, 100):
                            desc = h.describe(a.bottleneck_size)
                            self._add(Event(tick, EventType.CONCEPT_BREAKTHROUGH,
                                "Concept Breakthrough",
                                f"Rule #{len(self._concept_breakthroughs)}: {desc} by lineage {a.lineage_id}",
                                {"rule": desc, "accuracy": h.accuracy, "tests": h.tests}))

        if not self._seen_wm_90:
            for a in agents:
                if a.brain.cumulative_wm_accuracy > 0.90:
                    self._seen_wm_90 = True
                    self._add(Event(tick, EventType.WORLD_MODEL_MILESTONE,
                        "World Model >90% Accuracy",
                        f"Agent #{a.id} can predict the world with >90% accuracy. "
                        f"Bottleneck: {a.bottleneck_size} concepts, Think: {a.think_steps} steps",
                        {"agent_id": a.id, "accuracy": a.brain.cumulative_wm_accuracy,
                         "bottleneck": a.bottleneck_size}))
                    break

        mean_think = sum(a.think_steps for a in agents) / n
        if abs(mean_think - self._prev_think_mean) > 1.0 and self._prev_think_mean > 0:
            direction = "increased" if mean_think > self._prev_think_mean else "decreased"
            self._add(Event(tick, EventType.THINK_EVOLUTION,
                f"Thinking {direction.title()}",
                f"Average think steps {direction} from {self._prev_think_mean:.1f} to {mean_think:.1f}",
                {"from": self._prev_think_mean, "to": mean_think}))
        self._prev_think_mean = mean_think

        current_lineages = defaultdict(int)
        for a in agents:
            current_lineages[a.lineage_id] += 1

        if n > 20:
            for lid, count in current_lineages.items():
                if count > n * 0.6:
                    self._add(Event(tick, EventType.LINEAGE_DOMINANCE,
                        f"Lineage {lid} Dominates",
                        f"Lineage {lid} has {count}/{n} agents ({count/n*100:.0f}%)",
                        {"lineage_id": lid, "count": count, "percentage": count/n}))

        for lid, prev_count in self._prev_lineage_counts.items():
            if prev_count >= 10 and current_lineages.get(lid, 0) == 0:
                self._add(Event(tick, EventType.LINEAGE_EXTINCTION,
                    f"Lineage {lid} Extinct",
                    f"Lineage {lid} (had {prev_count} agents) has gone extinct",
                    {"lineage_id": lid, "previous_count": prev_count}))

        self._prev_lineage_counts = dict(current_lineages)

        if lang_stats and not self._seen_first_signal:
            if lang_stats.get("signals_heard", 0) > 5:
                self._seen_first_signal = True
                self._add(Event(tick, EventType.FIRST_SIGNAL,
                    "First Proto-Language Signals",
                    f"Agents are broadcasting and hearing signals. "
                    f"{lang_stats['signals_sent']} sent, {lang_stats['signals_heard']} heard",
                    lang_stats))

        if not self._seen_first_complex:
            for a in agents:
                cs = a.composable.stats
                if cs.get("max_complexity", 0) >= 4:
                    self._seen_first_complex = True
                    best = a.composable.get_best_rules(min_tests=5, min_accuracy=0.5)
                    desc = best[0].describe() if best else "complex rule"
                    self._add(Event(tick, EventType.FIRST_COMPLEX_RULE,
                        "First Complex Rule (4+ conditions)",
                        f"Agent #{a.id} evolved: {desc}",
                        {"agent_id": a.id, "complexity": cs["max_complexity"]}))
                    break

        if structures:
            self._check_structure_clusters(tick, structures)

    def _check_structure_clusters(self, tick, structures):
        if not hasattr(structures, 'structures'):
            return
        density = defaultdict(int)
        for (x, y), s in structures.structures.items():
            grid_x = x // 10
            grid_y = y // 10
            density[(grid_x, grid_y)] += 1

        for cell, count in density.items():
            if count >= 50:
                self._add(Event(tick, EventType.STRUCTURE_CITY,
                    "Structure City Formed",
                    f"Dense cluster of {count} structures at grid ({cell[0]*10},{cell[1]*10})",
                    {"count": count, "x": cell[0]*10, "y": cell[1]*10}))

    def _add(self, event: Event):
        self.events.append(event)
        self._save()
        severity_prefix = {"milestone": "[*]", "crisis": "[!]", "growth": "[+]",
                          "social": "[~]", "evolution": "[^]"}.get(event.severity, "[i]")
        print(f"  {severity_prefix} EVENT t={event.tick}: {event.title}")

    def _save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        data = {
            "events": [e.to_dict() for e in self.events],
            "state": {
                "seen_first_concept": self._seen_first_concept,
                "seen_first_signal": self._seen_first_signal,
                "seen_first_complex": self._seen_first_complex,
                "seen_wm_90": self._seen_wm_90,
                "max_generation": self._max_generation,
            }
        }
        with open(self._events_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_events_since(self, tick: int = 0) -> list[dict]:
        return [e.to_dict() for e in self.events if e.tick >= tick]

    def get_all_events(self) -> list[dict]:
        return [e.to_dict() for e in self.events]
