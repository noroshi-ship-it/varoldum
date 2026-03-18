
import json
import os
import time


class HallOfFameEntry:
    def __init__(self, agent_id, name, category, title, description,
                 generation, lineage_id, stats, tick_inducted, tick_born=0, tick_died=0):
        self.agent_id = agent_id
        self.name = name
        self.category = category
        self.title = title
        self.description = description
        self.generation = generation
        self.lineage_id = lineage_id
        self.stats = stats
        self.tick_inducted = tick_inducted
        self.tick_born = tick_born
        self.tick_died = tick_died

    def to_dict(self):
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "category": self.category,
            "title": self.title,
            "description": self.description,
            "generation": self.generation,
            "lineage_id": self.lineage_id,
            "stats": self.stats,
            "tick_inducted": self.tick_inducted,
            "tick_born": self.tick_born,
            "tick_died": self.tick_died,
        }


LEGENDARY_PREFIXES = [
    "Alpha", "Omega", "Nova", "Zen", "Flux", "Apex", "Echo", "Nyx",
    "Orion", "Vega", "Atlas", "Lyra", "Titan", "Solaris", "Nebula",
    "Chronos", "Aether", "Prism", "Cipher", "Axiom",
]

LEGENDARY_SUFFIXES = [
    "Prime", "Rex", "Sage", "Elder", "One", "Core", "Void",
    "Dawn", "Dusk", "Storm", "Flame", "Frost", "Wave", "Root",
]


def generate_legendary_name(agent_id, generation):
    prefix = LEGENDARY_PREFIXES[agent_id % len(LEGENDARY_PREFIXES)]
    suffix = LEGENDARY_SUFFIXES[generation % len(LEGENDARY_SUFFIXES)]
    return f"{prefix}-{suffix}"


class HallOfFame:

    def __init__(self, output_dir="output"):
        self.output_dir = output_dir
        self.entries: list[HallOfFameEntry] = []
        self._file = os.path.join(output_dir, "hall_of_fame.json")
        self._inducted_ids = set()

        self._longest_lived = []
        self._most_prolific = []
        self._best_world_model = []
        self._best_concept = []
        self._deepest_thinker = []
        self._language_pioneer_found = False
        self._dynasty_records = {}

        self._load()

    def _load(self):
        if os.path.exists(self._file):
            try:
                with open(self._file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                for e in data.get("entries", []):
                    entry = HallOfFameEntry(
                        e["agent_id"], e["name"], e["category"], e["title"],
                        e["description"], e["generation"], e["lineage_id"],
                        e.get("stats", {}), e["tick_inducted"],
                        e.get("tick_born", 0), e.get("tick_died", 0),
                    )
                    self.entries.append(entry)
                    self._inducted_ids.add(e["agent_id"])
                state = data.get("state", {})
                self._language_pioneer_found = state.get("language_pioneer_found", False)
            except (json.JSONDecodeError, KeyError):
                pass

    def check_dead_agent(self, agent, tick):
        aid = agent.id
        if aid in self._inducted_ids:
            return
        if agent.body.age < 100:
            return

        age = agent.body.age
        children = agent.children_count
        gen = agent.generation
        lid = agent.lineage_id
        wm_acc = agent.brain.cumulative_wm_accuracy
        think = agent.think_steps
        reward = agent.total_reward

        best_concept_acc = 0.0
        best_concept_desc = ""
        for h in agent.concept_hyp.hypotheses:
            if h.tests >= 50 and h.accuracy > best_concept_acc:
                best_concept_acc = h.accuracy
                best_concept_desc = h.describe(agent.bottleneck_size)

        agent_data = {
            "age": age, "children": children, "generation": gen,
            "lineage_id": lid, "wm_accuracy": round(float(wm_acc), 4),
            "think_steps": think, "total_reward": round(float(reward), 2),
            "bottleneck_size": agent.bottleneck_size,
            "best_concept_acc": round(float(best_concept_acc), 4),
            "best_concept_rule": best_concept_desc,
            "cause_of_death": getattr(agent.body, 'cause_of_death', 'unknown'),
        }

        name = generate_legendary_name(aid, gen)

        if age > 500:
            self._longest_lived.append((age, aid, agent_data, name, gen, lid, tick))
            self._longest_lived.sort(reverse=True)
            self._longest_lived = self._longest_lived[:5]
            if any(x[1] == aid for x in self._longest_lived):
                self._induct(aid, name, "longest_lived",
                    f"Long-Lived: {age} ticks",
                    f"{name} (Gen {gen}) survived for {age} ticks. "
                    f"World model: {wm_acc:.2%}, left {children} offspring.",
                    gen, lid, agent_data, tick, tick - age, tick)

        if children > 5:
            self._most_prolific.append((children, aid, agent_data, name, gen, lid, tick))
            self._most_prolific.sort(reverse=True)
            self._most_prolific = self._most_prolific[:5]
            if any(x[1] == aid for x in self._most_prolific):
                self._induct(aid, name, "most_prolific",
                    f"Prolific Ancestor: {children} offspring",
                    f"{name} (Gen {gen}) founded a strong lineage with {children} offspring.",
                    gen, lid, agent_data, tick, tick - age, tick)

        if wm_acc > 0.5:
            self._best_world_model.append((wm_acc, aid, agent_data, name, gen, lid, tick))
            self._best_world_model.sort(reverse=True)
            self._best_world_model = self._best_world_model[:5]
            if any(x[1] == aid for x in self._best_world_model):
                self._induct(aid, name, "world_modeler",
                    f"World Modeler: {wm_acc:.1%}",
                    f"{name} (Gen {gen}) predicted the world with {wm_acc:.1%} accuracy.",
                    gen, lid, agent_data, tick, tick - age, tick)

        if best_concept_acc > 0.9:
            self._best_concept.append((best_concept_acc, aid, agent_data, name, gen, lid, tick))
            self._best_concept.sort(reverse=True)
            self._best_concept = self._best_concept[:5]
            if any(x[1] == aid for x in self._best_concept):
                self._induct(aid, name, "concept_master",
                    f"Concept Master: {best_concept_acc:.0%}",
                    f"{name} (Gen {gen}) applied a self-discovered rule with "
                    f"{best_concept_acc:.0%} accuracy: {best_concept_desc}",
                    gen, lid, agent_data, tick, tick - age, tick)

        if think >= 4 and age >= 500:
            self._deepest_thinker.append((think, aid, agent_data, name, gen, lid, tick))
            self._deepest_thinker.sort(reverse=True)
            self._deepest_thinker = self._deepest_thinker[:5]
            if any(x[1] == aid for x in self._deepest_thinker):
                self._induct(aid, name, "deep_thinker",
                    f"Deep Thinker: {think} steps",
                    f"{name} (Gen {gen}) survived {age} ticks with {think}-step "
                    f"imagination depth.",
                    gen, lid, agent_data, tick, tick - age, tick)

    def check_language_pioneer(self, agent, signals_heard_count, tick):
        if self._language_pioneer_found:
            return
        if signals_heard_count >= 5:
            self._language_pioneer_found = True
            aid = agent.id
            gen = agent.generation
            lid = agent.lineage_id
            name = generate_legendary_name(aid, gen)
            self._induct(aid, name, "language_pioneer",
                "Language Pioneer",
                f"{name} (Gen {gen}) signals were heard by 5+ agents. "
                f"The dawn of proto-language.",
                gen, lid, {"signals_heard": signals_heard_count}, tick)

    def check_dynasty(self, lineage_id, count, founder_gen, tick):
        prev = self._dynasty_records.get(lineage_id, 0)
        if count > prev and count >= 20:
            self._dynasty_records[lineage_id] = count
            if count in (20, 50, 100, 200, 500):
                name = f"Lineage-{lineage_id}"
                self._induct(f"dynasty_{lineage_id}_{count}", name, "dynasty_founder",
                    f"Dynasty: Lineage {lineage_id} ({count} agents)",
                    f"Lineage {lineage_id} (founder gen {founder_gen}) "
                    f"reached {count} living agents.",
                    founder_gen, lineage_id,
                    {"lineage_size": count}, tick)

    def _induct(self, agent_id, name, category, title, description,
                generation, lineage_id, stats, tick, tick_born=0, tick_died=0):
        if agent_id in self._inducted_ids:
            return
        self._inducted_ids.add(agent_id)
        entry = HallOfFameEntry(
            agent_id, name, category, title, description,
            generation, lineage_id, stats, tick, tick_born, tick_died
        )
        self.entries.append(entry)
        self._save()
        print(f"  [HOF] {title}: {name}")

    def _save(self):
        os.makedirs(self.output_dir, exist_ok=True)
        data = {
            "entries": [e.to_dict() for e in self.entries],
            "state": {
                "language_pioneer_found": self._language_pioneer_found,
            },
            "updated": time.time(),
        }
        with open(self._file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def get_entries(self, category=None):
        if category:
            return [e.to_dict() for e in self.entries if e.category == category]
        return [e.to_dict() for e in self.entries]
