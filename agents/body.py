
import numpy as np
from config import Config
from agents.genome import get_trait

MAX_BODY_SUBSTANCES = 64

# Property indices in substance_props
_NUTRITIVE = 3
_STABILITY = 5


class Body:
    def __init__(self, genome: np.ndarray, cfg: Config, position: np.ndarray,
                 substance_props=None):
        self.position = position.copy().astype(np.float64)
        self.heading = np.random.uniform(-np.pi, np.pi)
        self.age = 0

        self.vx = 0.0
        self.vy = 0.0
        self.friction = 0.7

        self.size = get_trait(genome, "body_size")
        self.max_lifespan = int(cfg.max_lifespan * get_trait(genome, "max_lifespan_factor"))
        self.base_metabolic = cfg.base_metabolic_rate * self.size

        self.mineral_carried = 0.0
        self.max_mineral = 0.5

        # Natural mortality state
        self.infection_level = 0.0
        self.genetic_frailty = 0.0
        self.injury_level = 0.0
        self.cause_of_death = ""

        # === Substance composition ===
        # Body IS substances. Energy and health are derived, not stored.
        self._substance_props = substance_props  # (MAX, 6) or None
        self.composition = np.zeros(MAX_BODY_SUBSTANCES, dtype=np.float32)

        if substance_props is not None:
            n = min(len(substance_props), MAX_BODY_SUBSTANCES)
            # Weight vectors: how much each substance contributes to energy/health
            self._nutr = np.zeros(MAX_BODY_SUBSTANCES, dtype=np.float32)
            self._stab = np.zeros(MAX_BODY_SUBSTANCES, dtype=np.float32)
            self._nutr[:n] = substance_props[:n, _NUTRITIVE]
            self._stab[:n] = substance_props[:n, _STABILITY]
            # Pre-sorted consumption order (highest weight first)
            self._energy_consume_order = np.argsort(-self._nutr)
            self._health_consume_order = np.argsort(-self._stab)
        else:
            # Fallback: uniform weights
            self._nutr = np.full(MAX_BODY_SUBSTANCES, 0.3, dtype=np.float32)
            self._stab = np.full(MAX_BODY_SUBSTANCES, 0.5, dtype=np.float32)
            self._energy_consume_order = np.arange(MAX_BODY_SUBSTANCES)
            self._health_consume_order = np.arange(MAX_BODY_SUBSTANCES)

        # Initialize composition so energy=0.6, health=1.0
        self._init_composition(0.6, 1.0)

    def _init_composition(self, target_energy, target_health):
        """Set initial composition so derived energy and health match targets."""
        nutr = self._nutr
        stab = self._stab

        # Add nutritive substances for energy
        nutr_mask = nutr > 0.1
        if np.any(nutr_mask):
            # mass_per_sub * nutr[i] summed = target_energy
            total_nutr = float(np.sum(nutr[nutr_mask]))
            mass_per = target_energy / total_nutr
            self.composition[nutr_mask] += mass_per

        # Add structural substances for health
        stab_mask = stab > 0.3
        if np.any(stab_mask):
            total_stab = float(np.sum(stab[stab_mask]))
            mass_per = target_health / total_stab
            self.composition[stab_mask] += mass_per

    @property
    def energy(self) -> float:
        """Energy = dot(composition, nutritive_weights). Truly derived."""
        return float(np.clip(np.dot(self.composition, self._nutr), 0.0, 1.0))

    @energy.setter
    def energy(self, value):
        """Modify composition to achieve target energy."""
        value = float(np.clip(value, 0.0, 1.0))
        current = float(np.dot(self.composition, self._nutr))
        delta = value - current
        if abs(delta) < 1e-8:
            return

        if delta < 0:
            # Consume energy: remove nutritive substances, highest-nutritive first
            to_consume = -delta
            for idx in self._energy_consume_order:
                idx = int(idx)
                if self._nutr[idx] < 0.01 or self.composition[idx] < 1e-6:
                    continue
                available = self.composition[idx] * self._nutr[idx]
                take = min(to_consume, available)
                self.composition[idx] -= take / self._nutr[idx]
                to_consume -= take
                if to_consume < 1e-8:
                    break
        else:
            # Gain energy: add to highest-nutritive substance
            for idx in self._energy_consume_order:
                idx = int(idx)
                if self._nutr[idx] > 0.1:
                    self.composition[idx] += delta / self._nutr[idx]
                    break

    @property
    def health(self) -> float:
        """Health = dot(composition, stability_weights). Truly derived."""
        return float(np.clip(np.dot(self.composition, self._stab), 0.0, 1.0))

    @health.setter
    def health(self, value):
        """Modify composition to achieve target health."""
        value = float(np.clip(value, 0.0, 1.0))
        current = float(np.dot(self.composition, self._stab))
        delta = value - current
        if abs(delta) < 1e-8:
            return

        if delta < 0:
            # Damage: remove structural substances, highest-stability first
            to_consume = -delta
            for idx in self._health_consume_order:
                idx = int(idx)
                if self._stab[idx] < 0.01 or self.composition[idx] < 1e-6:
                    continue
                available = self.composition[idx] * self._stab[idx]
                take = min(to_consume, available)
                self.composition[idx] -= take / self._stab[idx]
                to_consume -= take
                if to_consume < 1e-8:
                    break
        else:
            # Heal: add to highest-stability substance
            for idx in self._health_consume_order:
                idx = int(idx)
                if self._stab[idx] > 0.1:
                    self.composition[idx] += delta / self._stab[idx]
                    break

    @property
    def x(self) -> int:
        return int(self.position[0])

    @property
    def y(self) -> int:
        return int(self.position[1])

    @property
    def speed(self) -> float:
        return np.sqrt(self.vx ** 2 + self.vy ** 2)

    def metabolize(self, brain_cost: float, temp_modifier: float = 1.0):
        cost = (self.base_metabolic + brain_cost) * temp_modifier
        # Consume actual nutritive substances
        self.energy = self.energy - cost
        self.age += 1

    def take_damage(self, amount: float):
        # Degrade actual structural substances
        self.health = max(0.0, self.health - amount)

    def heal(self, amount: float = 0.001):
        if self.energy > 0.3:
            self.health = min(1.0, self.health + amount)

    def heal_medicine(self, amount: float):
        self.health = min(1.0, self.health + amount)
        self.mineral_carried = max(0, self.mineral_carried - 0.1)

    def is_alive(self) -> bool:
        if self.energy <= 0:
            self.cause_of_death = self.cause_of_death or "starvation"
            return False
        if self.health <= 0:
            self.cause_of_death = self.cause_of_death or "damage"
            return False
        if self.age >= self.max_lifespan:
            self.cause_of_death = self.cause_of_death or "old_age"
            return False
        return True

    def update_natural_mortality(self, rng):
        """Stochastic natural death causes — called every tick."""
        dmg = 0.0

        # === INFECTION ===
        if self.infection_level <= 0:
            age_factor = self.age / max(1, self.max_lifespan)
            weakness = max(0, 1.0 - self.health)
            infection_chance = 0.0005 + 0.002 * age_factor + 0.003 * weakness
            if rng.random() < infection_chance:
                self.infection_level = rng.uniform(0.05, 0.3)

        if self.infection_level > 0:
            inf_dmg = self.infection_level * 0.02
            self.take_damage(inf_dmg)
            self.energy = self.energy - self.infection_level * 0.01
            dmg += inf_dmg
            if rng.random() < 0.02 * self.health:
                self.infection_level *= 0.7
            if self.infection_level < 0.01:
                self.infection_level = 0.0
            if rng.random() < 0.005:
                self.infection_level = min(1.0, self.infection_level * 1.3)
            if self.health <= 0:
                self.cause_of_death = "infection"

        # === GENETIC FRAILTY ===
        if self.genetic_frailty > 0:
            frail_dmg = self.genetic_frailty * 0.005
            self.take_damage(frail_dmg)
            self.energy = self.energy - self.genetic_frailty * 0.003
            dmg += frail_dmg
            if self.health <= 0:
                self.cause_of_death = "genetic_defect"

        # === RANDOM ACCIDENTS ===
        age_factor = self.age / max(1, self.max_lifespan)
        speed_factor = self.speed / max(0.01, 1.5 / self.size)
        accident_chance = 0.0003 + 0.001 * speed_factor + 0.0005 * age_factor
        if rng.random() < accident_chance:
            severity = rng.uniform(0.02, 0.15)
            self.take_damage(severity)
            self.injury_level = min(1.0, self.injury_level + severity)
            dmg += severity
            if self.health <= 0:
                self.cause_of_death = "accident"

        # Injuries heal slowly
        if self.injury_level > 0 and self.energy > 0.3:
            self.injury_level = max(0, self.injury_level - 0.002)

        # === AGE-RELATED DECLINE ===
        if age_factor > 0.4:
            decline_rate = (age_factor - 0.4) * 0.025
            self.take_damage(decline_rate)
            dmg += decline_rate
            if self.health <= 0:
                self.cause_of_death = "old_age"

        return dmg

    def init_genetic_frailty(self, rng):
        """Called at birth. Most agents are healthy, some have defects."""
        if rng.random() < 0.08:
            self.genetic_frailty = rng.uniform(0.05, 0.5)
        else:
            self.genetic_frailty = 0.0

    def move(self, dx: float, dy: float, w: int, h: int):
        inertia = self.size
        self.vx += dx / inertia
        self.vy += dy / inertia
        speed = self.speed
        max_speed = 1.5 / self.size
        if speed > max_speed:
            self.vx *= max_speed / speed
            self.vy *= max_speed / speed
        self.position[0] = (self.position[0] + self.vx) % w
        self.position[1] = (self.position[1] + self.vy) % h
        self.vx *= self.friction
        self.vy *= self.friction
        move_cost = 0.001 * (abs(dx) + abs(dy)) * self.size
        self.energy = self.energy - move_cost
        if abs(self.vx) > 0.01 or abs(self.vy) > 0.01:
            self.heading = np.arctan2(self.vy, self.vx)

    def eat(self, amount: float):
        """Generic eating: adds energy via the most nutritive substance."""
        self.energy = min(1.0, self.energy + amount * 0.8)

    def eat_substance(self, substance_idx, amount, nutritive_value=None):
        """Eat a specific substance — adds directly to body composition."""
        if 0 <= substance_idx < MAX_BODY_SUBSTANCES:
            self.composition[substance_idx] += amount * 0.8
            # Prevent unbounded growth
            np.clip(self.composition, 0, 5.0, out=self.composition)

    def collect_mineral(self, amount: float) -> float:
        space = self.max_mineral - self.mineral_carried
        collected = min(amount, space)
        self.mineral_carried += collected
        return collected

    def decompose(self) -> np.ndarray:
        """Return body composition for deposit back to the world on death."""
        result = self.composition.copy()
        self.composition[:] = 0
        return result

    def get_proprioception(self) -> np.ndarray:
        return np.array([
            self.energy,
            self.health,
            np.sin(self.heading),
            np.cos(self.heading),
            self.age / max(1, self.max_lifespan),
            self.mineral_carried / max(0.01, self.max_mineral),
            self.speed,
        ])
