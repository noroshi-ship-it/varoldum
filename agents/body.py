
import numpy as np
from config import Config
from agents.genome import get_trait


class Body:
    def __init__(self, genome: np.ndarray, cfg: Config, position: np.ndarray):
        self.position = position.copy().astype(np.float64)
        self.heading = np.random.uniform(-np.pi, np.pi)
        self.energy = 0.6
        self.health = 1.0
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
        self.infection_level = 0.0      # chronic infection [0-1]
        self.genetic_frailty = 0.0      # birth defect severity [0-1]
        self.injury_level = 0.0         # accumulated accident injuries [0-1]
        self.cause_of_death = ""        # what killed this agent

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
        self.energy -= cost
        self.age += 1

    def take_damage(self, amount: float):
        self.health -= amount
        self.health = max(0.0, self.health)

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
        """Stochastic natural death causes — called every tick.

        Returns damage dealt this tick from natural causes.
        """
        dmg = 0.0

        # === INFECTION: chronic disease, worse when health is low ===
        # Chance of catching infection increases with low health and age
        if self.infection_level <= 0:
            age_factor = self.age / max(1, self.max_lifespan)
            weakness = max(0, 1.0 - self.health)
            infection_chance = 0.0005 + 0.002 * age_factor + 0.003 * weakness
            if rng.random() < infection_chance:
                self.infection_level = rng.uniform(0.05, 0.3)

        if self.infection_level > 0:
            # Infection drains health and energy
            inf_dmg = self.infection_level * 0.02
            self.take_damage(inf_dmg)
            self.energy -= self.infection_level * 0.01
            dmg += inf_dmg
            # Recovery chance — better health = faster recovery
            if rng.random() < 0.02 * self.health:
                self.infection_level *= 0.7
            if self.infection_level < 0.01:
                self.infection_level = 0.0
            # Infection can worsen
            if rng.random() < 0.005:
                self.infection_level = min(1.0, self.infection_level * 1.3)
            if self.health <= 0:
                self.cause_of_death = "infection"

        # === GENETIC FRAILTY: birth defects, permanent weakness ===
        # Already set at birth (see init_genetic_frailty), just applies damage
        if self.genetic_frailty > 0:
            frail_dmg = self.genetic_frailty * 0.005
            self.take_damage(frail_dmg)
            self.energy -= self.genetic_frailty * 0.003
            dmg += frail_dmg
            if self.health <= 0:
                self.cause_of_death = "genetic_defect"

        # === RANDOM ACCIDENTS: falling, collision, bad luck ===
        age_factor = self.age / max(1, self.max_lifespan)
        speed_factor = self.speed / max(0.01, 1.5 / self.size)
        # Faster movement + old age = more accident prone
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

        # === AGE-RELATED DECLINE: organs fail with age ===
        if age_factor > 0.4:
            decline_rate = (age_factor - 0.4) * 0.025  # starts earlier, hits harder
            self.take_damage(decline_rate)
            dmg += decline_rate
            if self.health <= 0:
                self.cause_of_death = "old_age"

        return dmg

    def init_genetic_frailty(self, rng):
        """Called at birth. Most agents are healthy, some have defects."""
        if rng.random() < 0.08:  # 8% chance of birth defect
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
        self.energy -= move_cost
        if abs(self.vx) > 0.01 or abs(self.vy) > 0.01:
            self.heading = np.arctan2(self.vy, self.vx)

    def eat(self, amount: float):
        self.energy = min(1.0, self.energy + amount * 0.8)

    def collect_mineral(self, amount: float) -> float:
        space = self.max_mineral - self.mineral_carried
        collected = min(amount, space)
        self.mineral_carried += collected
        return collected

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
