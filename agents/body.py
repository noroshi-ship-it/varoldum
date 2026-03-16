
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
        return self.energy > 0 and self.health > 0 and self.age < self.max_lifespan

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
