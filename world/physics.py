
import numpy as np
from config import Config


CH_TEMPERATURE = 4
CH_RESOURCE_B = 5
CH_RESOURCE_C = 6
CH_SCENT = 7
CH_FERTILITY = 8
NUM_PHYSICS_CHANNELS = 9


class ChemistryRule:
    def __init__(self, input_a: int, input_b: int, output: int,
                 threshold_a: float, threshold_b: float, output_amount: float,
                 name: str):
        self.input_a = input_a
        self.input_b = input_b
        self.output = output
        self.threshold_a = threshold_a
        self.threshold_b = threshold_b
        self.output_amount = output_amount
        self.name = name


CHEMISTRY_RULES = [
    ChemistryRule(0, 5, -1, 0.3, 0.3, 0.2, "food+mineral=medicine"),
    ChemistryRule(5, 6, 8, 0.2, 0.2, 0.4, "mineral+toxin=fertilizer"),
    ChemistryRule(4, 0, -2, 0.7, 0.5, 0.1, "heat+food=spoilage"),
]


class Physics:

    def __init__(self, cfg: Config, rng: np.random.Generator):
        self._cfg = cfg
        self._rng = rng
        self.w = cfg.world_width
        self.h = cfg.world_height

        self.temperature = np.zeros((self.w, self.h))
        self.mineral = np.zeros((self.w, self.h))
        self.toxin = np.zeros((self.w, self.h))
        self.scent = np.zeros((self.w, self.h))
        self.fertility = np.ones((self.w, self.h)) * 0.5

        self._init_temperature()
        self._init_minerals()
        self._init_toxins()

    def _init_temperature(self):
        for y in range(self.h):
            lat = abs(y - self.h / 2) / (self.h / 2)
            self.temperature[:, y] = 0.7 - 0.5 * lat
        self.temperature += self._rng.normal(0, 0.05, (self.w, self.h))
        np.clip(self.temperature, 0, 1, out=self.temperature)

    def _init_minerals(self):
        n_deposits = max(1, (self.w * self.h) // 300)
        for _ in range(n_deposits):
            cx, cy = self._rng.integers(0, self.w), self._rng.integers(0, self.h)
            r = self._rng.integers(2, 6)
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        x, y = (cx + dx) % self.w, (cy + dy) % self.h
                        self.mineral[x, y] = min(1.0, self.mineral[x, y] + self._rng.uniform(0.2, 0.6))

    def _init_toxins(self):
        n_zones = max(1, (self.w * self.h) // 500)
        for _ in range(n_zones):
            cx, cy = self._rng.integers(0, self.w), self._rng.integers(0, self.h)
            r = self._rng.integers(2, 5)
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        x, y = (cx + dx) % self.w, (cy + dy) % self.h
                        self.toxin[x, y] = min(1.0, self.toxin[x, y] + self._rng.uniform(0.3, 0.7))

    def update(self, tick: int, season_phase: float, grid_resources: np.ndarray):
        self._update_temperature(season_phase)
        self._update_chemistry(grid_resources)
        self._update_scent()
        self._update_fertility(grid_resources)
        self._update_toxin_spread()

    def _update_temperature(self, season_phase: float):
        season_offset = 0.2 * np.sin(2 * np.pi * season_phase)
        base = np.zeros((self.w, self.h))
        for y in range(self.h):
            lat = abs(y - self.h / 2) / (self.h / 2)
            base[:, y] = 0.7 - 0.5 * lat + season_offset
        self.temperature += 0.01 * (base - self.temperature)
        self.temperature += self._rng.normal(0, 0.002, (self.w, self.h))
        np.clip(self.temperature, 0, 1, out=self.temperature)

    def _update_chemistry(self, grid_resources: np.ndarray):
        food = grid_resources[:, :, 0]

        hot_mask = self.temperature > 0.7
        food_mask = food > 0.3
        spoil_mask = hot_mask & food_mask
        grid_resources[:, :, 0] -= spoil_mask * 0.005

        react_mask = (self.mineral > 0.2) & (self.toxin > 0.2)
        react_amount = np.minimum(self.mineral, self.toxin) * 0.02
        self.fertility += react_mask * react_amount * 2
        self.mineral -= react_mask * react_amount
        self.toxin -= react_mask * react_amount

        np.clip(self.fertility, 0, 2, out=self.fertility)
        np.clip(self.mineral, 0, 1, out=self.mineral)
        np.clip(self.toxin, 0, 1, out=self.toxin)
        np.clip(grid_resources[:, :, 0], 0, 1, out=grid_resources[:, :, 0])

    def _update_scent(self):
        self.scent *= 0.92
        padded = np.pad(self.scent, 1, mode='wrap')
        neighbors = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                     padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
        self.scent += 0.05 * (neighbors - self.scent)
        np.clip(self.scent, 0, 1, out=self.scent)

    def _update_fertility(self, grid_resources: np.ndarray):
        self.fertility += 0.001 * (0.5 - self.fertility)

    def _update_toxin_spread(self):
        padded = np.pad(self.toxin, 1, mode='wrap')
        neighbors = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                     padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
        self.toxin += 0.003 * (neighbors - self.toxin)
        self.toxin *= 0.999
        np.clip(self.toxin, 0, 1, out=self.toxin)

    def stamp_scent(self, x: int, y: int, intensity: float = 0.5):
        self.scent[x % self.w, y % self.h] = min(
            1.0, self.scent[x % self.w, y % self.h] + intensity
        )

    def get_cell_physics(self, x: int, y: int) -> np.ndarray:
        return np.array([
            self.temperature[x % self.w, y % self.h],
            self.mineral[x % self.w, y % self.h],
            self.toxin[x % self.w, y % self.h],
            self.scent[x % self.w, y % self.h],
            self.fertility[x % self.w, y % self.h],
        ])

    def get_temperature(self, x: int, y: int) -> float:
        return float(self.temperature[x % self.w, y % self.h])

    def get_toxin(self, x: int, y: int) -> float:
        return float(self.toxin[x % self.w, y % self.h])

    def get_mineral(self, x: int, y: int) -> float:
        return float(self.mineral[x % self.w, y % self.h])

    def consume_mineral(self, x: int, y: int, amount: float = 0.2) -> float:
        available = self.mineral[x % self.w, y % self.h]
        collected = min(available, amount)
        self.mineral[x % self.w, y % self.h] -= collected
        return collected

    def apply_medicine(self, food_amount: float, mineral_amount: float) -> float:
        if food_amount >= 0.1 and mineral_amount >= 0.1:
            return min(food_amount, mineral_amount) * 0.5
        return 0.0

    def get_resource_growth_modifier(self, x: int, y: int) -> float:
        temp = self.temperature[x % self.w, y % self.h]
        fert = self.fertility[x % self.w, y % self.h]
        temp_factor = 1.0 - 2.0 * abs(temp - 0.55)
        temp_factor = max(0.1, temp_factor)
        return temp_factor * fert

    def get_metabolism_modifier(self, x: int, y: int) -> float:
        temp = self.temperature[x % self.w, y % self.h]
        return 0.7 + 0.6 * temp
