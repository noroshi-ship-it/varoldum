
import numpy as np
from config import Config
from world.grid import Grid


class Environment:
    def __init__(self, cfg: Config, grid: Grid, rng: np.random.Generator):
        self._cfg = cfg
        self._grid = grid
        self._rng = rng
        self.tick = 0
        self.season_phase = 0.0
        self.day_phase = 0.0

    @property
    def season_modifier(self) -> float:
        return 0.5 + 0.5 * np.sin(2 * np.pi * self.season_phase)

    @property
    def is_day(self) -> bool:
        return self.day_phase < 0.5

    @property
    def light_level(self) -> float:
        return 0.5 + 0.5 * np.cos(2 * np.pi * self.day_phase)

    def update(self, tick: int):
        self.tick = tick
        self.season_phase = (tick % self._cfg.season_period) / self._cfg.season_period
        self.day_phase = (tick % 100) / 100.0

        self._grid.update_resources(self.season_modifier)
        self._grid.update_hazards()
        self._grid.update_signals()

        if self._rng.random() < self._cfg.catastrophe_probability:
            self._spawn_catastrophe()

        if tick % 200 == 0:
            self._spawn_random_hazard()

    def _spawn_catastrophe(self):
        cx = self._rng.integers(0, self._grid.w)
        cy = self._rng.integers(0, self._grid.h)
        radius = self._rng.integers(8, 20)

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:
                    x = (cx + dx) % self._grid.w
                    y = (cy + dy) % self._grid.h
                    self._grid.cells[x, y, 0] *= 0.1
                    self._grid.spawn_hazard(x, y, 0.6)

    def _spawn_random_hazard(self):
        n = self._rng.integers(1, 4)
        for _ in range(n):
            x = self._rng.integers(0, self._grid.w)
            y = self._rng.integers(0, self._grid.h)
            self._grid.spawn_hazard(x, y, self._rng.uniform(0.3, 0.7))
