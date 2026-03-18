
import numpy as np
from config import Config


NUM_CHANNELS = 4
CH_RESOURCE = 0
CH_HAZARD = 1
CH_AGENT = 2
CH_SIGNAL = 3


class Grid:
    def __init__(self, cfg: Config, rng: np.random.Generator):
        self.w = cfg.world_width
        self.h = cfg.world_height
        self.cells = np.zeros((self.w, self.h, NUM_CHANNELS), dtype=np.float32)
        self._rng = rng
        self._cfg = cfg

        self._seed_resources()
        self.camouflaged = np.zeros((self.w, self.h), dtype=np.bool_)
        self._init_camouflage()

    def _seed_resources(self):
        n_clusters = max(1, (self.w * self.h) // 200)
        for _ in range(n_clusters):
            cx = self._rng.integers(0, self.w)
            cy = self._rng.integers(0, self.h)
            radius = self._rng.integers(3, 10)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:
                        x = (cx + dx) % self.w
                        y = (cy + dy) % self.h
                        self.cells[x, y, CH_RESOURCE] = min(
                            1.0, self.cells[x, y, CH_RESOURCE] + self._rng.uniform(0.3, 0.8)
                        )

    def get_cell(self, x: int, y: int) -> np.ndarray:
        return self.cells[x % self.w, y % self.h]

    def update_resources(self, season_modifier: float = 1.0):
        r = self.cells[:, :, CH_RESOURCE]
        growth = self._cfg.resource_growth_rate * season_modifier
        r += growth * r * (1.0 - r)

        if self._cfg.resource_diffusion > 0:
            d = self._cfg.resource_diffusion
            padded = np.pad(r, 1, mode='wrap')
            neighbors = (
                padded[:-2, 1:-1] + padded[2:, 1:-1] +
                padded[1:-1, :-2] + padded[1:-1, 2:]
            ) / 4.0
            r += d * (neighbors - r)

        np.clip(r, 0, 1, out=r)
        self.cells[:, :, CH_RESOURCE] = r

    def update_hazards(self):
        h = self.cells[:, :, CH_HAZARD]

        if self._cfg.hazard_spread_rate > 0:
            padded = np.pad(h, 1, mode='wrap')
            neighbors_max = np.maximum.reduce([
                padded[:-2, 1:-1], padded[2:, 1:-1],
                padded[1:-1, :-2], padded[1:-1, 2:]
            ])
            spread = self._cfg.hazard_spread_rate * (neighbors_max - h)
            h += np.maximum(0, spread)

        h *= (1.0 - self._cfg.hazard_decay_rate)
        np.clip(h, 0, 1, out=h)
        self.cells[:, :, CH_HAZARD] = h

    def update_signals(self):
        self.cells[:, :, CH_SIGNAL] *= 0.95

    def clear_agent_layer(self):
        self.cells[:, :, CH_AGENT] = 0.0

    def stamp_agent(self, x: int, y: int, signal: float = 0.0):
        self.cells[x % self.w, y % self.h, CH_AGENT] = 1.0
        if signal > 0:
            self.cells[x % self.w, y % self.h, CH_SIGNAL] = max(
                self.cells[x % self.w, y % self.h, CH_SIGNAL], signal
            )

    def spawn_hazard(self, x: int, y: int, intensity: float = 0.8):
        self.cells[x % self.w, y % self.h, CH_HAZARD] = min(
            1.0, self.cells[x % self.w, y % self.h, CH_HAZARD] + intensity
        )

    def _init_camouflage(self):
        resource = self.cells[:, :, CH_RESOURCE]
        mask = (resource > 0.3) & (self._rng.random((self.w, self.h)) < 0.15)
        self.camouflaged = mask
        self.cells[mask, CH_RESOURCE] = np.minimum(1.0, self.cells[mask, CH_RESOURCE] * 1.5)

    def refresh_camouflage(self, rng):
        self.camouflaged[:] = False
        resource = self.cells[:, :, CH_RESOURCE]
        mask = (resource > 0.3) & (rng.random((self.w, self.h)) < 0.15)
        self.camouflaged = mask
        self.cells[mask, CH_RESOURCE] = np.minimum(1.0, self.cells[mask, CH_RESOURCE] * 1.5)

    def consume_resource(self, x: int, y: int, amount: float = 0.3) -> float:
        available = self.cells[x % self.w, y % self.h, CH_RESOURCE]
        consumed = min(available, amount)
        self.cells[x % self.w, y % self.h, CH_RESOURCE] -= consumed
        return consumed
