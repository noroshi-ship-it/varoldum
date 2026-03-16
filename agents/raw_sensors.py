
import numpy as np
from config import Config
from agents.genome import get_trait
from world.grid import Grid


PATCH_RADIUS = 2
PATCH_SIZE = 2 * PATCH_RADIUS + 1
RAW_CHANNELS = 8
RAW_INPUT_DIM = PATCH_SIZE * PATCH_SIZE * RAW_CHANNELS


class RawSensors:

    def __init__(self, genome: np.ndarray, cfg: Config):
        self.sensor_range = int(round(get_trait(genome, "sensor_range")))
        self._cfg = cfg
        self._stride = max(1, self.sensor_range // PATCH_RADIUS) if self.sensor_range > PATCH_RADIUS else 1

    @property
    def output_dim(self) -> int:
        return RAW_INPUT_DIM

    def perceive(self, grid: Grid, position: np.ndarray, heading: float,
                 light_level: float = 1.0, rng=None, physics=None) -> np.ndarray:
        cx, cy = int(position[0]), int(position[1])
        w, h = grid.w, grid.h
        stride = self._stride

        cos_h = np.cos(heading)
        sin_h = np.sin(heading)

        patch = np.zeros(RAW_INPUT_DIM)
        idx = 0

        for dy in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
            for dx in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
                rx = int(round(cos_h * dx - sin_h * dy)) * stride
                ry = int(round(sin_h * dx + cos_h * dy)) * stride

                sx = (cx + rx) % w
                sy = (cy + ry) % h

                cell = grid.get_cell(sx, sy)
                patch[idx + 0] = cell[0]
                patch[idx + 1] = cell[1]
                patch[idx + 2] = cell[2]
                patch[idx + 3] = cell[3]

                if physics is not None:
                    phys = physics.get_cell_physics(sx, sy)
                    patch[idx + 4] = phys[0]
                    patch[idx + 5] = phys[1]
                    patch[idx + 6] = phys[2]
                    patch[idx + 7] = phys[3]
                else:
                    patch[idx + 4:idx + 8] = 0.0

                idx += RAW_CHANNELS

        for dy in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
            for dx in range(-PATCH_RADIUS, PATCH_RADIUS + 1):
                dist = abs(dx) + abs(dy)
                if dist > 0:
                    atten = 1.0 / (1.0 + dist * 0.3)
                    base = ((dy + PATCH_RADIUS) * PATCH_SIZE + (dx + PATCH_RADIUS)) * RAW_CHANNELS
                    patch[base:base + RAW_CHANNELS] *= atten

        for i in range(PATCH_SIZE * PATCH_SIZE):
            base = i * RAW_CHANNELS
            patch[base:base + 4] *= light_level

        if rng is not None:
            patch += rng.normal(0, 0.02, size=patch.shape)

        return np.clip(patch, 0.0, 1.0)
