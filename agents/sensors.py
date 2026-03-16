
import numpy as np
from config import Config
from agents.genome import get_trait
from utils.geometry import cone_cells
from world.grid import Grid, CH_RESOURCE, CH_HAZARD, CH_AGENT, CH_SIGNAL


class Sensors:
    def __init__(self, genome: np.ndarray, cfg: Config):
        self.sensor_range = int(round(get_trait(genome, "sensor_range")))
        self.cone_width = get_trait(genome, "sensor_cone_width")
        self.resolution = cfg.sensor_resolution
        self.channels = cfg.sensor_channels
        self._cfg = cfg

    @property
    def output_dim(self) -> int:
        return self.resolution * self.channels

    def perceive(
        self,
        grid: Grid,
        position: np.ndarray,
        heading: float,
        light_level: float = 1.0,
        rng: np.random.Generator = None,
        physics=None,
    ) -> np.ndarray:
        cells = cone_cells(
            position, heading, self.cone_width,
            self.sensor_range, self.resolution,
            grid.w, grid.h,
        )

        sensor_vec = np.zeros(self.resolution * self.channels)
        cells_per_ray = max(1, self.sensor_range)

        for i, (cx, cy, dist) in enumerate(cells):
            ray_idx = i // cells_per_ray
            if ray_idx >= self.resolution:
                break
            weight = 1.0 / (1.0 + dist)
            cell_data = grid.get_cell(cx, cy)
            base = ray_idx * self.channels

            sensor_vec[base + 0] += cell_data[CH_RESOURCE] * weight
            sensor_vec[base + 1] += cell_data[CH_HAZARD] * weight
            sensor_vec[base + 2] += cell_data[CH_AGENT] * weight
            sensor_vec[base + 3] += cell_data[CH_SIGNAL] * weight

            if physics is not None:
                phys = physics.get_cell_physics(cx, cy)
                sensor_vec[base + 4] += phys[0] * weight
                sensor_vec[base + 5] += phys[1] * weight
                sensor_vec[base + 6] += phys[2] * weight
                sensor_vec[base + 7] += phys[3] * weight

        for r in range(self.resolution):
            base = r * self.channels
            sensor_vec[base:base + self.channels] /= cells_per_ray

        sensor_vec *= light_level
        if physics is not None:
            for r in range(self.resolution):
                base = r * self.channels
                sensor_vec[base + 4] /= max(0.1, light_level)
                sensor_vec[base + 7] /= max(0.1, light_level)

        if rng is not None:
            noise = rng.normal(0, 0.02, size=sensor_vec.shape)
            sensor_vec += noise

        return np.clip(sensor_vec, 0, 1)
