
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

        # Disaster state
        self.disasters = DisasterSystem(cfg, grid, rng)

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

        # Update disaster system
        self.disasters.update(tick, self.season_phase)

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


class DisasterSystem:
    """Natural disasters that create selection pressure for larger brains.

    Each disaster type has precursor signals that smart agents can learn to detect
    and react to (flee, seek shelter, stockpile). Dumb agents just die.
    """

    def __init__(self, cfg: Config, grid: Grid, rng: np.random.Generator):
        self._cfg = cfg
        self._grid = grid
        self._rng = rng
        self.w = grid.w
        self.h = grid.h

        # Disaster intensity maps — agents can sense these
        self.earthquake_damage = np.zeros((self.w, self.h), dtype=np.float32)
        self.flood_level = np.zeros((self.w, self.h), dtype=np.float32)
        self.drought_severity = np.zeros((self.w, self.h), dtype=np.float32)
        self.plague_intensity = np.zeros((self.w, self.h), dtype=np.float32)

        # Precursor warning signals — subtle, detectable before disaster hits
        self.tremor_warning = np.zeros((self.w, self.h), dtype=np.float32)
        self.flood_warning = np.zeros((self.w, self.h), dtype=np.float32)
        self.drought_warning = 0.0  # global scalar
        self.plague_warning = np.zeros((self.w, self.h), dtype=np.float32)

        # Active disaster tracking
        self._active_floods: list[dict] = []
        self._active_plagues: list[dict] = []
        self._drought_active = False
        self._drought_remaining = 0

    def update(self, tick: int, season_phase: float):
        """Main disaster update — called every tick."""
        # Decay all damage/warning maps
        self.earthquake_damage *= 0.85
        self.flood_level *= 0.97
        self.plague_intensity *= 0.98
        self.tremor_warning *= 0.9
        self.flood_warning *= 0.95
        self.plague_warning *= 0.95

        # Earthquake: random, with tremor precursors
        if tick % 10 == 0:
            self._update_earthquakes(tick)

        # Flood: tied to terrain water, builds slowly
        if tick % 5 == 0:
            self._update_floods(tick)

        # Drought: seasonal, global resource crash
        self._update_drought(tick, season_phase)

        # Plague: density-dependent disease
        if tick % 20 == 0:
            self._update_plague(tick)

    def _update_earthquakes(self, tick: int):
        """Earthquakes with tremor precursors.

        Pattern: small tremors build over ~50 ticks, then a big quake hits.
        Smart agents learn: tremor_warning high → move away from area.
        """
        # Random tremor generation (precursor)
        if self._rng.random() < 0.10:  # ~10% per 10 ticks — frequent tremors
            cx = self._rng.integers(0, self.w)
            cy = self._rng.integers(0, self.h)
            radius = self._rng.integers(10, 30)
            intensity = self._rng.uniform(0.3, 0.8)

            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    dist2 = dx * dx + dy * dy
                    if dist2 <= radius * radius:
                        x = (cx + dx) % self.w
                        y = (cy + dy) % self.h
                        falloff = 1.0 - np.sqrt(dist2) / radius
                        self.tremor_warning[x, y] = max(
                            self.tremor_warning[x, y],
                            intensity * falloff
                        )

        # When tremor warning accumulates enough → actual earthquake
        max_tremor = np.max(self.tremor_warning)
        if max_tremor > 0.5 and self._rng.random() < 0.25:
            # Find epicenter
            epicenter = np.unravel_index(np.argmax(self.tremor_warning),
                                         self.tremor_warning.shape)
            cx, cy = epicenter
            magnitude = self._rng.uniform(0.6, 1.0)
            radius = int(12 + magnitude * 20)

            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    dist2 = dx * dx + dy * dy
                    if dist2 <= radius * radius:
                        x = (cx + dx) % self.w
                        y = (cy + dy) % self.h
                        falloff = 1.0 - np.sqrt(dist2) / radius
                        dmg = magnitude * falloff * 0.7
                        self.earthquake_damage[x, y] = max(
                            self.earthquake_damage[x, y], dmg
                        )
                        # Destroy resources in epicenter
                        self._grid.cells[x, y, 0] *= max(0.1, 1.0 - dmg)
                        # Spawn hazard debris
                        if dmg > 0.15:
                            self._grid.spawn_hazard(x, y, dmg * 0.5)

            # Release tremor warning after quake
            self.tremor_warning[max(0, cx-radius):min(self.w, cx+radius),
                               max(0, cy-radius):min(self.h, cy+radius)] *= 0.1

    def _update_floods(self, tick: int):
        """Floods: water rises in low areas, precursor is rising water warning.

        Pattern: flood_warning rises over ~30 ticks as water accumulates.
        Smart agents learn: flood_warning + low ground → move to high ground.
        """
        # Random flood event
        if self._rng.random() < 0.025:  # ~every 200 ticks — frequent floods
            cx = self._rng.integers(0, self.w)
            cy = self._rng.integers(0, self.h)
            radius = self._rng.integers(15, 35)
            severity = self._rng.uniform(0.5, 1.0)
            duration = self._rng.integers(30, 80)
            self._active_floods.append({
                "cx": cx, "cy": cy, "radius": radius,
                "severity": severity, "remaining": duration,
                "phase": "warning",  # warning → rising → peak → receding
                "warning_ticks": self._rng.integers(15, 30),
            })

        # Process active floods
        new_floods = []
        for flood in self._active_floods:
            cx, cy = flood["cx"], flood["cy"]
            r = flood["radius"]
            sev = flood["severity"]

            if flood["phase"] == "warning":
                # Precursor: rising water warning signal
                flood["warning_ticks"] -= 1
                warning_strength = sev * (1.0 - flood["warning_ticks"] / 30.0)
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        dist2 = dx * dx + dy * dy
                        if dist2 <= r * r:
                            x = (cx + dx) % self.w
                            y = (cy + dy) % self.h
                            falloff = 1.0 - np.sqrt(dist2) / r
                            self.flood_warning[x, y] = max(
                                self.flood_warning[x, y],
                                warning_strength * falloff * 0.6
                            )
                if flood["warning_ticks"] <= 0:
                    flood["phase"] = "active"

            elif flood["phase"] == "active":
                # Actual flood damage
                flood["remaining"] -= 1
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        dist2 = dx * dx + dy * dy
                        if dist2 <= r * r:
                            x = (cx + dx) % self.w
                            y = (cy + dy) % self.h
                            falloff = 1.0 - np.sqrt(dist2) / r
                            flood_dmg = sev * falloff * 0.3
                            self.flood_level[x, y] = max(
                                self.flood_level[x, y], flood_dmg
                            )
                            # Destroy resources (wash away)
                            self._grid.cells[x, y, 0] *= max(0.3, 1.0 - flood_dmg * 0.5)

            if flood["remaining"] > 0:
                new_floods.append(flood)
        self._active_floods = new_floods

    def _update_drought(self, tick: int, season_phase: float):
        """Drought: global resource scarcity, precursor is falling season modifier.

        Pattern: drought_warning rises gradually, then resources crash.
        Smart agents learn: drought_warning high → stockpile food, reduce activity.
        """
        # Drought probability increases during harsh seasons
        season_harsh = max(0, 1.0 - 2.0 * (0.5 + 0.5 * np.sin(2 * np.pi * season_phase)))

        if not self._drought_active:
            # Higher chance during harsh season
            if self._rng.random() < 0.001 + 0.003 * season_harsh:
                self._drought_active = True
                self._drought_remaining = self._rng.integers(100, 300)
                self.drought_warning = 0.3  # initial warning

        if self._drought_active:
            self._drought_remaining -= 1
            # Drought warning ramps up
            self.drought_warning = min(1.0, self.drought_warning + 0.005)
            # Drought severity increases
            drought_factor = min(1.0, self.drought_warning)
            self.drought_severity[:] = drought_factor * 0.3

            # Globally reduce resource growth
            self._grid.cells[:, :, 0] *= (1.0 - drought_factor * 0.05)

            if self._drought_remaining <= 0:
                self._drought_active = False
                self.drought_warning = 0.0
                self.drought_severity[:] = 0.0
        else:
            self.drought_warning = max(0, self.drought_warning - 0.01)

    def _update_plague(self, tick: int):
        """Plague: spreads in dense populations, precursor is nearby sickness.

        Pattern: plague_warning appears near sick areas. Spreads to neighbors.
        Smart agents learn: plague_warning → avoid crowds, isolate.
        """
        # Spontaneous plague outbreak in dense areas
        if self._rng.random() < 0.04:  # 4x more plague attempts
            cx = self._rng.integers(0, self.w)
            cy = self._rng.integers(0, self.h)
            # Check if area has agents (use agent grid channel)
            agent_density = 0.0
            radius = 5
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if dx * dx + dy * dy <= radius * radius:
                        x = (cx + dx) % self.w
                        y = (cy + dy) % self.h
                        agent_density += self._grid.cells[x, y, 2]  # agent channel

            # Only spawn plague in dense areas
            if agent_density > 3:
                virulence = self._rng.uniform(0.4, 0.9)
                spread_radius = self._rng.integers(8, 20)
                for dx in range(-spread_radius, spread_radius + 1):
                    for dy in range(-spread_radius, spread_radius + 1):
                        dist2 = dx * dx + dy * dy
                        if dist2 <= spread_radius * spread_radius:
                            x = (cx + dx) % self.w
                            y = (cy + dy) % self.h
                            falloff = 1.0 - np.sqrt(dist2) / spread_radius
                            self.plague_intensity[x, y] = max(
                                self.plague_intensity[x, y],
                                virulence * falloff
                            )
                            # Warning spreads wider than plague itself
                            self.plague_warning[x, y] = max(
                                self.plague_warning[x, y],
                                virulence * falloff * 0.8
                            )

        # Plague spreads via diffusion (contagion)
        if np.max(self.plague_intensity) > 0.05:
            padded = np.pad(self.plague_intensity, 1, mode='wrap')
            neighbors = (
                padded[:-2, 1:-1] + padded[2:, 1:-1] +
                padded[1:-1, :-2] + padded[1:-1, 2:]
            ) / 4.0
            # Spreads faster where agents are present
            agent_boost = 1.0 + self._grid.cells[:, :, 2] * 2.0
            spread = 0.02 * (neighbors - self.plague_intensity) * agent_boost
            self.plague_intensity += np.maximum(0, spread)
            np.clip(self.plague_intensity, 0, 1, out=self.plague_intensity)

            # Warning tracks plague
            self.plague_warning = np.maximum(
                self.plague_warning, self.plague_intensity * 0.7
            )

    def get_disaster_damage(self, x: int, y: int) -> dict:
        """Get all disaster damage at a position. Called by main loop."""
        x, y = x % self.w, y % self.h
        return {
            "earthquake": float(self.earthquake_damage[x, y]),
            "flood": float(self.flood_level[x, y]),
            "drought": float(self.drought_severity[x, y]),
            "plague": float(self.plague_intensity[x, y]),
        }

    def get_disaster_warnings(self, x: int, y: int) -> dict:
        """Get precursor warning signals at a position. Fed to agent sensors."""
        x, y = x % self.w, y % self.h
        return {
            "tremor": float(self.tremor_warning[x, y]),
            "flood": float(self.flood_warning[x, y]),
            "drought": float(self.drought_warning),
            "plague": float(self.plague_warning[x, y]),
        }

    def get_total_damage(self, x: int, y: int) -> float:
        """Total disaster damage at position (for quick access)."""
        dmg = self.get_disaster_damage(x, y)
        return dmg["earthquake"] + dmg["flood"] + dmg["drought"] * 0.05 + dmg["plague"] * 0.08

    def get_stats(self) -> dict:
        return {
            "earthquake_max": float(np.max(self.earthquake_damage)),
            "flood_max": float(np.max(self.flood_level)),
            "drought_severity": float(self.drought_warning),
            "plague_max": float(np.max(self.plague_intensity)),
            "active_floods": len(self._active_floods),
            "drought_active": self._drought_active,
        }
