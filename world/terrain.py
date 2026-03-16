
import numpy as np


class Terrain:
    """Height map, water flow, tectonic events.

    Creates spatial structure: mountains, valleys, rivers, lakes.
    Earthquakes and volcanoes reshape the world.
    """

    # Rock types determine which substances are released by erosion
    N_ROCK_TYPES = 8

    def __init__(self, width, height, rng=None):
        self.w = width
        self.h = height
        self.rng = rng or np.random.default_rng(42)

        self.height = np.zeros((width, height), dtype=np.float32)
        self.water = np.zeros((width, height), dtype=np.float32)
        self.rock_type = np.zeros((width, height), dtype=np.int8)
        self.tectonic_stress = np.zeros((width, height), dtype=np.float32)

        # Fault lines: predetermined stress accumulation zones
        self.fault_mask = np.zeros((width, height), dtype=np.float32)

        # Recent events for agent sensing
        self.recent_earthquake = np.zeros((width, height), dtype=np.float32)

        # Rock-to-substance mapping: which chemistry substance each rock releases
        # Shape: (N_ROCK_TYPES,) -> substance index
        self.rock_substance_map = np.zeros(self.N_ROCK_TYPES, dtype=np.int32)

        self._init_height()
        self._init_rocks()
        self._init_faults()

    def _init_height(self):
        """Generate height map using FFT-based noise."""
        W, H = self.w, self.h

        # Multi-octave noise via FFT
        height = np.zeros((W, H), dtype=np.float32)
        for scale, weight in [(32, 0.5), (16, 0.25), (8, 0.15), (4, 0.1)]:
            noise = self.rng.standard_normal((W, H)).astype(np.float32)
            # Low-pass filter in frequency domain
            freq = np.fft.fft2(noise)
            kx = np.fft.fftfreq(W)[:, None]
            ky = np.fft.fftfreq(H)[None, :]
            k2 = kx**2 + ky**2
            filt = np.exp(-k2 * scale**2 * 2 * np.pi**2)
            filtered = np.real(np.fft.ifft2(freq * filt)).astype(np.float32)
            height += filtered * weight

        # Normalize to [0, 1]
        height -= height.min()
        if height.max() > 0:
            height /= height.max()

        # Add some flat areas (plains) by clamping low regions
        height = np.where(height < 0.3, height * 0.5, height)

        self.height = height

    def _init_rocks(self):
        """Assign rock types based on height and random clusters."""
        # Base: rock type correlates with height bands
        self.rock_type = (self.height * (self.N_ROCK_TYPES - 1)).astype(np.int8)

        # Add random clusters for variety
        n_clusters = max(5, self.w * self.h // 2000)
        for _ in range(n_clusters):
            cx = self.rng.integers(0, self.w)
            cy = self.rng.integers(0, self.h)
            r = self.rng.integers(3, 10)
            rtype = self.rng.integers(0, self.N_ROCK_TYPES)
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        x = (cx + dx) % self.w
                        y = (cy + dy) % self.h
                        self.rock_type[x, y] = rtype

        # Map rocks to chemistry substance indices (minerals 4-7, spread across)
        for i in range(self.N_ROCK_TYPES):
            self.rock_substance_map[i] = 4 + (i % 4)  # maps to mineral substances

    def _init_faults(self):
        """Create fault lines where tectonic stress accumulates."""
        # 2-4 fault lines as smooth curves across the map
        n_faults = self.rng.integers(2, 5)
        for _ in range(n_faults):
            # Random start and direction
            x = self.rng.integers(0, self.w)
            y = self.rng.integers(0, self.h)
            angle = self.rng.uniform(0, 2 * np.pi)
            length = self.rng.integers(self.w // 3, self.w)

            for step in range(length):
                ix = int(x) % self.w
                iy = int(y) % self.h
                # Paint fault with gaussian cross-section
                for d in range(-2, 3):
                    nx = (ix + int(d * np.cos(angle + np.pi / 2))) % self.w
                    ny = (iy + int(d * np.sin(angle + np.pi / 2))) % self.h
                    self.fault_mask[nx, ny] = max(
                        self.fault_mask[nx, ny],
                        np.exp(-d * d / 2.0)
                    )
                x += np.cos(angle)
                y += np.sin(angle)
                # Slight random walk in direction
                angle += self.rng.normal(0, 0.05)

    def update(self, tick, chemistry=None):
        """Main terrain update."""
        # Water flow every 10 ticks
        if tick % 10 == 0:
            self._update_water()

        # Tectonic stress every 50 ticks
        if tick % 50 == 0:
            self._update_tectonics(chemistry)

        # Decay recent earthquake signal
        if tick % 5 == 0:
            self.recent_earthquake *= 0.8

    def _update_water(self):
        """Water cycle: rain, flow, evaporation."""
        h = self.height
        w = self.water

        # Rain on high ground
        w += 0.002 * (h > 0.5).astype(np.float32)

        # Flow downhill using height gradient
        grad_x = np.roll(h, -1, 0) - np.roll(h, 1, 0)
        grad_y = np.roll(h, -1, 1) - np.roll(h, 1, 1)

        # Water moves in direction of steepest descent
        flow_x = w * np.clip(-grad_x * 0.15, -0.05, 0.05)
        flow_y = w * np.clip(-grad_y * 0.15, -0.05, 0.05)

        # Remove water that flows out
        outflow = np.abs(flow_x) + np.abs(flow_y)
        w -= np.minimum(outflow, w)

        # Add water that flows in from neighbors
        w += np.roll(np.maximum(flow_x, 0), 1, axis=0)
        w += np.roll(np.maximum(-flow_x, 0), -1, axis=0)
        w += np.roll(np.maximum(flow_y, 0), 1, axis=1)
        w += np.roll(np.maximum(-flow_y, 0), -1, axis=1)

        # Evaporation (faster at low altitude = warmer)
        evap = 0.005 * (1.0 - h * 0.5)
        w -= evap
        w += 0.0005  # tiny base moisture everywhere

        # Accumulation in valleys
        low_ground = h < 0.2
        w += low_ground.astype(np.float32) * 0.001

        np.clip(w, 0, 1, out=w)
        self.water = w

    def _update_tectonics(self, chemistry=None):
        """Tectonic stress buildup and release."""
        # Stress accumulates along fault lines
        self.tectonic_stress += 0.005 * self.fault_mask

        # Add random micro-stress
        self.tectonic_stress += self.rng.uniform(0, 0.001, (self.w, self.h)).astype(np.float32)

        # Check for earthquakes
        max_pos = np.unravel_index(np.argmax(self.tectonic_stress), self.tectonic_stress.shape)
        max_stress = self.tectonic_stress[max_pos]

        if max_stress > 0.8:
            self._earthquake(max_pos[0], max_pos[1], max_stress, chemistry)

        # Check for volcanic eruption (very rare, needs extreme stress)
        if max_stress > 0.95:
            self._volcano(max_pos[0], max_pos[1], chemistry)

    def _earthquake(self, cx, cy, intensity, chemistry=None):
        """Earthquake: shakes terrain, releases underground minerals, damages area."""
        radius = int(5 + intensity * 10)

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                dist2 = dx * dx + dy * dy
                if dist2 > radius * radius:
                    continue
                x = (cx + dx) % self.w
                y = (cy + dy) % self.h

                falloff = 1.0 - np.sqrt(dist2) / radius

                # Shake height
                self.height[x, y] += self.rng.normal(0, 0.03 * falloff)

                # Release minerals based on rock type
                if chemistry is not None and falloff > 0.3:
                    rtype = self.rock_type[x, y]
                    sub_idx = self.rock_substance_map[rtype]
                    chemistry.deposit_substance(x, y, sub_idx, 0.1 * falloff)

                # Earthquake signal for agents
                self.recent_earthquake[x, y] = max(
                    self.recent_earthquake[x, y], falloff * intensity
                )

        # Release tectonic stress in area
        for dx in range(-radius * 2, radius * 2 + 1):
            for dy in range(-radius * 2, radius * 2 + 1):
                x = (cx + dx) % self.w
                y = (cy + dy) % self.h
                self.tectonic_stress[x, y] *= 0.1

        np.clip(self.height, 0, 1, out=self.height)

    def _volcano(self, cx, cy, chemistry=None):
        """Volcanic eruption: creates new terrain, deposits unique substances."""
        radius = self.rng.integers(4, 8)

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                dist = np.sqrt(dx * dx + dy * dy)
                if dist > radius:
                    continue
                x = (cx + dx) % self.w
                y = (cy + dy) % self.h

                falloff = 1.0 - dist / radius

                # Raise terrain (volcano cone)
                self.height[x, y] = min(1.0, self.height[x, y] + 0.3 * falloff)

                # Deposit volatile substances (hot, reactive)
                if chemistry is not None:
                    for sub_idx in range(12, 16):  # volatiles
                        chemistry.deposit_substance(x, y, sub_idx, 0.2 * falloff)

                # Strong earthquake signal
                self.recent_earthquake[x, y] = 1.0

                # Evaporate water
                self.water[x, y] *= max(0, 1.0 - falloff)

        # Reset stress
        self.tectonic_stress[cx, cy] = 0

    def get_height(self, x, y):
        return float(self.height[x % self.w, y % self.h])

    def get_water(self, x, y):
        return float(self.water[x % self.w, y % self.h])

    def get_slope(self, x, y):
        """Get terrain slope magnitude at position."""
        x, y = x % self.w, y % self.h
        dx = self.height[(x + 1) % self.w, y] - self.height[(x - 1) % self.w, y]
        dy = self.height[x, (y + 1) % self.h] - self.height[x, (y - 1) % self.h]
        return float(np.sqrt(dx * dx + dy * dy))

    def get_earthquake_intensity(self, x, y):
        return float(self.recent_earthquake[x % self.w, y % self.h])

    def is_flooded(self, x, y, threshold=0.5):
        return self.water[x % self.w, y % self.h] > threshold

    def get_stats(self):
        return {
            "mean_height": float(np.mean(self.height)),
            "water_coverage": float(np.mean(self.water > 0.1)),
            "max_tectonic_stress": float(np.max(self.tectonic_stress)),
            "earthquake_active": float(np.max(self.recent_earthquake)),
        }
