
import numpy as np


class HiddenPhysics:
    """Invisible variables that agents cannot directly sense but whose effects are observable.

    - radiation: invisible health damage, increases mutation rate
    - magnetic_field: (x,y) vector field, useful for navigation if agents evolve magnetosense
    - underground: hidden mineral deposits, revealed by earthquakes
    - soil_ph: affects plant growth, changed by chemical reactions

    Agents must learn correlations: "I get sick here" -> radiation zone,
    "plants grow well here" -> good soil pH, etc.
    """

    def __init__(self, width, height, rng=None):
        self.w = width
        self.h = height
        self.rng = rng or np.random.default_rng(42)

        # Radiation: invisible zones that damage health and boost mutation
        self.radiation = np.zeros((width, height), dtype=np.float32)

        # Magnetic field: 2D vector field for navigation
        self.magnetic_field = np.zeros((width, height, 2), dtype=np.float32)

        # Underground resources: hidden until earthquake reveals them
        self.underground = np.zeros((width, height, 4), dtype=np.float32)

        # Soil pH: 0=acidic, 0.5=neutral, 1=alkaline — affects plant growth
        self.soil_ph = np.full((width, height), 0.5, dtype=np.float32)

        # Radiation shielding substance combo (discoverable)
        self.shield_substances = None  # set after chemistry is available

        self._init_radiation()
        self._init_magnetic()
        self._init_underground()
        self._init_soil_ph()

    def _init_radiation(self):
        """Create radiation zones: a few hotspots with gaussian falloff."""
        n_sources = max(2, self.w * self.h // 3000)
        for _ in range(n_sources):
            cx = self.rng.integers(0, self.w)
            cy = self.rng.integers(0, self.h)
            intensity = self.rng.uniform(0.3, 0.9)
            radius = self.rng.integers(5, 15)

            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    dist2 = dx * dx + dy * dy
                    if dist2 > radius * radius:
                        continue
                    x = (cx + dx) % self.w
                    y = (cy + dy) % self.h
                    falloff = np.exp(-dist2 / (2.0 * (radius / 2.5) ** 2))
                    self.radiation[x, y] = max(self.radiation[x, y], intensity * falloff)

        np.clip(self.radiation, 0, 1, out=self.radiation)

    def _init_magnetic(self):
        """Create a smooth magnetic field with anomalies near mineral deposits."""
        # Base dipole-like field (global direction)
        for y in range(self.h):
            lat = (y - self.h / 2) / (self.h / 2)  # -1 to 1
            self.magnetic_field[:, y, 0] = 0.3 * (1.0 - abs(lat))  # x-component: weak at poles
            self.magnetic_field[:, y, 1] = 0.5 * lat  # y-component: points toward poles

        # Add anomalies (random perturbations)
        n_anomalies = max(3, self.w * self.h // 2000)
        for _ in range(n_anomalies):
            cx = self.rng.integers(0, self.w)
            cy = self.rng.integers(0, self.h)
            r = self.rng.integers(4, 10)
            strength = self.rng.uniform(0.2, 0.6)
            angle = self.rng.uniform(0, 2 * np.pi)
            dx_comp = strength * np.cos(angle)
            dy_comp = strength * np.sin(angle)

            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    dist2 = dx * dx + dy * dy
                    if dist2 > r * r:
                        continue
                    x = (cx + dx) % self.w
                    y = (cy + dy) % self.h
                    falloff = np.exp(-dist2 / (2.0 * (r / 2.0) ** 2))
                    self.magnetic_field[x, y, 0] += dx_comp * falloff
                    self.magnetic_field[x, y, 1] += dy_comp * falloff

        np.clip(self.magnetic_field, -1, 1, out=self.magnetic_field)

    def _init_underground(self):
        """Hidden resource deposits: 4 types of underground material."""
        for layer in range(4):
            n_deposits = max(2, self.w * self.h // 4000)
            for _ in range(n_deposits):
                cx = self.rng.integers(0, self.w)
                cy = self.rng.integers(0, self.h)
                r = self.rng.integers(3, 8)
                richness = self.rng.uniform(0.3, 0.8)

                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        dist2 = dx * dx + dy * dy
                        if dist2 > r * r:
                            continue
                        x = (cx + dx) % self.w
                        y = (cy + dy) % self.h
                        falloff = 1.0 - np.sqrt(dist2) / r
                        self.underground[x, y, layer] = max(
                            self.underground[x, y, layer],
                            richness * falloff
                        )

        np.clip(self.underground, 0, 1, out=self.underground)

    def _init_soil_ph(self):
        """Initialize soil pH with spatial variation."""
        # FFT-based smooth noise
        noise = self.rng.standard_normal((self.w, self.h)).astype(np.float32)
        freq = np.fft.fft2(noise)
        kx = np.fft.fftfreq(self.w)[:, None]
        ky = np.fft.fftfreq(self.h)[None, :]
        k2 = kx ** 2 + ky ** 2
        filt = np.exp(-k2 * 20 ** 2 * 2 * np.pi ** 2)
        filtered = np.real(np.fft.ifft2(freq * filt)).astype(np.float32)

        # Normalize to [0.2, 0.8] centered on 0.5
        filtered -= filtered.mean()
        std = filtered.std()
        if std > 0:
            filtered /= std
        self.soil_ph = 0.5 + 0.15 * filtered
        np.clip(self.soil_ph, 0, 1, out=self.soil_ph)

    def setup_shield(self, chemistry):
        """Pick which substance combination blocks radiation (seed-dependent)."""
        n = chemistry.n_substances
        self.shield_substances = (
            int(self.rng.integers(0, n)),
            int(self.rng.integers(0, n)),
        )

    def update(self, tick, temperature=None, chemistry=None, earthquake_signal=None):
        """Update hidden variables."""
        if tick % 20 == 0:
            self._update_radiation(chemistry)
        if tick % 50 == 0:
            self._update_soil_ph(temperature)
        if earthquake_signal is not None and tick % 10 == 0:
            self._reveal_underground(earthquake_signal, chemistry)

    def _update_radiation(self, chemistry=None):
        """Radiation slowly drifts and decays. Shielding substances reduce it locally."""
        # Slow diffusion
        padded = np.pad(self.radiation, 1, mode='wrap')
        neighbors = (padded[:-2, 1:-1] + padded[2:, 1:-1] +
                     padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
        self.radiation += 0.01 * (neighbors - self.radiation)

        # Very slow natural decay
        self.radiation *= 0.9995

        # Shielding: certain substances reduce local radiation
        if chemistry is not None and self.shield_substances is not None:
            s1, s2 = self.shield_substances
            shield_amount = np.minimum(
                chemistry.substances[:, :, s1],
                chemistry.substances[:, :, s2]
            )
            self.radiation -= shield_amount * 0.02
            np.clip(self.radiation, 0, 1, out=self.radiation)

    def _update_soil_ph(self, temperature=None):
        """Soil pH slowly reverts to neutral, affected by temperature."""
        # Revert toward 0.5
        self.soil_ph += 0.002 * (0.5 - self.soil_ph)

        # High temperature makes soil slightly more acidic
        if temperature is not None:
            hot_shift = (temperature - 0.5) * 0.001
            self.soil_ph -= hot_shift

        np.clip(self.soil_ph, 0, 1, out=self.soil_ph)

    def _reveal_underground(self, earthquake_signal, chemistry=None):
        """Earthquakes push underground resources to the surface."""
        if chemistry is None:
            return

        # Where earthquake is strong, reveal underground
        reveal_mask = earthquake_signal > 0.3
        if not np.any(reveal_mask):
            return

        for layer in range(4):
            revealed = self.underground[:, :, layer] * earthquake_signal * 0.1 * reveal_mask
            # Map underground layers to chemistry substances (8-11 = organic group)
            sub_idx = 8 + layer
            if sub_idx < chemistry.n_substances:
                chemistry.substances[:, :, sub_idx] += revealed
                self.underground[:, :, layer] -= revealed

        np.clip(self.underground, 0, 1, out=self.underground)

    def get_radiation(self, x, y):
        """Radiation at position. Agents can't sense this directly."""
        return float(self.radiation[x % self.w, y % self.h])

    def get_radiation_damage(self, x, y):
        """Health damage from radiation. This is what agents experience (effect, not cause)."""
        rad = self.radiation[x % self.w, y % self.h]
        if rad < 0.1:
            return 0.0
        return float(rad * 0.02)

    def get_mutation_modifier(self, x, y):
        """Radiation increases mutation rate. Higher radiation = more mutation."""
        rad = self.radiation[x % self.w, y % self.h]
        return 1.0 + rad * 3.0  # 1x to 4x mutation rate

    def get_magnetic(self, x, y):
        """Magnetic field vector at position. Only useful if agent has magnetosense."""
        x, y = x % self.w, y % self.h
        return self.magnetic_field[x, y].copy()

    def get_magnetic_strength(self, x, y):
        """Magnetic field magnitude. Anomalies indicate mineral deposits."""
        vec = self.magnetic_field[x % self.w, y % self.h]
        return float(np.sqrt(vec[0] ** 2 + vec[1] ** 2))

    def get_soil_ph(self, x, y):
        """Soil pH at position."""
        return float(self.soil_ph[x % self.w, y % self.h])

    def get_plant_growth_modifier(self, x, y):
        """How much soil pH helps/hurts plant growth. Neutral pH (0.5) is best."""
        ph = self.soil_ph[x % self.w, y % self.h]
        # Gaussian around 0.5 with some tolerance
        return float(np.exp(-8.0 * (ph - 0.5) ** 2))

    def acidify_soil(self, x, y, amount=0.05):
        """Chemical reactions can change soil pH."""
        x, y = x % self.w, y % self.h
        self.soil_ph[x, y] = max(0, self.soil_ph[x, y] - amount)

    def alkalize_soil(self, x, y, amount=0.05):
        x, y = x % self.w, y % self.h
        self.soil_ph[x, y] = min(1, self.soil_ph[x, y] + amount)

    def get_stats(self):
        return {
            "mean_radiation": float(np.mean(self.radiation)),
            "max_radiation": float(np.max(self.radiation)),
            "mean_soil_ph": float(np.mean(self.soil_ph)),
            "underground_total": float(np.sum(self.underground)),
            "magnetic_mean_strength": float(np.mean(
                np.sqrt(self.magnetic_field[:, :, 0] ** 2 + self.magnetic_field[:, :, 1] ** 2)
            )),
        }
