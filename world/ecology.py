
import numpy as np


class Ecology:
    """Grid-level ecology: plants and fauna as population density arrays.

    No individual entities — everything is density fields for performance.
    Creates a food web: substances -> plants -> herbivores -> predators -> decomposers -> soil.
    """

    def __init__(self, width, height, n_plant_species=4, n_fauna_species=3, rng=None):
        self.w = width
        self.h = height
        self.n_plants = n_plant_species
        self.n_fauna = n_fauna_species
        self.rng = rng or np.random.default_rng(42)

        # Plant biomass per species: (W, H, n_plants)
        self.plants = np.zeros((width, height, n_plant_species), dtype=np.float32)

        # Fauna population density: (W, H, n_fauna)
        # 0=herbivore, 1=predator, 2=decomposer
        self.fauna = np.zeros((width, height, n_fauna_species), dtype=np.float32)

        # Plant properties: [growth_rate, nutritive, toxicity, preferred_temp]
        self.plant_props = np.zeros((n_plant_species, 4), dtype=np.float32)

        # Fauna properties: [speed, aggression, size, reproduction_rate]
        self.fauna_props = np.zeros((n_fauna_species, 4), dtype=np.float32)

        # Dead organic matter (feeds decomposers)
        self.dead_matter = np.zeros((width, height), dtype=np.float32)

        self._init_plants()
        self._init_fauna()

    def _init_plants(self):
        """Generate plant species with random properties."""
        for sp in range(self.n_plants):
            self.plant_props[sp] = [
                self.rng.uniform(0.01, 0.05),  # growth_rate
                self.rng.uniform(0.1, 0.9),     # nutritive
                self.rng.uniform(0.0, 0.4) if self.rng.random() > 0.3 else self.rng.uniform(0.4, 0.8),
                self.rng.uniform(0.2, 0.8),     # preferred_temp
            ]

        # Seed initial plant populations
        for sp in range(self.n_plants):
            n_patches = max(3, self.w * self.h // 3000)
            for _ in range(n_patches):
                cx = self.rng.integers(0, self.w)
                cy = self.rng.integers(0, self.h)
                r = self.rng.integers(5, 15)
                intensity = self.rng.uniform(0.1, 0.4)
                xs = np.arange(max(0, cx - r), min(self.w, cx + r + 1))
                ys = np.arange(max(0, cy - r), min(self.h, cy + r + 1))
                for x in xs:
                    for y in ys:
                        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                        if dist <= r:
                            self.plants[x % self.w, y % self.h, sp] += intensity * (1 - dist / r)
            np.clip(self.plants[:, :, sp], 0, 1, out=self.plants[:, :, sp])

    def _init_fauna(self):
        """Initialize fauna properties and sparse populations."""
        # Herbivores: slow, not aggressive, small, reproduce fast
        self.fauna_props[0] = [0.3, 0.1, 0.3, 0.03]
        # Predators: fast, aggressive, big, reproduce slow
        self.fauna_props[1] = [0.7, 0.8, 0.7, 0.01]
        # Decomposers: slow, not aggressive, small, reproduce moderate
        self.fauna_props[2] = [0.1, 0.0, 0.2, 0.02]

        # Sparse initial herbivores where plants are
        total_plants = np.sum(self.plants, axis=2)
        self.fauna[:, :, 0] = np.where(total_plants > 0.2, 0.05, 0.0)

        # Sparse predators (~5x higher initial density)
        pred_mask = self.rng.random((self.w, self.h)) < 0.05
        self.fauna[:, :, 1] = pred_mask.astype(np.float32) * 0.1

        # Decomposers everywhere at low density
        self.fauna[:, :, 2] = 0.01

    def update(self, tick, temperature, water, light_level, soil_ph=None):
        """Main ecology update."""
        if tick % 5 == 0:
            self._update_plants(temperature, water, light_level, soil_ph)
        if tick % 10 == 0:
            self._update_fauna()
            self._update_decomposition()

    def _update_plants(self, temperature, water, light_level, soil_ph=None):
        """Plant growth: depends on water, light, temp match, soil pH, competition."""
        total = np.sum(self.plants, axis=2)

        # Soil pH factor: neutral (0.5) is best
        if soil_ph is not None:
            ph_factor = np.exp(-8.0 * (soil_ph - 0.5) ** 2)
        else:
            ph_factor = 1.0

        for sp in range(self.n_plants):
            props = self.plant_props[sp]
            growth_rate = props[0]
            pref_temp = props[3]

            # Temperature match: gaussian around preferred temp
            temp_match = np.exp(-4.0 * (temperature - pref_temp) ** 2)

            # Water requirement
            water_factor = np.clip(water * 3.0, 0.1, 1.0)

            # Light
            light_factor = 0.3 + 0.7 * light_level

            # Growth: logistic with environmental factors
            growth = (growth_rate * temp_match * water_factor * light_factor
                      * ph_factor * self.plants[:, :, sp] * (1.0 - total))

            self.plants[:, :, sp] += growth

            # Natural decay
            self.plants[:, :, sp] *= 0.998

            # Herbivore consumption
            herb_consumption = self.fauna[:, :, 0] * self.plants[:, :, sp] * 0.1
            self.plants[:, :, sp] -= herb_consumption
            # Dead plant matter from decay
            self.dead_matter += self.plants[:, :, sp] * 0.001

        np.clip(self.plants, 0, 1, out=self.plants)

    def _update_fauna(self):
        """Fauna dynamics: herbivores eat plants, predators eat herbivores+agents."""
        # Herbivores: grow where plants are, starve where none
        total_plants = np.sum(self.plants, axis=2)
        herb = self.fauna[:, :, 0]
        pred = self.fauna[:, :, 1]

        # Herbivore growth from eating plants
        herb_food = np.clip(total_plants - 0.1, 0, 1)
        herb_growth = self.fauna_props[0, 3] * herb * herb_food * (1.0 - herb)
        herb_death = 0.01 * herb * (1.0 - herb_food) + pred * herb * 0.2
        self.fauna[:, :, 0] += herb_growth - herb_death

        # Predator growth from eating herbivores
        pred_food = np.clip(herb - 0.05, 0, 1)
        pred_growth = self.fauna_props[1, 3] * pred * pred_food * (1.0 - pred)
        pred_death = 0.005 * pred * (1.0 - pred_food)
        self.fauna[:, :, 1] += pred_growth - pred_death

        # Dead fauna -> dead matter
        self.dead_matter += (herb_death + pred_death) * 0.5

        # Fauna diffusion (movement)
        for f in range(self.n_fauna):
            speed = self.fauna_props[f, 0]
            c = self.fauna[:, :, f]
            avg = (np.roll(c, 1, 0) + np.roll(c, -1, 0) +
                   np.roll(c, 1, 1) + np.roll(c, -1, 1)) / 4.0
            c += speed * 0.05 * (avg - c)
            self.fauna[:, :, f] = c

        np.clip(self.fauna, 0, 1, out=self.fauna)

    def _update_decomposition(self):
        """Decomposers convert dead matter into fertility."""
        decomp = self.fauna[:, :, 2]

        # Decomposer growth from dead matter
        food = np.clip(self.dead_matter - 0.01, 0, 1)
        growth = self.fauna_props[2, 3] * decomp * food * (1.0 - decomp)
        death = 0.008 * decomp * (1.0 - food)
        self.fauna[:, :, 2] += growth - death

        # Decomposition: dead matter -> nothing (converted to fertility in physics)
        decomp_rate = decomp * 0.05
        consumed = np.minimum(decomp_rate, self.dead_matter)
        self.dead_matter -= consumed

        # Natural decay of dead matter
        self.dead_matter *= 0.995

        np.clip(self.dead_matter, 0, 1, out=self.dead_matter)
        np.clip(self.fauna[:, :, 2], 0, 1, out=self.fauna[:, :, 2])

    def get_plant_nutrition(self, x, y):
        """What an agent gets from eating plants at this location.
        Returns (total_nutritive, total_toxicity)."""
        x, y = x % self.w, y % self.h
        conc = self.plants[x, y]
        total_nut = 0.0
        total_tox = 0.0
        for sp in range(self.n_plants):
            if conc[sp] > 0.01:
                total_nut += conc[sp] * self.plant_props[sp, 1]
                total_tox += conc[sp] * self.plant_props[sp, 2]
        return total_nut, total_tox

    def consume_plants(self, x, y, amount=0.1):
        """Agent eats plants at location. Returns (energy_gained, damage_taken)."""
        x, y = x % self.w, y % self.h
        total_nut = 0.0
        total_tox = 0.0

        for sp in range(self.n_plants):
            available = self.plants[x, y, sp]
            consumed = min(amount / self.n_plants, available)
            if consumed > 0.001:
                self.plants[x, y, sp] -= consumed
                total_nut += consumed * self.plant_props[sp, 1]
                total_tox += consumed * self.plant_props[sp, 2]

        return total_nut, total_tox

    def get_predator_danger(self, x, y):
        """Predator attack probability at this location."""
        x, y = x % self.w, y % self.h
        pred_density = self.fauna[x, y, 1]
        aggression = self.fauna_props[1, 1]
        return float(pred_density * aggression)

    def add_dead_matter(self, x, y, amount=0.1):
        """When an agent dies, add organic matter."""
        x, y = x % self.w, y % self.h
        self.dead_matter[x, y] = min(1.0, self.dead_matter[x, y] + amount)

    def regime_shift(self, rng, tick):
        """Phase 15: Shift plant nutritive/toxicity — invalidates cached knowledge."""
        if self.n_plants < 2:
            return
        sp_a, sp_b = rng.choice(self.n_plants, size=2, replace=False)
        drift = 0.3
        nut_a, nut_b = self.plant_props[sp_a, 1], self.plant_props[sp_b, 1]
        tox_a, tox_b = self.plant_props[sp_a, 2], self.plant_props[sp_b, 2]
        self.plant_props[sp_a, 1] = nut_a * (1 - drift) + nut_b * drift
        self.plant_props[sp_b, 1] = nut_b * (1 - drift) + nut_a * drift
        self.plant_props[sp_a, 2] = tox_a * (1 - drift) + tox_b * drift
        self.plant_props[sp_b, 2] = tox_b * (1 - drift) + tox_a * drift

    def get_plant_density(self, x, y):
        """Total plant density at position."""
        return float(np.sum(self.plants[x % self.w, y % self.h]))

    def get_herbivore_density(self, x, y):
        return float(self.fauna[x % self.w, y % self.h, 0])

    def get_predator_density(self, x, y):
        return float(self.fauna[x % self.w, y % self.h, 1])

    def get_stats(self):
        return {
            "total_plant_biomass": float(np.sum(self.plants)),
            "total_herbivores": float(np.sum(self.fauna[:, :, 0])),
            "total_predators": float(np.sum(self.fauna[:, :, 1])),
            "total_decomposers": float(np.sum(self.fauna[:, :, 2])),
            "dead_matter": float(np.sum(self.dead_matter)),
        }
