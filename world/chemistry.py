
import numpy as np


class Chemistry:
    """Property-based chemistry system with 16 substances.

    Each substance has a 6D property vector:
    [reactivity, volatility, toxicity, nutritive, catalytic, stability]

    Reactions emerge from property interactions — not hardcoded.
    Different world seeds produce different chemistries.
    """

    PROP_REACTIVITY = 0
    PROP_VOLATILITY = 1
    PROP_TOXICITY = 2
    PROP_NUTRITIVE = 3
    PROP_CATALYTIC = 4
    PROP_STABILITY = 5
    N_PROPERTIES = 6

    # Substance categories
    CAT_PRIMORDIAL = 0  # 0-3: common, stable, everywhere
    CAT_MINERAL = 1     # 4-7: clustered deposits, stable
    CAT_ORGANIC = 2     # 8-11: produced by reactions, nutritive or toxic
    CAT_VOLATILE = 3    # 12-15: reactive, evaporate fast, diffuse

    def __init__(self, width, height, n_substances=16, rng=None):
        self.width = width
        self.height = height
        self.n_substances = n_substances
        self.rng = rng or np.random.default_rng(42)

        # Concentration of each substance at each cell
        self.substances = np.zeros((width, height, n_substances), dtype=np.float32)

        # Property matrix: what each substance IS
        self.substance_props = np.zeros((n_substances, self.N_PROPERTIES), dtype=np.float32)

        # Precomputed reaction data
        self.affinity_matrix = np.zeros((n_substances, n_substances), dtype=np.float32)
        self.reaction_products = np.full((n_substances, n_substances), -1, dtype=np.int32)
        self.yield_matrix = np.zeros((n_substances, n_substances), dtype=np.float32)
        self.heat_matrix = np.zeros((n_substances, n_substances), dtype=np.float32)

        # Catalyst lookup: catalyst_for[i,j] = substance index that catalyzes i+j, or -1
        self.catalyst_for = np.full((n_substances, n_substances), -1, dtype=np.int32)

        # Diffusion rates per substance (volatiles diffuse faster)
        self.diffusion_rates = np.zeros(n_substances, dtype=np.float32)

        # Medicine recipes: list of (A, B, min_temp, max_temp, heal_amount)
        self.medicine_recipes = []

        self._init_substances()
        self._init_reactions()
        self._init_medicine_recipes()
        self._distribute_initial()

    def _init_substances(self):
        """Generate substance properties from RNG. Different seed = different chemistry."""
        props = self.substance_props

        # Primordial (0-3): high stability, low reactivity, moderate everything
        for i in range(4):
            props[i] = [
                self.rng.uniform(0.05, 0.2),   # reactivity: low
                self.rng.uniform(0.05, 0.15),   # volatility: low
                self.rng.uniform(0.0, 0.1),     # toxicity: very low
                self.rng.uniform(0.1, 0.4),     # nutritive: some food value
                self.rng.uniform(0.0, 0.15),    # catalytic: low
                self.rng.uniform(0.7, 0.95),    # stability: high
            ]

        # Minerals (4-7): stable, non-volatile, potentially catalytic
        for i in range(4, 8):
            props[i] = [
                self.rng.uniform(0.1, 0.4),     # reactivity: moderate
                self.rng.uniform(0.0, 0.1),     # volatility: very low
                self.rng.uniform(0.0, 0.2),     # toxicity: low
                self.rng.uniform(0.0, 0.1),     # nutritive: barely edible
                self.rng.uniform(0.1, 0.6),     # catalytic: can be high
                self.rng.uniform(0.6, 0.9),     # stability: high
            ]

        # Organics (8-11): produced by reactions, nutritive OR toxic
        for i in range(8, 12):
            is_toxic = self.rng.random() > 0.5
            props[i] = [
                self.rng.uniform(0.2, 0.5),     # reactivity: moderate
                self.rng.uniform(0.1, 0.3),     # volatility: some
                self.rng.uniform(0.5, 0.9) if is_toxic else self.rng.uniform(0.0, 0.15),
                self.rng.uniform(0.0, 0.1) if is_toxic else self.rng.uniform(0.4, 0.9),
                self.rng.uniform(0.0, 0.2),     # catalytic: low
                self.rng.uniform(0.3, 0.6),     # stability: medium
            ]

        # Volatiles (12-15): reactive, evaporate fast, spread quickly
        for i in range(12, 16):
            props[i] = [
                self.rng.uniform(0.5, 0.95),    # reactivity: high
                self.rng.uniform(0.5, 0.9),     # volatility: high
                self.rng.uniform(0.1, 0.5),     # toxicity: variable
                self.rng.uniform(0.0, 0.2),     # nutritive: low
                self.rng.uniform(0.0, 0.3),     # catalytic: some
                self.rng.uniform(0.1, 0.4),     # stability: low
            ]

        # Diffusion rates: proportional to volatility
        self.diffusion_rates = 0.005 + 0.03 * props[:, self.PROP_VOLATILITY]

    def _init_reactions(self):
        """Precompute reaction affinities, products, yields, and catalysts."""
        props = self.substance_props
        n = self.n_substances

        # Affinity: reactivity product * (1 - stability average)
        # High reactivity + low stability = reacts easily
        for i in range(n):
            for j in range(i + 1, n):
                r_prod = props[i, self.PROP_REACTIVITY] * props[j, self.PROP_REACTIVITY]
                s_avg = (props[i, self.PROP_STABILITY] + props[j, self.PROP_STABILITY]) / 2
                affinity = r_prod * (1.0 - s_avg * 0.7)
                self.affinity_matrix[i, j] = affinity
                self.affinity_matrix[j, i] = affinity

        # Products: blend properties, find nearest existing substance
        for i in range(n):
            for j in range(i + 1, n):
                if self.affinity_matrix[i, j] < 0.02:
                    continue  # no reaction

                # Product properties = weighted blend with nonlinear mixing
                blend = (props[i] + props[j]) / 2
                # Reactions tend to increase stability and decrease reactivity
                blend[self.PROP_STABILITY] = min(1.0, blend[self.PROP_STABILITY] + 0.15)
                blend[self.PROP_REACTIVITY] = max(0.0, blend[self.PROP_REACTIVITY] - 0.1)
                # Toxicity can emerge from mixing
                if props[i, self.PROP_TOXICITY] > 0.3 or props[j, self.PROP_TOXICITY] > 0.3:
                    blend[self.PROP_TOXICITY] = min(1.0, blend[self.PROP_TOXICITY] + 0.1)

                # Find nearest substance (excluding reactants)
                best_dist = float('inf')
                best_k = -1
                for k in range(n):
                    if k == i or k == j:
                        continue
                    dist = np.sum((props[k] - blend) ** 2)
                    if dist < best_dist:
                        best_dist = dist
                        best_k = k

                self.reaction_products[i, j] = best_k
                self.reaction_products[j, i] = best_k

                # Yield: inverse of stability difference (stable products form easier)
                self.yield_matrix[i, j] = 0.5 + 0.5 * props[best_k, self.PROP_STABILITY]
                self.yield_matrix[j, i] = self.yield_matrix[i, j]

                # Heat: exothermic if product is more stable than reactants
                reactant_stability = (props[i, self.PROP_STABILITY] + props[j, self.PROP_STABILITY]) / 2
                product_stability = props[best_k, self.PROP_STABILITY]
                self.heat_matrix[i, j] = (product_stability - reactant_stability) * 0.3
                self.heat_matrix[j, i] = self.heat_matrix[i, j]

        # Catalysts: substances with high catalytic property catalyze reactions
        # between substances they're chemically "between" in property space
        for c in range(n):
            if props[c, self.PROP_CATALYTIC] < 0.3:
                continue
            for i in range(n):
                for j in range(i + 1, n):
                    if self.affinity_matrix[i, j] < 0.01:
                        continue
                    if c == i or c == j:
                        continue
                    # Catalyst works if it's "between" reactants in some property dimension
                    between_count = 0
                    for p in range(self.N_PROPERTIES):
                        lo = min(props[i, p], props[j, p])
                        hi = max(props[i, p], props[j, p])
                        if lo - 0.1 <= props[c, p] <= hi + 0.1:
                            between_count += 1
                    if between_count >= 3:
                        self.catalyst_for[i, j] = c
                        self.catalyst_for[j, i] = c
                        break  # only one catalyst per reaction

    def _init_medicine_recipes(self):
        """Generate 2-4 medicine recipes that agents must discover."""
        n_recipes = self.rng.integers(2, 5)
        for _ in range(n_recipes):
            a = self.rng.integers(0, self.n_substances)
            b = self.rng.integers(0, self.n_substances)
            while b == a:
                b = self.rng.integers(0, self.n_substances)
            min_temp = self.rng.uniform(0.3, 0.5)
            max_temp = min_temp + self.rng.uniform(0.15, 0.3)
            heal = self.rng.uniform(0.05, 0.2)
            self.medicine_recipes.append((int(a), int(b), float(min_temp), float(max_temp), float(heal)))

    def _distribute_initial(self):
        """Place initial substance deposits in the world."""
        W, H = self.width, self.height

        # Primordial: low background everywhere
        for i in range(4):
            self.substances[:, :, i] = self.rng.uniform(0.01, 0.05, (W, H)).astype(np.float32)

        # Minerals: clustered deposits
        n_deposits = max(3, W * H // 3000)
        for i in range(4, 8):
            for _ in range(n_deposits):
                cx = self.rng.integers(0, W)
                cy = self.rng.integers(0, H)
                r = self.rng.integers(3, 8)
                intensity = self.rng.uniform(0.2, 0.6)
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if dx * dx + dy * dy <= r * r:
                            x = (cx + dx) % W
                            y = (cy + dy) % H
                            dist = np.sqrt(dx * dx + dy * dy)
                            self.substances[x, y, i] += intensity * max(0, 1 - dist / r)
            np.clip(self.substances[:, :, i], 0, 1, out=self.substances[:, :, i])

        # Organics: very sparse initial (mostly produced by reactions)
        for i in range(8, 12):
            mask = self.rng.random((W, H)) < 0.02
            self.substances[:, :, i] = mask.astype(np.float32) * self.rng.uniform(0.05, 0.15)

        # Volatiles: scattered pockets
        for i in range(12, 16):
            n_pockets = max(2, W * H // 5000)
            for _ in range(n_pockets):
                cx = self.rng.integers(0, W)
                cy = self.rng.integers(0, H)
                r = self.rng.integers(2, 5)
                intensity = self.rng.uniform(0.1, 0.4)
                for dx in range(-r, r + 1):
                    for dy in range(-r, r + 1):
                        if dx * dx + dy * dy <= r * r:
                            x = (cx + dx) % W
                            y = (cy + dy) % H
                            self.substances[x, y, i] += intensity
            np.clip(self.substances[:, :, i], 0, 1, out=self.substances[:, :, i])

    def update(self, tick, temperature):
        """Main chemistry update. Called from main loop.

        Args:
            tick: current simulation tick
            temperature: (W, H) float32 array of temperatures
        """
        # Reactions every 4 ticks
        if tick % 4 == 0:
            self._run_reactions(temperature)

        # Diffusion every 2 ticks
        if tick % 2 == 0:
            self._diffuse(temperature)

        # Decay unstable substances
        if tick % 8 == 0:
            self._decay()

    def _run_reactions(self, temperature):
        """Run all pairwise reactions. Vectorized numpy."""
        conc = self.substances
        n = self.n_substances
        heat_produced = np.zeros((self.width, self.height), dtype=np.float32)

        for i in range(n):
            for j in range(i + 1, n):
                aff = self.affinity_matrix[i, j]
                if aff < 0.02:
                    continue

                prod_idx = self.reaction_products[i, j]
                if prod_idx < 0:
                    continue

                ci = conc[:, :, i]
                cj = conc[:, :, j]

                # Only react where both present above threshold
                mask = (ci > 0.01) & (cj > 0.01)
                if not np.any(mask):
                    continue

                # Base reaction rate
                rate = aff * np.minimum(ci, cj) * 0.02

                # Temperature factor: Arrhenius-like
                # Higher temp = faster reactions (up to a point)
                temp_factor = np.clip(temperature * 2.0, 0.1, 2.0)
                rate *= temp_factor

                # Catalyst boost
                cat_idx = self.catalyst_for[i, j]
                if cat_idx >= 0:
                    cat_present = conc[:, :, cat_idx] > 0.05
                    rate = np.where(cat_present, rate * 3.0, rate)

                rate *= mask

                # Consume reactants, produce product
                consumed = np.minimum(rate, np.minimum(ci, cj))
                conc[:, :, i] -= consumed
                conc[:, :, j] -= consumed
                conc[:, :, prod_idx] += consumed * self.yield_matrix[i, j]

                # Heat
                heat_produced += consumed * self.heat_matrix[i, j]

        # Clamp all
        np.clip(conc, 0, 1, out=conc)

        return heat_produced

    def _diffuse(self, temperature):
        """Diffuse substances based on volatility. Volatiles spread faster at high temp."""
        for i in range(self.n_substances):
            c = self.substances[:, :, i]
            if np.max(c) < 0.001:
                continue

            rate = self.diffusion_rates[i]

            # Volatiles diffuse faster at high temperature
            vol = self.substance_props[i, self.PROP_VOLATILITY]
            if vol > 0.3:
                effective_rate = rate * (1.0 + temperature * vol)
            else:
                effective_rate = rate

            # 4-neighbor diffusion with wrap
            avg = (
                np.roll(c, 1, 0) + np.roll(c, -1, 0) +
                np.roll(c, 1, 1) + np.roll(c, -1, 1)
            ) / 4.0

            c += effective_rate * (avg - c)

            # Evaporation for volatiles at high temp
            if vol > 0.4:
                evap_rate = 0.002 * vol * temperature
                c -= evap_rate
                np.clip(c, 0, 1, out=c)

            self.substances[:, :, i] = c

    def _decay(self):
        """Unstable substances slowly decay."""
        for i in range(self.n_substances):
            stability = self.substance_props[i, self.PROP_STABILITY]
            if stability < 0.5:
                decay_rate = 0.001 * (1.0 - stability)
                self.substances[:, :, i] *= (1.0 - decay_rate)

    def get_concentrations(self, x, y):
        """Get all substance concentrations at a position. Returns (n_substances,) array."""
        return self.substances[x % self.width, y % self.height].copy()

    def get_top_substances(self, x, y, n=4):
        """Get indices and concentrations of top-n substances at position."""
        conc = self.substances[x % self.width, y % self.height]
        indices = np.argsort(conc)[-n:][::-1]
        return indices, conc[indices]

    def consume_substance(self, x, y, substance_idx, amount=0.1):
        """Agent consumes a substance. Returns actual amount consumed."""
        x, y = x % self.width, y % self.height
        available = self.substances[x, y, substance_idx]
        consumed = min(amount, available)
        self.substances[x, y, substance_idx] -= consumed
        return consumed

    def deposit_substance(self, x, y, substance_idx, amount):
        """Agent deposits a substance at location."""
        x, y = x % self.width, y % self.height
        self.substances[x, y, substance_idx] = min(1.0, self.substances[x, y, substance_idx] + amount)

    def get_nutritive_value(self, substance_idx):
        """How much energy an agent gets from eating this substance."""
        return self.substance_props[substance_idx, self.PROP_NUTRITIVE]

    def get_toxicity(self, substance_idx):
        """How much damage an agent takes from this substance."""
        return self.substance_props[substance_idx, self.PROP_TOXICITY]

    def check_medicine(self, x, y, temperature_here):
        """Check if medicine can be created at this location.
        Returns heal amount or 0."""
        x, y = x % self.width, y % self.height
        conc = self.substances[x, y]

        for a, b, min_t, max_t, heal in self.medicine_recipes:
            if conc[a] > 0.05 and conc[b] > 0.05:
                if min_t <= temperature_here <= max_t:
                    # Consume ingredients
                    used = min(0.05, conc[a], conc[b])
                    self.substances[x, y, a] -= used
                    self.substances[x, y, b] -= used
                    return heal * (used / 0.05)
        return 0.0

    def get_reaction_count(self):
        """How many active reactions exist (for stats)."""
        return int(np.sum(self.affinity_matrix > 0.02)) // 2

    def get_stats(self):
        """Summary statistics for logging."""
        total = np.sum(self.substances)
        per_substance = np.sum(self.substances, axis=(0, 1))
        dominant = int(np.argmax(per_substance))
        return {
            "total_substance_mass": float(total),
            "dominant_substance": dominant,
            "n_active_reactions": self.get_reaction_count(),
            "n_medicine_recipes": len(self.medicine_recipes),
        }
