
import numpy as np

MAX_SUBSTANCES = 64
NOVELTY_THRESHOLD = 0.18  # squared distance above which a new substance is born


class Chemistry:
    """Open-ended property-based chemistry.

    Starts with 16 seed substances. Reactions between substances produce
    blended property vectors. When a blend is sufficiently novel (far from
    all existing substances), a NEW substance is created — up to 64 total.

    Each substance has a 6D property vector:
    [reactivity, volatility, toxicity, nutritive, catalytic, stability]
    """

    PROP_REACTIVITY = 0
    PROP_VOLATILITY = 1
    PROP_TOXICITY = 2
    PROP_NUTRITIVE = 3
    PROP_CATALYTIC = 4
    PROP_STABILITY = 5
    N_PROPERTIES = 6

    # Substance categories (for base 16 only)
    CAT_PRIMORDIAL = 0  # 0-3: common, stable, everywhere
    CAT_MINERAL = 1     # 4-7: clustered deposits, stable
    CAT_ORGANIC = 2     # 8-11: produced by reactions, nutritive or toxic
    CAT_VOLATILE = 3    # 12-15: reactive, evaporate fast, diffuse

    def __init__(self, width, height, n_substances=16, rng=None):
        self.width = width
        self.height = height
        self.n_base = n_substances  # original seed substances
        self.n_substances = n_substances  # grows as new substances emerge
        self.rng = rng or np.random.default_rng(42)

        # Pre-allocate everything to MAX_SUBSTANCES
        self.substances = np.zeros((width, height, MAX_SUBSTANCES), dtype=np.float32)
        self.substance_props = np.zeros((MAX_SUBSTANCES, self.N_PROPERTIES), dtype=np.float32)
        self.affinity_matrix = np.zeros((MAX_SUBSTANCES, MAX_SUBSTANCES), dtype=np.float32)
        self.reaction_products = np.full((MAX_SUBSTANCES, MAX_SUBSTANCES), -1, dtype=np.int32)
        self.yield_matrix = np.zeros((MAX_SUBSTANCES, MAX_SUBSTANCES), dtype=np.float32)
        self.heat_matrix = np.zeros((MAX_SUBSTANCES, MAX_SUBSTANCES), dtype=np.float32)
        self.catalyst_for = np.full((MAX_SUBSTANCES, MAX_SUBSTANCES), -1, dtype=np.int32)
        self.diffusion_rates = np.zeros(MAX_SUBSTANCES, dtype=np.float32)

        # Track which substances are emergent (born from reactions)
        self.substance_origin = np.full(MAX_SUBSTANCES, -1, dtype=np.int32)  # -1 = seed
        self.substance_born_tick = np.full(MAX_SUBSTANCES, -1, dtype=np.int32)
        self._creation_tick = 0

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
                self.rng.uniform(0.05, 0.2),
                self.rng.uniform(0.05, 0.15),
                self.rng.uniform(0.0, 0.1),
                self.rng.uniform(0.1, 0.4),
                self.rng.uniform(0.0, 0.15),
                self.rng.uniform(0.7, 0.95),
            ]

        # Minerals (4-7): stable, non-volatile, potentially catalytic
        for i in range(4, 8):
            props[i] = [
                self.rng.uniform(0.1, 0.4),
                self.rng.uniform(0.0, 0.1),
                self.rng.uniform(0.0, 0.2),
                self.rng.uniform(0.0, 0.1),
                self.rng.uniform(0.1, 0.6),
                self.rng.uniform(0.6, 0.9),
            ]

        # Organics (8-11): produced by reactions, nutritive OR toxic
        for i in range(8, 12):
            is_toxic = self.rng.random() > 0.5
            props[i] = [
                self.rng.uniform(0.2, 0.5),
                self.rng.uniform(0.1, 0.3),
                self.rng.uniform(0.5, 0.9) if is_toxic else self.rng.uniform(0.0, 0.15),
                self.rng.uniform(0.0, 0.1) if is_toxic else self.rng.uniform(0.4, 0.9),
                self.rng.uniform(0.0, 0.2),
                self.rng.uniform(0.3, 0.6),
            ]

        # Volatiles (12-15): reactive, evaporate fast, spread quickly
        for i in range(12, 16):
            props[i] = [
                self.rng.uniform(0.5, 0.95),
                self.rng.uniform(0.5, 0.9),
                self.rng.uniform(0.1, 0.5),
                self.rng.uniform(0.0, 0.2),
                self.rng.uniform(0.0, 0.3),
                self.rng.uniform(0.1, 0.4),
            ]

        # Diffusion rates for base substances
        for i in range(self.n_base):
            self.diffusion_rates[i] = 0.005 + 0.03 * props[i, self.PROP_VOLATILITY]

    def _init_reactions(self):
        """Precompute reaction affinities, products, yields, and catalysts."""
        self._compute_all_affinities()
        self._compute_all_products()
        self._compute_all_catalysts()

    def _compute_all_affinities(self):
        """Compute affinity between every pair of active substances."""
        props = self.substance_props
        n = self.n_substances
        for i in range(n):
            for j in range(i + 1, n):
                self._compute_affinity(i, j, props)

    def _compute_affinity(self, i, j, props=None):
        if props is None:
            props = self.substance_props
        r_prod = props[i, self.PROP_REACTIVITY] * props[j, self.PROP_REACTIVITY]
        s_avg = (props[i, self.PROP_STABILITY] + props[j, self.PROP_STABILITY]) / 2
        affinity = r_prod * (1.0 - s_avg * 0.7)
        self.affinity_matrix[i, j] = affinity
        self.affinity_matrix[j, i] = affinity

    def _compute_all_products(self):
        """Compute reaction products for common pairs. Rare pairs left for runtime discovery.

        Only pre-compute products for pairs where BOTH reactants are in the same
        category (primordial+primordial, mineral+mineral, etc.) or where affinity
        is below 0.06. Cross-category high-affinity pairs are left uncomputed —
        they'll be discovered at runtime when substances actually meet, potentially
        creating new emergent substances.
        """
        props = self.substance_props
        n = self.n_substances
        for i in range(n):
            for j in range(i + 1, n):
                if self.affinity_matrix[i, j] < 0.02:
                    continue
                # Pre-compute only "obvious" reactions; leave rare ones for runtime
                same_category = (i // 4) == (j // 4)
                low_affinity = self.affinity_matrix[i, j] < 0.06
                if same_category or low_affinity:
                    self._compute_product(i, j, props)
                # else: left with reaction_products[i,j] = -1, discovered at runtime

    def _compute_product(self, i, j, props=None):
        """Compute product of reaction i+j. May create a new substance."""
        if props is None:
            props = self.substance_props

        blend = (props[i] + props[j]) / 2.0
        blend[self.PROP_STABILITY] = min(1.0, blend[self.PROP_STABILITY] + 0.15)
        blend[self.PROP_REACTIVITY] = max(0.0, blend[self.PROP_REACTIVITY] - 0.1)
        if props[i, self.PROP_TOXICITY] > 0.3 or props[j, self.PROP_TOXICITY] > 0.3:
            blend[self.PROP_TOXICITY] = min(1.0, blend[self.PROP_TOXICITY] + 0.1)
        # Nonlinear mixing: cross-category reactions push properties beyond averages
        # This is what makes emergent substances genuinely novel
        cat_i, cat_j = i // 4, j // 4
        if i < self.n_base and j < self.n_base and cat_i != cat_j:
            # Synergy: properties multiply rather than average for dissimilar reactants
            diff = np.abs(props[i] - props[j])
            # Where reactants differ most, the product gets pushed further out
            blend += self.rng.uniform(-0.15, 0.25, self.N_PROPERTIES) * diff
        elif i >= self.n_base or j >= self.n_base:
            # Second-generation: emergent + anything → more drift
            blend += self.rng.uniform(-0.1, 0.2, self.N_PROPERTIES)
        blend = np.clip(blend, 0.0, 1.0)

        best_k = self._find_or_create_product(blend, i, j)

        self.reaction_products[i, j] = best_k
        self.reaction_products[j, i] = best_k

        self.yield_matrix[i, j] = 0.5 + 0.5 * props[best_k, self.PROP_STABILITY]
        self.yield_matrix[j, i] = self.yield_matrix[i, j]

        reactant_stability = (props[i, self.PROP_STABILITY] + props[j, self.PROP_STABILITY]) / 2
        product_stability = props[best_k, self.PROP_STABILITY]
        self.heat_matrix[i, j] = (product_stability - reactant_stability) * 0.3
        self.heat_matrix[j, i] = self.heat_matrix[i, j]

    def _find_or_create_product(self, blend, reactant_i, reactant_j):
        """Find nearest existing substance or create a new emergent one."""
        props = self.substance_props
        n = self.n_substances

        best_dist = float('inf')
        best_k = -1
        for k in range(n):
            if k == reactant_i or k == reactant_j:
                continue
            dist = float(np.sum((props[k] - blend) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_k = k

        # If blend is novel enough and we have room, create a new substance
        if best_dist > NOVELTY_THRESHOLD and n < MAX_SUBSTANCES:
            best_k = self._create_substance(blend)

        return best_k

    def _create_substance(self, properties):
        """Birth a new emergent substance. Returns its index."""
        idx = self.n_substances
        self.n_substances += 1

        self.substance_props[idx] = np.clip(properties, 0.0, 1.0)
        self.diffusion_rates[idx] = 0.005 + 0.03 * properties[self.PROP_VOLATILITY]
        self.substance_origin[idx] = 1  # emergent
        self.substance_born_tick[idx] = self._creation_tick

        # Compute affinities with all existing substances
        for k in range(idx):
            self._compute_affinity(k, idx)

        # Compute products for reactions involving this new substance
        # (but don't recurse — new products from this won't create further substances
        #  until the next _init_reactions cycle, preventing chain creation)
        for k in range(idx):
            if self.affinity_matrix[k, idx] >= 0.02:
                self._compute_product_no_create(k, idx)

        return idx

    def _compute_product_no_create(self, i, j):
        """Compute product without creating new substances (prevents cascade)."""
        props = self.substance_props
        blend = (props[i] + props[j]) / 2.0
        blend[self.PROP_STABILITY] = min(1.0, blend[self.PROP_STABILITY] + 0.15)
        blend[self.PROP_REACTIVITY] = max(0.0, blend[self.PROP_REACTIVITY] - 0.1)
        if props[i, self.PROP_TOXICITY] > 0.3 or props[j, self.PROP_TOXICITY] > 0.3:
            blend[self.PROP_TOXICITY] = min(1.0, blend[self.PROP_TOXICITY] + 0.1)
        blend = np.clip(blend, 0.0, 1.0)

        best_dist = float('inf')
        best_k = -1
        for k in range(self.n_substances):
            if k == i or k == j:
                continue
            dist = float(np.sum((props[k] - blend) ** 2))
            if dist < best_dist:
                best_dist = dist
                best_k = k

        if best_k < 0:
            return

        self.reaction_products[i, j] = best_k
        self.reaction_products[j, i] = best_k
        self.yield_matrix[i, j] = 0.5 + 0.5 * props[best_k, self.PROP_STABILITY]
        self.yield_matrix[j, i] = self.yield_matrix[i, j]
        reactant_stability = (props[i, self.PROP_STABILITY] + props[j, self.PROP_STABILITY]) / 2
        product_stability = props[best_k, self.PROP_STABILITY]
        self.heat_matrix[i, j] = (product_stability - reactant_stability) * 0.3
        self.heat_matrix[j, i] = self.heat_matrix[i, j]

    def _discover_new_reactions(self):
        """Runtime: scan for substance pairs missing affinities or products.
        Called periodically — allows emergent substances to react with each other."""
        n = self.n_substances
        for i in range(n):
            for j in range(i + 1, n):
                # Compute affinity if missing (new substance pairs)
                if self.affinity_matrix[i, j] < 1e-9:
                    self._compute_affinity(i, j)
                # Compute product if affinity exists but no product yet
                if self.affinity_matrix[i, j] >= 0.02 and self.reaction_products[i, j] < 0:
                    self._compute_product(i, j)
        # Recompute catalysts for any new substances
        self._compute_all_catalysts()

    def _compute_all_catalysts(self):
        props = self.substance_props
        n = self.n_substances
        for c in range(n):
            if props[c, self.PROP_CATALYTIC] < 0.3:
                continue
            for i in range(n):
                for j in range(i + 1, n):
                    if self.affinity_matrix[i, j] < 0.01:
                        continue
                    if c == i or c == j:
                        continue
                    if self.catalyst_for[i, j] >= 0:
                        continue  # already has a catalyst
                    between_count = 0
                    for p in range(self.N_PROPERTIES):
                        lo = min(props[i, p], props[j, p])
                        hi = max(props[i, p], props[j, p])
                        if lo - 0.1 <= props[c, p] <= hi + 0.1:
                            between_count += 1
                    if between_count >= 3:
                        self.catalyst_for[i, j] = c
                        self.catalyst_for[j, i] = c

    def _init_medicine_recipes(self):
        """Generate 2-4 medicine recipes that agents must discover."""
        n_recipes = self.rng.integers(2, 5)
        for _ in range(n_recipes):
            a = self.rng.integers(0, self.n_base)
            b = self.rng.integers(0, self.n_base)
            while b == a:
                b = self.rng.integers(0, self.n_base)
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
        """Main chemistry update. Called from main loop."""
        self._creation_tick = tick

        # Reactions every 4 ticks
        if tick % 4 == 0:
            self._run_reactions(temperature)

        # Diffusion every 2 ticks
        if tick % 2 == 0:
            self._diffuse(temperature)

        # Decay unstable substances
        if tick % 8 == 0:
            self._decay()

        # Runtime discovery: compute missing reaction products (may create substances)
        if tick > 0 and tick % 50 == 0:
            self._discover_new_reactions()

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
                    # Runtime: compute product on-the-fly (may create new substance)
                    self._compute_product(i, j)
                    prod_idx = self.reaction_products[i, j]
                    if prod_idx < 0:
                        continue

                ci = conc[:, :, i]
                cj = conc[:, :, j]

                mask = (ci > 0.01) & (cj > 0.01)
                if not np.any(mask):
                    continue

                rate = aff * np.minimum(ci, cj) * 0.02
                temp_factor = np.clip(temperature * 2.0, 0.1, 2.0)
                rate *= temp_factor

                cat_idx = self.catalyst_for[i, j]
                if cat_idx >= 0:
                    cat_present = conc[:, :, cat_idx] > 0.05
                    rate = np.where(cat_present, rate * 3.0, rate)

                rate *= mask

                consumed = np.minimum(rate, np.minimum(ci, cj))
                conc[:, :, i] -= consumed
                conc[:, :, j] -= consumed
                conc[:, :, prod_idx] += consumed * self.yield_matrix[i, j]

                heat_produced += consumed * self.heat_matrix[i, j]

        np.clip(conc, 0, 1, out=conc)
        return heat_produced

    def _diffuse(self, temperature):
        """Diffuse substances based on volatility."""
        for i in range(self.n_substances):
            c = self.substances[:, :, i]
            if np.max(c) < 0.001:
                continue

            rate = self.diffusion_rates[i]
            vol = self.substance_props[i, self.PROP_VOLATILITY]
            if vol > 0.3:
                effective_rate = rate * (1.0 + temperature * vol)
            else:
                effective_rate = rate

            avg = (
                np.roll(c, 1, 0) + np.roll(c, -1, 0) +
                np.roll(c, 1, 1) + np.roll(c, -1, 1)
            ) / 4.0

            c += effective_rate * (avg - c)

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
        """Get all substance concentrations at a position."""
        return self.substances[x % self.width, y % self.height, :self.n_substances].copy()

    def get_top_substances(self, x, y, n=4):
        """Get indices and concentrations of top-n substances at position."""
        conc = self.substances[x % self.width, y % self.height, :self.n_substances]
        indices = np.argsort(conc)[-n:][::-1]
        return indices, conc[indices]

    def consume_substance(self, x, y, substance_idx, amount=0.1):
        """Agent consumes a substance. Returns actual amount consumed."""
        if substance_idx < 0 or substance_idx >= self.n_substances:
            return 0.0
        x, y = x % self.width, y % self.height
        available = self.substances[x, y, substance_idx]
        consumed = min(amount, available)
        self.substances[x, y, substance_idx] -= consumed
        return consumed

    def deposit_substance(self, x, y, substance_idx, amount):
        """Agent deposits a substance at location."""
        if substance_idx < 0 or substance_idx >= self.n_substances:
            return
        x, y = x % self.width, y % self.height
        self.substances[x, y, substance_idx] = min(
            1.0, self.substances[x, y, substance_idx] + amount)

    def deposit_composition(self, x, y, composition):
        """Deposit an entire composition array at location (e.g. body decomposition)."""
        x, y = x % self.width, y % self.height
        n = min(len(composition), self.n_substances)
        self.substances[x, y, :n] += composition[:n]
        np.clip(self.substances[x, y], 0, 1, out=self.substances[x, y])

    def get_nutritive_value(self, substance_idx):
        """How much energy an agent gets from eating this substance."""
        if substance_idx < 0 or substance_idx >= MAX_SUBSTANCES:
            return 0.0
        return float(self.substance_props[substance_idx, self.PROP_NUTRITIVE])

    def get_toxicity(self, substance_idx):
        """How much damage an agent takes from this substance."""
        if substance_idx < 0 or substance_idx >= MAX_SUBSTANCES:
            return 0.0
        return float(self.substance_props[substance_idx, self.PROP_TOXICITY])

    def check_medicine(self, x, y, temperature_here):
        """Check if medicine can be created at this location."""
        x, y = x % self.width, y % self.height
        conc = self.substances[x, y]

        for a, b, min_t, max_t, heal in self.medicine_recipes:
            if conc[a] > 0.05 and conc[b] > 0.05:
                if min_t <= temperature_here <= max_t:
                    used = min(0.05, conc[a], conc[b])
                    self.substances[x, y, a] -= used
                    self.substances[x, y, b] -= used
                    return heal * (used / 0.05)
        return 0.0

    def get_reaction_count(self):
        """How many active reactions exist."""
        n = self.n_substances
        return int(np.sum(self.affinity_matrix[:n, :n] > 0.02)) // 2

    def get_stats(self):
        """Summary statistics for logging."""
        n = self.n_substances
        total = float(np.sum(self.substances[:, :, :n]))
        per_substance = np.sum(self.substances[:, :, :n], axis=(0, 1))
        dominant = int(np.argmax(per_substance))
        n_emergent = int(np.sum(self.substance_origin[:n] >= 0))
        return {
            "total_substance_mass": total,
            "dominant_substance": dominant,
            "n_active_reactions": self.get_reaction_count(),
            "n_medicine_recipes": len(self.medicine_recipes),
            "n_substances": n,
            "n_emergent_substances": n_emergent,
        }
