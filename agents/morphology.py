
import numpy as np

MORPHOLOGY_GENE_COUNT = 10


class BodyTrait:
    __slots__ = ['name', 'value', 'cost_per_tick']

    def __init__(self, name: str, value: float, cost_multiplier: float):
        self.name = name
        self.value = float(np.clip(value, 0, 1))
        self.cost_per_tick = self.value * cost_multiplier


class SensorTrait:
    __slots__ = ['name', 'active', 'sensitivity', 'cost_per_tick', 'extra_input_dim']

    def __init__(self, name: str, gene_value: float, cost: float, input_dim: int):
        self.name = name
        self.active = gene_value > 0.5
        self.sensitivity = float(np.clip(gene_value, 0, 1)) if self.active else 0.0
        self.cost_per_tick = cost if self.active else 0.0
        self.extra_input_dim = input_dim if self.active else 0


class Morphology:

    def __init__(self, morph_genes: np.ndarray):
        genes = np.zeros(MORPHOLOGY_GENE_COUNT)
        n = min(len(morph_genes), MORPHOLOGY_GENE_COUNT)
        genes[:n] = morph_genes[:n]

        self.armor = BodyTrait("armor", genes[0], 0.001)
        self.claws = BodyTrait("claws", genes[1], 0.0008)
        self.speed_mod = BodyTrait("speed_mod", genes[2], 0.0006)
        self.camouflage = BodyTrait("camouflage", genes[3], 0.0005)
        self.thick_skin = BodyTrait("thick_skin", genes[4], 0.0004)

        self.body_traits = [self.armor, self.claws, self.speed_mod,
                            self.camouflage, self.thick_skin]

        self.thermosense = SensorTrait("thermosense", genes[5], 0.0003, 2)
        self.chemosense = SensorTrait("chemosense", genes[6], 0.0004, 2)
        self.scent_track = SensorTrait("scent_track", genes[7], 0.0003, 2)
        self.magnetosense = SensorTrait("magnetosense", genes[8], 0.0002, 2)
        self.vibration = SensorTrait("vibration", genes[9], 0.0003, 1)

        self.sensor_traits = [self.thermosense, self.chemosense,
                              self.scent_track, self.magnetosense, self.vibration]

    @property
    def total_metabolic_cost(self) -> float:
        body_cost = sum(t.cost_per_tick for t in self.body_traits)
        sensor_cost = sum(t.cost_per_tick for t in self.sensor_traits)
        return body_cost + sensor_cost

    @property
    def extra_sensor_dim(self) -> int:
        return sum(t.extra_input_dim for t in self.sensor_traits)

    @property
    def damage_reduction(self) -> float:
        return self.armor.value * 0.4 + self.thick_skin.value * 0.2

    @property
    def toxin_resistance(self) -> float:
        return self.thick_skin.value * 0.5

    @property
    def combat_power(self) -> float:
        return 0.5 + self.claws.value * 1.0

    @property
    def speed_multiplier(self) -> float:
        return 1.0 + self.speed_mod.value * 0.5

    @property
    def visibility(self) -> float:
        return max(0.1, 1.0 - self.camouflage.value * 0.6)

    def get_extra_sensor_input(self, temperature: float, toxin: float,
                                mineral: float, scent: float,
                                abs_x: float, abs_y: float,
                                nearby_movement: float) -> np.ndarray:
        parts = []
        if self.thermosense.active:
            temp_grad = (temperature - 0.5) * self.thermosense.sensitivity
            temp_rate = temperature * self.thermosense.sensitivity
            parts.extend([temp_grad, temp_rate])
        if self.chemosense.active:
            parts.extend([
                toxin * self.chemosense.sensitivity,
                mineral * self.chemosense.sensitivity,
            ])
        if self.scent_track.active:
            parts.extend([
                scent * self.scent_track.sensitivity,
                scent * self.scent_track.sensitivity * 0.5,
            ])
        if self.magnetosense.active:
            parts.extend([
                abs_x * self.magnetosense.sensitivity,
                abs_y * self.magnetosense.sensitivity,
            ])
        if self.vibration.active:
            parts.append(nearby_movement * self.vibration.sensitivity)
        return np.array(parts) if parts else np.array([])

    def describe(self) -> str:
        body = ", ".join(f"{t.name}={t.value:.2f}" for t in self.body_traits if t.value > 0.1)
        sensors = ", ".join(t.name for t in self.sensor_traits if t.active)
        return f"Body[{body}] Sensors[{sensors}] Cost={self.total_metabolic_cost:.4f}/tick"
