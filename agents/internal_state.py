
import numpy as np
from agents.genome import get_trait


class InternalState:
    def __init__(self, genome: np.ndarray):
        self.hunger = 0.3
        self.fear = 0.0
        self.curiosity = 0.5
        self.temperature_comfort = 0.5

        self._hunger_k = get_trait(genome, "hunger_sensitivity")
        self._fear_k = get_trait(genome, "fear_sensitivity")
        self._curiosity_k = get_trait(genome, "curiosity_sensitivity")

    def update(self, energy: float, health: float, sensor_vec: np.ndarray,
               surprise: float, temperature: float = 0.5):
        dt = 0.1

        target_hunger = (1.0 - energy) * self._hunger_k
        self.hunger += dt * (target_hunger - self.hunger)

        hazard_signal = 0.0
        toxin_signal = 0.0
        if len(sensor_vec) >= 8:
            hazard_signal = sensor_vec[1::8].sum() if len(sensor_vec) > 1 else 0.0
            toxin_signal = sensor_vec[6::8].sum() if len(sensor_vec) > 6 else 0.0
        threat = hazard_signal + toxin_signal * 0.5 + max(0, 0.5 - health)
        target_fear = threat * self._fear_k
        self.fear += dt * (target_fear - self.fear)

        safety = (1.0 - self.fear) * energy
        novelty_drive = (1.0 - surprise) * 0.3 + safety * 0.3
        target_curiosity = novelty_drive * self._curiosity_k
        self.curiosity += dt * (target_curiosity - self.curiosity)

        temp_discomfort = abs(temperature - 0.5) * 2.0
        self.temperature_comfort += dt * ((1.0 - temp_discomfort) - self.temperature_comfort)

        self.hunger = float(np.clip(self.hunger, 0, 1))
        self.fear = float(np.clip(self.fear, 0, 1))
        self.curiosity = float(np.clip(self.curiosity, 0, 1))
        self.temperature_comfort = float(np.clip(self.temperature_comfort, 0, 1))

    def as_vector(self) -> np.ndarray:
        return np.array([self.hunger, self.fear, self.curiosity, self.temperature_comfort])
