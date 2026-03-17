
import numpy as np
from agents.genome import get_trait


class InternalState:
    def __init__(self, genome: np.ndarray):
        self.hunger = 0.3
        self.fear = 0.0
        self.curiosity = 0.5
        self.temperature_comfort = 0.5

        # Phase 5A: Social emotional needs
        self.social_need = 0.0
        self.trust_state = 0.5
        self.social_satisfaction = 0.0

        self._hunger_k = get_trait(genome, "hunger_sensitivity")
        self._fear_k = get_trait(genome, "fear_sensitivity")
        self._curiosity_k = get_trait(genome, "curiosity_sensitivity")

        # Social sensitivity genes
        self._social_sensitivity_k = get_trait(genome, "social_sensitivity")
        self._trust_sensitivity_k = get_trait(genome, "trust_sensitivity")
        self._social_reward_k = get_trait(genome, "social_reward_sensitivity")

    def update(self, energy: float, health: float, sensor_vec: np.ndarray,
               surprise: float, temperature: float = 0.5,
               nearby_agents: int = 0,
               positive_interaction: float = 0.0,
               negative_interaction: float = 0.0,
               social_reward: float = 0.0):
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

        # Social need: continuous response to nearby agent count
        # Genome sensitivity controls scale, not hardcoded "3 agents = full"
        isolation = 1.0 / (1.0 + nearby_agents)  # smooth: 1→0.5→0.33→0.25...
        target_social = isolation * self._social_sensitivity_k
        self.social_need += dt * (target_social - self.social_need)

        # Trust state: drifts based on interaction outcomes
        # Genome sensitivity controls how reactive, not hardcoded weights
        trust_delta = (positive_interaction - negative_interaction) * self._trust_sensitivity_k * 0.1
        target_trust = np.clip(self.trust_state + trust_delta, 0, 1)
        self.trust_state += dt * (target_trust - self.trust_state)
        # Natural decay toward neutral — prevents saturation
        self.trust_state += dt * (0.5 - self.trust_state) * 0.05

        # Social satisfaction: spikes on social reward, decays naturally
        decay = self.social_satisfaction * 0.03  # proportional decay, not fixed
        target_sat = max(0, self.social_satisfaction - decay) + social_reward * self._social_reward_k * 0.3
        self.social_satisfaction += dt * (target_sat - self.social_satisfaction)

        self.hunger = float(np.clip(self.hunger, 0, 1))
        self.fear = float(np.clip(self.fear, 0, 1))
        self.curiosity = float(np.clip(self.curiosity, 0, 1))
        self.temperature_comfort = float(np.clip(self.temperature_comfort, 0, 1))
        self.social_need = float(np.clip(self.social_need, 0, 1))
        self.trust_state = float(np.clip(self.trust_state, 0, 1))
        self.social_satisfaction = float(np.clip(self.social_satisfaction, 0, 1))

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.hunger, self.fear, self.curiosity, self.temperature_comfort,
            self.social_need, self.trust_state, self.social_satisfaction,
        ])
