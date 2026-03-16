
from dataclasses import dataclass


@dataclass
class Config:
    world_width: int = 192
    world_height: int = 192
    resource_growth_rate: float = 0.025
    resource_diffusion: float = 0.008
    hazard_spread_rate: float = 0.008
    hazard_decay_rate: float = 0.008
    season_period: int = 1000
    catastrophe_probability: float = 0.001

    initial_population: int = 200
    max_population: int = 2000
    base_metabolic_rate: float = 0.003
    brain_metabolic_cost: float = 0.000003
    stm_capacity: int = 8
    ltm_capacity: int = 32
    max_sensor_range: int = 8
    max_sensor_cone: float = 3.14
    sensor_resolution: int = 5

    hidden_size: int = 32
    gru_hidden_size: int = 16
    action_dim: int = 12
    self_model_hidden: int = 16

    signal_dim: int = 4
    hear_radius: int = 8

    reproduction_threshold: float = 0.65
    reproduction_cost: float = 0.35
    maturity_age: int = 40
    base_mutation_rate: float = 0.05
    max_lifespan: int = 1500

    learning_rate: float = 0.01
    discount_gamma: float = 0.9
    exploration_bonus_scale: float = 0.1

    min_population: int = 30
    extinction_recovery_count: int = 50

    seed: int = 42
    max_ticks: int = 50000
    log_interval: int = 100
    snapshot_interval: int = 100

    think_energy_cost: float = 0.001

    @property
    def sensor_channels(self) -> int:
        return 8

    @property
    def sensor_input_dim(self) -> int:
        return self.sensor_resolution * self.sensor_channels

    @property
    def raw_input_dim(self) -> int:
        return 5 * 5 * 8

    @property
    def proprioception_dim(self) -> int:
        return 7

    @property
    def internal_state_dim(self) -> int:
        return 4

    @property
    def memory_summary_dim(self) -> int:
        return 16

    @property
    def self_model_input_dim(self) -> int:
        return self.gru_hidden_size + self.internal_state_dim + self.proprioception_dim

    @property
    def context_dim(self) -> int:
        return (
            self.proprioception_dim
            + self.internal_state_dim
            + self.memory_summary_dim
            + self.internal_state_dim
            + self.signal_dim
        )

    @property
    def total_brain_input_dim(self) -> int:
        return (
            self.sensor_input_dim
            + self.proprioception_dim
            + self.internal_state_dim
            + self.memory_summary_dim
            + self.internal_state_dim
        )
