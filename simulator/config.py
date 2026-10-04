from pydantic import BaseModel, Field
from typing import Dict, List, Tuple

class PartConfig(BaseModel):
    name: str
    part_class: str = Field(..., pattern="^(wearout|random)$")
    weibull_shape: float
    weibull_scale: float
    repair_time_mean: float  # lognormal mu
    repair_time_sigma: float # lognormal sigma
    price: float

class SimulatorConfig(BaseModel):
    # Fleet
    n_jets: int = 24
    n_bases: int = 2
    
    # Parts (4 wearout, 4 random)
    parts: List[PartConfig] = [
        PartConfig(name="engine_starter", part_class="wearout", weibull_shape=2.5, weibull_scale=1000.0, repair_time_mean=2.0, repair_time_sigma=0.5, price=25000),
        PartConfig(name="hydraulic_pump", part_class="wearout", weibull_shape=2.2, weibull_scale=1200.0, repair_time_mean=1.5, repair_time_sigma=0.4, price=18000),
        PartConfig(name="fuel_pump", part_class="wearout", weibull_shape=2.8, weibull_scale=1500.0, repair_time_mean=1.2, repair_time_sigma=0.3, price=15000),
        PartConfig(name="actuator", part_class="wearout", weibull_shape=2.0, weibull_scale=800.0, repair_time_mean=2.5, repair_time_sigma=0.6, price=30000),
        PartConfig(name="radar_module", part_class="random", weibull_shape=1.0, weibull_scale=2000.0, repair_time_mean=3.0, repair_time_sigma=0.7, price=150000),
        PartConfig(name="ew_unit", part_class="random", weibull_shape=1.0, weibull_scale=2500.0, repair_time_mean=3.5, repair_time_sigma=0.8, price=200000),
        PartConfig(name="mission_computer", part_class="random", weibull_shape=1.0, weibull_scale=3000.0, repair_time_mean=1.0, repair_time_sigma=0.2, price=80000),
        PartConfig(name="power_unit", part_class="random", weibull_shape=1.05, weibull_scale=1800.0, repair_time_mean=2.0, repair_time_sigma=0.5, price=40000)
    ]
    
    # Lots
    n_lots: int = 5
    bad_lot_idx: int = 2
    bad_lot_scale_multiplier: float = 0.5  # fails twice as fast
    
    # Spares
    base_spares_initial: int = 2
    depot_spares_initial: int = 10
    procurement_lead_time_mean: float = 14.0 # days
    
    # Maintenance
    base_bays: int = 3
    cannibalisation_damage_prob: float = 0.15
    
    # Simulation
    horizon_days: int = 90
    sorties_per_day_per_jet: int = 2
    
    # Teardown Confusion Matrix (P(grade_j | true_state_i))
    # States: 1(Good), 2(Degraded), 3(Severe), 4(Failed)
    teardown_confusion: List[List[float]] = [
        [0.90, 0.08, 0.02, 0.00], # True 1
        [0.10, 0.80, 0.10, 0.00], # True 2
        [0.00, 0.15, 0.80, 0.05], # True 3
        [0.00, 0.00, 0.05, 0.95], # True 4
    ]

# Default seed pool for reproducibility
SEEDS = {
    "train_1": 1001,
    "train_2": 1002,
    "eval_1": 2001,
    "twin_A": 3001,
    "twin_B": 3001 # Twin worlds share the seed but differ in 'world' parameter
}
