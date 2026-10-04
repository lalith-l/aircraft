"""
Simulator Generator — Core event loop for structural causal model.
"""

import random
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any
from simulator.config import SimulatorConfig, PartConfig

@dataclass
class SimEvent:
    t: float
    event_type: str
    jet_id: str
    part_name: str = ""
    part_sn: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

@dataclass
class GroundTruth:
    part_sn: str
    t: float
    true_health_state: int  # 1 (Good) to 4 (Failed)
    lot_id: str
    
class Simulator:
    def __init__(self, config: SimulatorConfig, seed: int, world: str):
        self.config = config
        self.seed = seed
        self.world = world # "A" or "B" for twin-world counterfactuals
        
        self.rng = np.random.RandomState(seed)
        self.py_rng = random.Random(seed)
        
        # State
        self.current_t = 0.0
        self.events: List[SimEvent] = []
        self.ground_truth: List[GroundTruth] = []
        
        # Parts initialization
        self.part_lots = {} # sn -> lot_id
        self.part_health = {} # sn -> float (cumulative hazard / damage)
        self.part_states = {} # sn -> int (1..4)
        
        # Generate initial parts pool
        self._init_fleet()
        
    def _init_fleet(self):
        """Initialize jets, bases, parts, lots."""
        self.jets = {f"J-{i:02d}": {"base": f"Base-{i%self.config.n_bases}", "parts": {}} 
                     for i in range(self.config.n_jets)}
                     
        # Assign lots
        lots = [f"LOT-{i}" for i in range(self.config.n_lots)]
        bad_lot = lots[self.config.bad_lot_idx]
        
        part_counter = 0
        for jet_id, jet in self.jets.items():
            for pcfg in self.config.parts:
                sn = f"SN-{pcfg.name}-{part_counter}"
                part_counter += 1
                
                lot = self.py_rng.choice(lots)
                self.part_lots[sn] = lot
                self.part_health[sn] = 0.0
                self.part_states[sn] = 1 # Good
                jet["parts"][pcfg.name] = sn
                
    def _update_true_state(self, sn: str, pcfg: PartConfig, damage_delta: float):
        """Update cumulative hazard and discretize state."""
        # Scale modifier for the bad lot
        scale = pcfg.weibull_scale
        if self.part_lots[sn] == f"LOT-{self.config.bad_lot_idx}":
            scale *= self.config.bad_lot_scale_multiplier
            
        self.part_health[sn] += damage_delta
        
        # Simple thresholding for discrete states based on scale
        # In reality, health would be a draw from the hazard function.
        # For the SCM, we track deterministic damage and map to state.
        ratio = self.part_health[sn] / scale
        if ratio > 1.0:
            state = 4 # Failed
        elif ratio > 0.8:
            state = 3 # Severe
        elif ratio > 0.5:
            state = 2 # Degraded
        else:
            state = 1 # Good
            
        if state != self.part_states[sn]:
            self.part_states[sn] = state
            self.ground_truth.append(GroundTruth(
                part_sn=sn, t=self.current_t, true_health_state=state, lot_id=self.part_lots[sn]
            ))
            
    def step_day(self):
        """Simulate one day of operations."""
        for jet_id, jet in self.jets.items():
            # Missions
            for _ in range(self.config.sorties_per_day_per_jet):
                mission_type = self.py_rng.choice(["CAS", "CAP"])
                regime = self.py_rng.choice(["normal", "hot_dusty", "surge_high_g"])
                
                # Regime multiplies damage
                regime_mult = 1.0
                if regime == "hot_dusty": regime_mult = 1.5
                if regime == "surge_high_g": regime_mult = 2.0
                
                flight_hours = self.rng.uniform(1.5, 4.0)
                self.current_t += flight_hours # local time offset approx
                
                self.events.append(SimEvent(
                    t=self.current_t, event_type="sortie", jet_id=jet_id, 
                    details={"mission": mission_type, "regime": regime, "hours": flight_hours}
                ))
                
                # Wear out parts
                for pname, sn in jet["parts"].items():
                    pcfg = next(p for p in self.config.parts if p.name == pname)
                    # For Weibull, damage increments non-linearly with age, but we'll use a simple linear proxy for the simulation core
                    # Random parts have k=1 (constant hazard), wearout has k>1.
                    if pcfg.part_class == "wearout":
                        damage = flight_hours * regime_mult * (self.part_health[sn] + 1)**(pcfg.weibull_shape - 1)
                    else:
                        damage = flight_hours * regime_mult # constant
                    
                    self._update_true_state(sn, pcfg, damage)
                    
                    # Check failure (abort)
                    if self.part_states[sn] == 4:
                        self.events.append(SimEvent(
                            t=self.current_t, event_type="failure_abort", jet_id=jet_id,
                            part_name=pname, part_sn=sn
                        ))
                        # Trigger maintenance
                        self._maintenance_action(jet_id, pname, sn)
                        break # Abort rest of mission
                
    def _maintenance_action(self, jet_id: str, pname: str, sn: str):
        """Base maintenance, spares, cannibalisation."""
        # Check spares (simplified)
        has_spare = self.py_rng.random() < 0.8
        action = "replace_from_spares"
        
        if not has_spare:
            # Cannibalisation
            action = "cannibalise"
            if self.py_rng.random() < self.config.cannibalisation_damage_prob:
                # Induced damage
                self._update_true_state(sn, next(p for p in self.config.parts if p.name == pname), 500.0) # spike damage
        
        self.events.append(SimEvent(
            t=self.current_t, event_type="maintenance", jet_id=jet_id,
            part_name=pname, part_sn=sn, details={"action": action}
        ))
        
        # Reset health for replacement (simplified)
        if action == "replace_from_spares":
            self.part_health[sn] = 0.0
            self.part_states[sn] = 1

    def run(self) -> Tuple[List[SimEvent], List[GroundTruth]]:
        """Run the simulation loop."""
        for day in range(self.config.horizon_days):
            self.current_t = float(day * 24)
            self.step_day()
            
        return self.events, self.ground_truth

def run(config: SimulatorConfig, seed: int, world: str = "A") -> Tuple[List[dict], List[dict]]:
    """Entry point returning dictionaries for easy serialization."""
    sim = Simulator(config, seed, world)
    events, gt = sim.run()
    
    events_dicts = [{"t": e.t, "type": e.event_type, "jet": e.jet_id, "part": e.part_name, "sn": e.part_sn, "details": e.details} for e in events]
    gt_dicts = [{"sn": g.part_sn, "t": g.t, "true_state": g.true_health_state, "lot": g.lot_id} for g in gt]
    
    return events_dicts, gt_dicts
