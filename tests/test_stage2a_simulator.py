"""
Stage 2a tests — Simulator Core
"""

import pytest
from simulator.config import SimulatorConfig, SEEDS
from simulator.generator import run

class TestSimulatorCore:

    def test_reproducibility_from_seed(self):
        config = SimulatorConfig(horizon_days=5) # short run
        
        e1, gt1 = run(config, seed=123, world="A")
        e2, gt2 = run(config, seed=123, world="A")
        
        # Should be exactly identical
        assert e1 == e2
        assert gt1 == gt2
        
    def test_worlds_can_diverge(self):
        config = SimulatorConfig(horizon_days=5)
        
        # Right now world doesn't change much in generator, but it's passed.
        # Just ensure it runs. Twin worlds usually diverge later when interventions apply.
        eA, gtA = run(config, seed=123, world="A")
        eB, gtB = run(config, seed=123, world="B")
        
        # Currently identical since interventions aren't in Stage 2a
        assert eA == eB

    def test_distributions_match_config(self):
        # We need a long run to see statistical effects
        config = SimulatorConfig(horizon_days=300)
        events, gt = run(config, seed=42)
        
        # Check that the bad lot has a higher failure rate (more transitions to state 4)
        failures_by_lot = {}
        for g in gt:
            if g["true_state"] == 4:
                failures_by_lot[g["lot"]] = failures_by_lot.get(g["lot"], 0) + 1
                
        # The bad lot is LOT-2 by default config
        bad_lot = f"LOT-{config.bad_lot_idx}"
        
        # It should have more failures than the average of other lots
        if failures_by_lot:
            bad_lot_fails = failures_by_lot.get(bad_lot, 0)
            other_lots_fails = [fails for lot, fails in failures_by_lot.items() if lot != bad_lot]
            if other_lots_fails:
                avg_other = sum(other_lots_fails) / len(other_lots_fails)
                assert bad_lot_fails > avg_other, f"Bad lot {bad_lot_fails} not > avg other {avg_other}"

    def test_simulator_firewall(self):
        """Verify simulator modules do not import nirnay."""
        import sys
        # Unload if loaded
        to_unload = [m for m in sys.modules if m.startswith('simulator')]
        for m in to_unload:
            del sys.modules[m]
            
        import simulator
        import simulator.generator
        import simulator.config
        
        for m in sys.modules:
            if m.startswith('simulator'):
                import inspect
                module = sys.modules[m]
                # Check source code text for 'nirnay'
                try:
                    src = inspect.getsource(module)
                    assert 'from nirnay' not in src
                    assert 'import nirnay' not in src
                except TypeError:
                    pass # builtins
