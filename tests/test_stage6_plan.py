"""
Stage 6 tests — Planner and Promise.

Tests mission-aware solver, Whittle allocator, shadow prices, 
and conformal promise calibration.
"""

import numpy as np
import pytest
from nirnay.plan.solver import MissionSolver, ShadowPricer, WhittleAllocator, ConformalPromise

class TestMissionSolver:
    def test_solver_produces_frontier(self):
        solver = MissionSolver(n_aircraft=5)
        # 3 fully healthy, 1 marginal, 1 broken
        health = np.array([1.0, 1.0, 1.0, 0.6, 0.1])
        
        frontier = solver.compute_frontier(health, p_threshold=0.5)
        
        assert len(frontier) > 0
        # Given we have 4 aircraft with health >= 0.5, the max CAP + CAS should be 4
        max_total = max(f["n_cas"] + f["n_cap"] for f in frontier)
        assert max_total == 4
        
        # Check Pareto structure: as CAP increases, CAS should decrease
        caps = [f["n_cap"] for f in frontier]
        assert caps == sorted(caps)


class TestWhittleAllocator:
    def test_compute_indices(self):
        # 3 parts, 5 states
        beliefs = np.array([
            [0.1, 0.1, 0.1, 0.2, 0.5], # Highly failed
            [0.8, 0.1, 0.1, 0.0, 0.0], # Very healthy
            [0.1, 0.1, 0.1, 0.6, 0.1], # Severe
        ])
        degradations = np.array([0.1, 0.05, 0.5])
        
        indices = WhittleAllocator.compute_indices(beliefs, degradations)
        
        assert len(indices) == 3
        assert indices[0] > indices[1] # Failed > Healthy
        assert indices[2] > indices[1] # Severe > Healthy
        assert indices[0] == pytest.approx(0.5 + 0.2 * 0.1)
        assert indices[2] == pytest.approx(0.1 + 0.6 * 0.5)


class TestShadowPricer:
    def test_shadow_prices(self):
        solver = MissionSolver(n_aircraft=10, bay_capacity=2, tech_capacity=2)
        # Assuming capacities don't strictly constrain the simplified model right now,
        # but the API contract expects it to return a sorted list.
        health = np.ones(10)
        
        prices = ShadowPricer.compute_prices(solver, health)
        
        assert len(prices) == 2
        assert prices[0]["price"] >= prices[1]["price"]
        assert "resource" in prices[0]


class TestConformalPromise:
    def test_calibrate_promise(self):
        predictions = np.array([10, 9, 8, 10, 9])
        # Actuals are strictly worse, meaning we over-promised
        actuals = np.array([8, 8, 7, 7, 7])
        
        lam = ConformalPromise.calibrate(predictions, actuals, target_alpha=0.1)
        
        # lambda must be strictly positive to shift predictions down
        assert lam > 0.0
        
        # Expected over-promise with new lambda
        over_promises = np.maximum(0, (predictions - lam) - actuals)
        risk = np.mean(over_promises)
        assert risk <= 0.1 # Should satisfy the target (modulo finite sample bound)
