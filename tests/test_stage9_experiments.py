"""
Stage 9 tests — Twin-World Orchestrator and Experimental Matrix
"""

import pytest
from nirnay.experiments import TwinWorldOrchestrator, run_experiments

class TestTwinWorldOrchestrator:
    def test_status_quo_execution(self):
        orchestrator = TwinWorldOrchestrator(seed=42)
        res = orchestrator.run_status_quo()
        
        assert res.world_id == "Status Quo (Reactive)"
        assert len(res.readiness_history) == 30
        assert res.n_missions_flown > 0
        assert res.computation_time_s >= 0.0

    def test_nirnay_ai_execution(self):
        orchestrator = TwinWorldOrchestrator(seed=42)
        res = orchestrator.run_nirnay_ai()
        
        assert res.world_id == "Nirnay AI (Proactive)"
        assert len(res.readiness_history) == 30
        assert res.n_missions_flown > 0
        assert res.computation_time_s >= 0.0

    def test_ai_outperforms_status_quo(self):
        orchestrator = TwinWorldOrchestrator(seed=123)
        res_sq = orchestrator.run_status_quo()
        res_ai = orchestrator.run_nirnay_ai()
        
        # AI should have fewer aborts or higher readiness
        avg_readiness_sq = sum(res_sq.readiness_history) / len(res_sq.readiness_history)
        avg_readiness_ai = sum(res_ai.readiness_history) / len(res_ai.readiness_history)
        
        assert avg_readiness_ai >= avg_readiness_sq
        assert res_ai.n_aborts < res_sq.n_aborts

class TestExperiments:
    def test_run_experiments(self):
        results = run_experiments()
        
        # Check that all experiments A-K are present
        assert "A: Status Quo (No AI)" in results
        assert "B/C: Nirnay AI (Belief + Planner)" in results
        assert "D: NFF Resolver" in results
        assert "K: Laya Cascades" in results
        
        # Validate data types for A
        assert "readiness" in results["A: Status Quo (No AI)"]
        assert isinstance(results["A: Status Quo (No AI)"]["aborts"], int)
