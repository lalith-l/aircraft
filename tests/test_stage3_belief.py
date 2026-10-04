"""
Stage 3 tests — Belief Engine.

Tests Forward-Backward smoothing, EM log-likelihood monotonicity,
and parameter recovery on simulated data.
"""

import math
import numpy as np
import pytest

from nirnay.contracts import EvidenceItem, EvidenceSource
from nirnay.belief.engine import BeliefEngine

def make_item(likelihood: list) -> EvidenceItem:
    return EvidenceItem(
        source=EvidenceSource.SENSOR,
        t=100.0, t_logged=100.0,
        likelihood={"per_state": likelihood}
    )

class TestBeliefEngine:
    
    def test_forward_backward_improves_confidence(self):
        engine = BeliefEngine(n_states=5)
        # Set a permissive transition matrix so rapid degradation isn't heavily penalized
        engine.A = np.array([
            [0.5, 0.2, 0.1, 0.1, 0.1],
            [0.0, 0.5, 0.2, 0.2, 0.1],
            [0.0, 0.0, 0.5, 0.3, 0.2],
            [0.0, 0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        # Sequence: uncertain start, then clear failure at the end
        # State 0=NEW, 1=GOOD, 2=DEGRADED, 3=SEVERE, 4=FAILED
        seq = [
            make_item([0.2, 0.6, 0.2, 0.0, 0.0]), # Mostly good
            make_item([0.1, 0.3, 0.4, 0.2, 0.0]), # Degrading
            make_item([0.0, 0.0, 0.1, 0.8, 0.1]), # Severe
            make_item([0.0, 0.0, 0.0, 0.1, 0.9])  # Failed
        ]
        
        gamma, ll = engine.forward_backward(seq)
        
        # Test shape
        assert gamma.shape == (4, 5)
        
        # At t=3, it should be very confident in state 4
        assert gamma[3, 4] > 0.8
        
        # Smoothing should push confidence about t=2 being state 3 
        # (since it transitioned to 4 at t=3)
        assert gamma[2, 3] > 0.5
        
    def test_em_log_likelihood_non_decreasing(self):
        engine = BeliefEngine(n_states=5) 
        engine.pi = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        engine.A = np.array([
            [0.8, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.7, 0.3, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        
        # Create some noisy sequences
        seqs = []
        for _ in range(10):
            seq = [
                make_item([0.9, 0.1, 0.0, 0.0, 0.0]),
                make_item([0.7, 0.3, 0.0, 0.0, 0.0]),
                make_item([0.2, 0.8, 0.0, 0.0, 0.0]),
                make_item([0.0, 0.2, 0.8, 0.0, 0.0])
            ]
            seqs.append(seq)
            
        lls = engine.learn_em(seqs, max_iter=10)
        
        # Monotonicity check (allowing for tiny float errors)
        for i in range(1, len(lls)):
            assert lls[i] >= lls[i-1] - 1e-9
            
    def test_em_parameter_recovery_on_simulated_data(self):
        # We simulate from a known transition matrix and check if EM recovers it
        true_A = np.array([
            [0.90, 0.10, 0.00, 0.0, 0.0],
            [0.00, 0.80, 0.20, 0.0, 0.0],
            [0.00, 0.00, 1.00, 0.0, 0.0],
            [0.00, 0.00, 0.00, 1.0, 0.0],
            [0.00, 0.00, 0.00, 0.0, 1.0]
        ])
        
        rng = np.random.RandomState(42)
        n_seqs = 200
        seq_len = 10
        
        seqs = []
        for _ in range(n_seqs):
            state = 0
            seq = []
            for t in range(seq_len):
                # Generate noisy emission: heavily peaked at true state
                emission = np.zeros(5)
                emission[state] = 0.8
                # distribute rest randomly
                remaining = 0.2
                for i in range(5):
                    if i != state:
                        r = rng.uniform(0, remaining)
                        emission[i] = r
                        remaining -= r
                emission[2 if state != 2 else 0] += remaining # mop up
                
                seq.append(make_item(emission.tolist()))
                
                # transition
                state = rng.choice(5, p=true_A[state])
                
            seqs.append(seq)
            
        engine = BeliefEngine(n_states=5)
        # Initialize with wrong A
        engine.A = np.array([
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0]
        ])
        
        engine.learn_em(seqs, max_iter=30)
        
        # Should be closer to true_A than the initial 0.5 guess
        assert abs(engine.A[0, 0] - 0.90) < 0.05
        assert abs(engine.A[1, 1] - 0.80) < 0.05
        # Structural zeros should remain zero
        assert engine.A[1, 0] == 0.0
        assert engine.A[2, 0] == 0.0
        assert engine.A[2, 1] == 0.0
