"""
Stage 5 tests — Laya Evidence Compiler.

Tests model verification, calibration status gating, and the cascade logic.
"""

import numpy as np
from nirnay.laya.compiler import LayaClient, EvidenceCompiler, QUESTIONS

class TestLayaClient:
    def test_verify_model(self):
        client = LayaClient()
        info = client.verify_model()
        assert info["verified"] is True
        assert info["model"] == "Laya"
        
    def test_calibration_gate(self):
        client = LayaClient()
        assert client.status == "FALLBACK"
        
        # Insufficient data
        passed = client.calibrate([{"text": "a"} for _ in range(10)])
        assert not passed
        assert client.status == "FALLBACK"
        
        # Sufficient data
        passed = client.calibrate([{"text": "a"} for _ in range(100)])
        assert passed
        assert client.status == "FINE-TUNED"
        assert client.calibration_error < 0.1
        
    def test_ask_abstains_in_fallback(self):
        client = LayaClient()
        ans = client.ask("intermittent fault observed", QUESTIONS[0])
        assert ans is None # Abstains because status is FALLBACK


class TestEvidenceCompiler:
    def test_rules_baseline(self):
        compiler = EvidenceCompiler(LayaClient())
        # Rules should catch simple NFF
        ans = compiler._rules_baseline("pilot reported issue, nff on bench", "q_nff")
        assert ans is True
        
        # Rules should abstain on complex logs like late entries
        ans = compiler._rules_baseline("[LATE ENTRY] nff but maybe not", "q_nff")
        assert ans is None
        
    def test_cascade_process_text(self):
        client = LayaClient()
        client.status = "FINE-TUNED" # Bypass calibration for test
        compiler = EvidenceCompiler(client)
        
        text = "severe vibration reported during flight"
        likelihood = compiler.process_text(text)
        
        # Question 'q_vibration' triggers.
        # Its p(yes|h) is [0.05, 0.05, 0.20, 0.60, 0.80]
        # State 3 and 4 should be much higher than 0 and 1.
        assert likelihood[4] > likelihood[0]
        assert likelihood[3] > likelihood[1]
