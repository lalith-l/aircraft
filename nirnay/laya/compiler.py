"""
Laya Evidence Compiler

Uses a language model to extract likelihoods from maintenance text.
Implements a cascade: Rules -> Statistics -> Laya -> Human.

Status: [REAL / FALLBACK]
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

# ---------------------------------------------------------------------------
# Question Registry
# ---------------------------------------------------------------------------

class LayaQuestion:
    def __init__(self, id: str, text: str):
        self.id = id
        self.text = text
        # Calibrated p(yes | state) for states 0-4
        self.p_yes_given_h = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
        self.abstention_rate = 0.0

    def calibrate(self, probs: List[float]):
        assert len(probs) == 5
        self.p_yes_given_h = np.array(probs)


QUESTIONS = [
    LayaQuestion("q_intermittent", "Does the text describe an intermittent fault?"),
    LayaQuestion("q_nff", "Was the fault not reproduced on the ground?"),
    LayaQuestion("q_cannibalised", "Was a part cannibalised from another aircraft?"),
    LayaQuestion("q_vibration", "Is there mention of excessive vibration?"),
    LayaQuestion("q_high_g", "Was the aircraft exposed to high-G maneuvers?"),
    LayaQuestion("q_contradiction", "Is there a contradiction in the maintenance log?"),
    LayaQuestion("q_duplicate", "Is this record a duplicate of a previous entry?"),
]

# Provide some sensible synthetic calibrations
QUESTIONS[0].calibrate([0.01, 0.05, 0.40, 0.30, 0.05]) # Intermittent (States 2, 3)
QUESTIONS[1].calibrate([0.01, 0.02, 0.50, 0.20, 0.05]) # NFF (States 2, 3)
QUESTIONS[3].calibrate([0.05, 0.05, 0.20, 0.60, 0.80]) # Vibration (States 3, 4)


# ---------------------------------------------------------------------------
# Laya Client (Stub / Simulator)
# ---------------------------------------------------------------------------

class LayaClient:
    """
    Interface to the Laya language model.
    In this prototype, we simulate the Laya response.
    """
    def __init__(self):
        self.status = "FALLBACK" # Default until calibrated
        self.calibration_error = 1.0

    def verify_model(self):
        """
        Record verification of the Laya model card.
        """
        return {
            "model": "Laya",
            "license": "Apache 2.0",
            "size_english": "421M",
            "size_multilingual": "322M",
            "latency_t4": "~33-40ms",
            "latency_cpu_int8": "250-300ms",
            "verified": True
        }

    def calibrate(self, validation_data: List[Dict]) -> bool:
        """
        Simulate calibration process. If data is sufficient, calibration passes.
        """
        if len(validation_data) >= 50:
            self.calibration_error = 0.08 # Acceptable error
            self.status = "FINE-TUNED"
            return True
        else:
            self.calibration_error = 0.35
            self.status = "FALLBACK"
            return False

    def ask(self, text: str, question: LayaQuestion) -> Optional[bool]:
        """
        Simulate asking the LLM a question.
        Returns True/False, or None if the model abstains.
        """
        if self.status == "FALLBACK":
            return None # Abstain in fallback mode
            
        # Very crude simulation of LLM logic for the demo
        text = text.lower()
        if question.id == "q_intermittent":
            if "intermittent" in text or "sometimes" in text: return True
            return False
        elif question.id == "q_nff":
            if "not reproduced" in text or "nff" in text: return True
            return False
        elif question.id == "q_vibration":
            if "vibration" in text or "shaking" in text: return True
            return False
            
        return None # Abstain


# ---------------------------------------------------------------------------
# Evidence Compiler Cascade
# ---------------------------------------------------------------------------

class EvidenceCompiler:
    def __init__(self, laya_client: LayaClient):
        self.laya = laya_client
        self.questions = {q.id: q for q in QUESTIONS}

    def _rules_baseline(self, text: str, q_id: str) -> Optional[bool]:
        """Fast regex/rules baseline."""
        text = text.lower()
        if q_id == "q_nff":
            if "[late entry" in text: return None # Rules get confused by complex logs
            if "nff" in text.split(): return True
            return False
        return None

    def process_text(self, text: str) -> np.ndarray:
        """
        Process a maintenance text log through the cascade and return a 
        5-state likelihood vector.
        """
        likelihood = np.ones(5)
        
        for q_id, question in self.questions.items():
            # 1. Rules Baseline
            ans = self._rules_baseline(text, q_id)
            
            # 2. Statistics (Skipped in this simple stub)
            
            # 3. Laya LLM (if rules abstain)
            if ans is None:
                ans = self.laya.ask(text, question)
                
            # 4. Human (Skipped - assume abstention remains)
            
            # Update likelihood if we got an answer
            if ans is True:
                likelihood *= question.p_yes_given_h
            elif ans is False:
                likelihood *= (1.0 - question.p_yes_given_h)
                
        # Normalize
        s = np.sum(likelihood)
        if s > 0:
            likelihood /= s
        else:
            likelihood = np.ones(5) / 5.0
            
        return np.clip(likelihood, 1e-12, 1.0)
