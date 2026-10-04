"""
NFF Telemetry-to-Test Resolver

Uses an exact nearest-neighbour index over flight-data feature windows
to propose test envelopes for No Fault Found (NFF) parts.

When a part tests "NFF" on the bench, this module retrieves the most
similar historical flight-data windows where a fault WAS confirmed,
and proposes the test conditions that replicated those faults.

Status: [DEMO over real code]
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class TestEnvelope:
    """A proposed test condition for an NFF part."""
    condition_id: str
    temperature_c: float
    vibration_hz: float
    pressure_psi: float
    duration_min: float
    confidence: float  # how similar the retrieved case was
    source_case_id: str


@dataclass
class FlightWindow:
    """A feature vector extracted from a flight-data window."""
    case_id: str
    part_sn: str
    features: np.ndarray         # (D,) feature vector
    fault_confirmed: bool
    test_conditions: Optional[TestEnvelope] = None


class NFFResolver:
    """
    Exact nearest-neighbour index for NFF resolution.

    Architecture: We use brute-force L2 for the demo.
    In production this would be FAISS IndexFlatL2 for exact search.
    """

    def __init__(self):
        self.index: List[FlightWindow] = []

    def add(self, window: FlightWindow):
        """Add a flight-data window to the index."""
        self.index.append(window)

    def build_from_windows(self, windows: List[FlightWindow]):
        """Bulk-load windows."""
        self.index = list(windows)

    def query(self, query_features: np.ndarray, k: int = 3,
              fault_only: bool = True) -> List[Tuple[FlightWindow, float]]:
        """
        Find the k nearest neighbours to the query feature vector.

        Args:
            query_features: (D,) feature vector from the NFF part's last flight.
            k: number of neighbours.
            fault_only: if True, only return windows where fault was confirmed.

        Returns:
            List of (FlightWindow, distance) sorted by distance ascending.
        """
        candidates = self.index
        if fault_only:
            candidates = [w for w in candidates if w.fault_confirmed]

        if not candidates:
            return []

        # Brute-force L2
        dists = []
        for w in candidates:
            d = float(np.linalg.norm(query_features - w.features))
            dists.append((w, d))

        dists.sort(key=lambda x: x[1])
        return dists[:k]

    def propose_test(self, query_features: np.ndarray, k: int = 3
                     ) -> List[TestEnvelope]:
        """
        Propose test envelopes based on nearest confirmed-fault cases.
        """
        neighbours = self.query(query_features, k=k, fault_only=True)
        proposals = []

        for window, dist in neighbours:
            if window.test_conditions is not None:
                env = TestEnvelope(
                    condition_id=f"proposed_{window.case_id}",
                    temperature_c=window.test_conditions.temperature_c,
                    vibration_hz=window.test_conditions.vibration_hz,
                    pressure_psi=window.test_conditions.pressure_psi,
                    duration_min=window.test_conditions.duration_min,
                    confidence=1.0 / (1.0 + dist),
                    source_case_id=window.case_id,
                )
                proposals.append(env)

        return proposals
