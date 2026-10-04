"""
Lot Contagion Detector

Flags manufacturing lots with elevated failure rates using a
Gamma-Poisson posterior. Triggers when P(λ_lot > κ·λ_0) >= 0.9.

Status: [DEMO over real code]
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from scipy import stats as sp_stats


@dataclass
class LotStatus:
    """Status report for a manufacturing lot."""
    lot_id: str
    n_parts: int
    n_failures: int
    exposure_hours: float
    posterior_rate: float
    baseline_rate: float
    rate_ratio: float
    p_elevated: float
    flagged: bool


class LotContagionDetector:
    """
    Monitors manufacturing lots for elevated failure rates.

    Uses a Gamma-Poisson conjugate model with a conservative prior.
    Flags a lot when P(λ_lot > κ·λ_baseline) >= flag_threshold.
    """

    def __init__(self, baseline_rate: float = 0.001,
                 kappa: float = 2.0,
                 flag_threshold: float = 0.9,
                 prior_alpha: float = 1.0,
                 prior_beta: float = 1000.0):
        """
        Args:
            baseline_rate: fleet-wide average failure rate per hour.
            kappa: multiplier — flag if lot rate exceeds kappa * baseline.
            flag_threshold: probability threshold for flagging.
            prior_alpha, prior_beta: Gamma prior (shrinkage toward baseline).
        """
        self.baseline_rate = baseline_rate
        self.kappa = kappa
        self.flag_threshold = flag_threshold
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta

        # lot_id -> {n_parts, n_failures, exposure_hours}
        self.lots: Dict[str, Dict] = {}

    def register_lot(self, lot_id: str, n_parts: int):
        """Register a new manufacturing lot."""
        self.lots[lot_id] = {
            "n_parts": n_parts,
            "n_failures": 0,
            "exposure_hours": 0.0,
        }

    def record_exposure(self, lot_id: str, hours: float):
        """Record additional operating hours for a lot."""
        if lot_id in self.lots:
            self.lots[lot_id]["exposure_hours"] += hours

    def record_failure(self, lot_id: str):
        """Record a failure event for a part from this lot."""
        if lot_id in self.lots:
            self.lots[lot_id]["n_failures"] += 1

    def evaluate_lot(self, lot_id: str) -> Optional[LotStatus]:
        """Evaluate a single lot's posterior failure rate."""
        if lot_id not in self.lots:
            return None

        info = self.lots[lot_id]
        n_fail = info["n_failures"]
        exposure = info["exposure_hours"]

        alpha_post = self.prior_alpha + n_fail
        beta_post = self.prior_beta + exposure

        posterior_rate = alpha_post / beta_post
        rate_ratio = posterior_rate / self.baseline_rate if self.baseline_rate > 0 else 0.0

        # P(λ_lot > κ·λ_baseline)
        threshold_rate = self.kappa * self.baseline_rate
        p_elevated = float(1.0 - sp_stats.gamma.cdf(
            threshold_rate, a=alpha_post, scale=1.0 / beta_post
        ))

        return LotStatus(
            lot_id=lot_id,
            n_parts=info["n_parts"],
            n_failures=n_fail,
            exposure_hours=exposure,
            posterior_rate=posterior_rate,
            baseline_rate=self.baseline_rate,
            rate_ratio=rate_ratio,
            p_elevated=p_elevated,
            flagged=(p_elevated >= self.flag_threshold),
        )

    def evaluate_all(self) -> List[LotStatus]:
        """Evaluate all lots. Returns list sorted by P(elevated) descending."""
        results = []
        for lot_id in self.lots:
            status = self.evaluate_lot(lot_id)
            if status is not None:
                results.append(status)
        results.sort(key=lambda s: s.p_elevated, reverse=True)
        return results

    def flagged_lots(self) -> List[LotStatus]:
        """Return only lots that are flagged."""
        return [s for s in self.evaluate_all() if s.flagged]
