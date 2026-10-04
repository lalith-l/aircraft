"""
Detection Module — E-detector alert guard, sensor-trust gate, sibling detection,
and baseline models.

Status: [REAL]
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from nirnay.contracts import EvidenceItem, EvidenceSource, Alert


# ===========================================================================
# E-Detector Alert Guard
# ===========================================================================

class EDetector:
    """
    Sequential e-detector for change detection.

    Maintains a running e-value statistic M_t:
        e_t = P(obs | degraded) / P(obs | healthy)
        M_t = max(1, M_{t-1}) * e_t

    Alerts when M_t >= b (the budget-derived threshold).
    """

    def __init__(self, threshold_b: float = 10.0):
        self.threshold_b = threshold_b
        self.M = 1.0
        self.history: List[float] = []

    def update(self, likelihood_degraded: float,
               likelihood_healthy: float) -> Tuple[float, bool]:
        """
        Process one observation.
        Returns (M_t, triggered).
        """
        if likelihood_healthy < 1e-12:
            likelihood_healthy = 1e-12
        e_t = likelihood_degraded / likelihood_healthy
        self.M = max(1.0, self.M) * e_t
        self.history.append(self.M)
        return self.M, self.M >= self.threshold_b

    def reset(self):
        self.M = 1.0
        self.history.clear()


class AlertBudgetGuard:
    """
    Budgeted alert system.

    Budget b = N_parts * n_obs_per_day / B_crew

    Only fire alerts when the e-detector statistic exceeds the budget-derived
    threshold. This controls false-alarm rate while maintaining detection power.
    """

    def __init__(self, n_parts: int, n_obs_per_day: int, b_crew: int):
        self.n_parts = n_parts
        self.n_obs_per_day = n_obs_per_day
        self.b_crew = b_crew
        self.threshold = (n_parts * n_obs_per_day) / b_crew
        self.detectors: Dict[str, EDetector] = {}

    def get_detector(self, part_sn: str) -> EDetector:
        if part_sn not in self.detectors:
            self.detectors[part_sn] = EDetector(threshold_b=self.threshold)
        return self.detectors[part_sn]

    def process(self, part_sn: str, likelihood_degraded: float,
                likelihood_healthy: float) -> Optional[Alert]:
        det = self.get_detector(part_sn)
        M, triggered = det.update(likelihood_degraded, likelihood_healthy)

        if triggered:
            alert = Alert(
                part_sn=part_sn,
                M=M,
                threshold_b=self.threshold,
                tier=self._tier(M),
                budget_left=max(0, self.threshold - M),
            )
            det.reset()
            return alert
        return None

    @staticmethod
    def _tier(M: float) -> int:
        if M >= 100:
            return 3
        if M >= 50:
            return 2
        return 1


# ===========================================================================
# Sensor Trust Gate
# ===========================================================================

class SensorTrustGate:
    """
    Predict each sensor from its neighbours.
    If the residual is large, lower the sensor's trust weight.

    Uses a simple rolling leave-one-out regression to detect
    faulty/drifting sensors without needing labelled data.
    """

    def __init__(self, n_sensors: int, window: int = 20,
                 residual_threshold: float = 3.0):
        self.n_sensors = n_sensors
        self.window = window
        self.residual_threshold = residual_threshold
        self._buffers: Dict[str, List[np.ndarray]] = {}

    def _add_reading(self, part_sn: str, readings: np.ndarray):
        """Store a reading vector (1 x n_sensors) into the sliding buffer."""
        if part_sn not in self._buffers:
            self._buffers[part_sn] = []
        self._buffers[part_sn].append(readings.copy())
        if len(self._buffers[part_sn]) > self.window:
            self._buffers[part_sn].pop(0)

    def compute_weights(self, part_sn: str, readings: np.ndarray) -> np.ndarray:
        """
        Compute trust weights for each sensor based on leave-one-out prediction.
        Returns array of weights in [0, 1], one per sensor.
        """
        self._add_reading(part_sn, readings)
        buf = self._buffers.get(part_sn, [])

        if len(buf) < 5:
            return np.ones(self.n_sensors)

        data = np.array(buf)  # (window, n_sensors)
        weights = np.ones(self.n_sensors)

        for i in range(self.n_sensors):
            # Predict sensor i from the others using their mean correlation
            others = np.delete(data, i, axis=1)
            mean_others = others.mean(axis=1)

            if np.std(mean_others) < 1e-6:
                continue

            # Linear regression of sensor i on mean of others
            x = mean_others
            y = data[:, i]
            slope = np.cov(x, y)[0, 1] / (np.var(x) + 1e-12)
            intercept = np.mean(y) - slope * np.mean(x)
            predicted = slope * x[-1] + intercept
            residual = abs(y[-1] - predicted)
            normalised_residual = residual / (np.std(y) + 1e-12)

            if normalised_residual > self.residual_threshold:
                weights[i] = max(0.1, 1.0 - (normalised_residual -
                                               self.residual_threshold) * 0.2)

        return np.clip(weights, 0.1, 1.0)


# ===========================================================================
# Sibling Detection
# ===========================================================================

class SiblingDetector:
    """
    Compare a part's sensor readings against its siblings (same part type,
    same operating regime) to detect anomalies without labels.

    Uses the residual from the sibling median and MAD.
    """

    def __init__(self, mad_threshold: float = 3.0):
        self.mad_threshold = mad_threshold
        self._regime_readings: Dict[str, List[np.ndarray]] = {}

    def add_sibling_reading(self, regime_key: str, reading: np.ndarray):
        """Add a reading from a sibling in the same regime."""
        if regime_key not in self._regime_readings:
            self._regime_readings[regime_key] = []
        self._regime_readings[regime_key].append(reading.copy())

    def score(self, regime_key: str, reading: np.ndarray) -> float:
        """
        Score a reading against its siblings.
        Returns an anomaly score (higher = more anomalous).
        """
        siblings = self._regime_readings.get(regime_key, [])
        if len(siblings) < 3:
            return 0.0

        data = np.array(siblings)
        median = np.median(data, axis=0)
        mad = np.median(np.abs(data - median), axis=0) + 1e-12
        deviation = np.abs(reading - median) / mad
        return float(np.max(deviation))


# ===========================================================================
# Baseline Models (LightGBM + 1D-CNN stubs)
# ===========================================================================

class LightGBMBaseline:
    """
    LightGBM windowed-feature baseline for RUL prediction.
    Status: [REAL]
    """

    def __init__(self, n_estimators: int = 100, seed: int = 42):
        self.n_estimators = n_estimators
        self.seed = seed
        self.model = None

    def train(self, X_windows: np.ndarray, y_rul: np.ndarray):
        """
        Train on windowed features.
        X_windows: (n_samples, window_size, n_features)
        y_rul: (n_samples,)
        """
        import lightgbm as lgb

        # Flatten windows to (n_samples, window_size * n_features)
        X_flat = X_windows.reshape(X_windows.shape[0], -1)

        self.model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            verbosity=-1,
        )
        self.model.fit(X_flat, y_rul)

    def predict(self, X_windows: np.ndarray) -> np.ndarray:
        X_flat = X_windows.reshape(X_windows.shape[0], -1)
        return self.model.predict(X_flat)


class CNN1DBaseline:
    """
    1D-CNN baseline for RUL prediction.
    Status: [PLANNED] — requires PyTorch; stub for now.
    """

    def __init__(self):
        self.status = "PLANNED"

    def train(self, X_windows: np.ndarray, y_rul: np.ndarray):
        raise NotImplementedError(
            "1D-CNN baseline requires PyTorch. Status: [PLANNED]."
        )

    def predict(self, X_windows: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "1D-CNN baseline requires PyTorch. Status: [PLANNED]."
        )
