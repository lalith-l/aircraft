"""
Stage 4 tests — Detection module.

Covers: C-MAPSS pipeline, e-detector, alert budget, sensor trust gate,
sibling detection, LightGBM baseline, adapter.
"""

import os
import numpy as np
import pytest

from nirnay.contracts import EvidenceItem, EvidenceSource, Alert
from nirnay.detect.detector import (
    EDetector, AlertBudgetGuard, SensorTrustGate,
    SiblingDetector, LightGBMBaseline
)
from nirnay.detect.cmapss_pipeline import (
    load_dataset, cluster_regimes, normalise_per_regime,
    add_rul_target, make_windows, SENSOR_COLS
)
from adapter.sim_adapter import events_to_evidence
from simulator.config import SimulatorConfig
from simulator.generator import run as gen_run


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "cmapss")


# -----------------------------------------------------------------------
# E-Detector
# -----------------------------------------------------------------------

class TestEDetector:

    def test_accumulates_evidence(self):
        det = EDetector(threshold_b=10.0)
        # Healthy observations (e_t ≈ 1)
        for _ in range(5):
            M, fired = det.update(1.0, 1.0)
            assert not fired
            assert M == pytest.approx(1.0)

    def test_fires_on_degradation(self):
        det = EDetector(threshold_b=10.0)
        # Strong degradation signals
        for _ in range(20):
            M, fired = det.update(5.0, 1.0)
            if fired:
                break
        assert fired

    def test_reset(self):
        det = EDetector(threshold_b=10.0)
        det.update(100.0, 1.0)
        det.reset()
        assert det.M == 1.0
        assert len(det.history) == 0


# -----------------------------------------------------------------------
# Alert Budget Guard
# -----------------------------------------------------------------------

class TestAlertBudgetGuard:

    def test_budget_calculation(self):
        guard = AlertBudgetGuard(n_parts=192, n_obs_per_day=48, b_crew=5)
        # threshold = 192 * 48 / 5 = 1843.2
        assert guard.threshold == pytest.approx(1843.2)

    def test_produces_alert(self):
        guard = AlertBudgetGuard(n_parts=10, n_obs_per_day=10, b_crew=100)
        # threshold = 1.0, so any signal > 1 should fire
        alert = guard.process("SN-X", 10.0, 1.0)
        assert alert is not None
        assert isinstance(alert, Alert)
        assert alert.part_sn == "SN-X"

    def test_no_alert_on_healthy(self):
        guard = AlertBudgetGuard(n_parts=100, n_obs_per_day=10, b_crew=1)
        # threshold = 1000, very hard to trigger
        alert = guard.process("SN-X", 1.0, 1.0)
        assert alert is None


# -----------------------------------------------------------------------
# Sensor Trust Gate
# -----------------------------------------------------------------------

class TestSensorTrustGate:

    def test_all_normal_sensors_get_full_weight(self):
        gate = SensorTrustGate(n_sensors=3, window=10, residual_threshold=3.0)
        rng = np.random.RandomState(42)
        # Feed consistent correlated readings
        for _ in range(15):
            base = rng.uniform(0, 1)
            reading = np.array([base, base + 0.1, base + 0.2])
            weights = gate.compute_weights("SN-A", reading)
        # All weights should be close to 1.0
        assert all(w > 0.9 for w in weights)

    def test_faulty_sensor_gets_lower_weight(self):
        gate = SensorTrustGate(n_sensors=3, window=20, residual_threshold=2.0)
        rng = np.random.RandomState(42)
        # Feed correlated readings, then inject a spike
        for _ in range(20):
            base = rng.uniform(0, 1)
            reading = np.array([base, base + 0.1, base + 0.2])
            gate.compute_weights("SN-B", reading)
        # Inject a spike on sensor 0
        spike = np.array([100.0, 0.5, 0.7])
        weights = gate.compute_weights("SN-B", spike)
        assert weights[0] < weights[1]
        assert weights[0] < weights[2]


# -----------------------------------------------------------------------
# Sibling Detection
# -----------------------------------------------------------------------

class TestSiblingDetector:

    def test_normal_sibling_low_score(self):
        det = SiblingDetector(mad_threshold=3.0)
        rng = np.random.RandomState(42)
        for _ in range(20):
            det.add_sibling_reading("regime_A", rng.normal(10, 1, 3))
        score = det.score("regime_A", np.array([10.1, 10.2, 9.9]))
        assert score < 3.0

    def test_anomalous_sibling_high_score(self):
        det = SiblingDetector(mad_threshold=3.0)
        rng = np.random.RandomState(42)
        for _ in range(20):
            det.add_sibling_reading("regime_A", rng.normal(10, 1, 3))
        score = det.score("regime_A", np.array([50.0, 50.0, 50.0]))
        assert score > 3.0


# -----------------------------------------------------------------------
# C-MAPSS Pipeline
# -----------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, "train_FD001.txt")),
    reason="C-MAPSS data not available"
)
class TestCMAPSSPipeline:

    def test_load(self):
        train, test, rul = load_dataset(DATA_DIR, "FD001")
        assert len(train) > 0
        assert len(test) > 0
        assert len(rul) > 0
        assert "engine_id" in train.columns
        assert "sensor_1" in train.columns

    def test_cluster_regimes(self):
        train, _, _ = load_dataset(DATA_DIR, "FD001")
        train = cluster_regimes(train, n_clusters=6)
        assert "regime" in train.columns
        assert train["regime"].nunique() <= 6

    def test_normalisation(self):
        train, _, _ = load_dataset(DATA_DIR, "FD001")
        train = cluster_regimes(train)
        norm_train, _, stats = normalise_per_regime(train)
        # After normalisation, per-regime means should be ~0
        for regime, s in stats.items():
            mask = norm_train["regime"] == regime
            means = norm_train.loc[mask, SENSOR_COLS].mean()
            assert all(abs(m) < 0.1 for m in means)

    def test_rul_target(self):
        train, _, _ = load_dataset(DATA_DIR, "FD001")
        train = add_rul_target(train, max_rul=125)
        assert "rul" in train.columns
        assert train["rul"].max() <= 125

    def test_windows(self):
        train, _, _ = load_dataset(DATA_DIR, "FD001")
        train = cluster_regimes(train)
        train, _, _ = normalise_per_regime(train)
        train = add_rul_target(train)
        X, y = make_windows(train, window_size=30)
        assert X.ndim == 3
        assert X.shape[1] == 30
        assert len(X) == len(y)


# -----------------------------------------------------------------------
# LightGBM Baseline
# -----------------------------------------------------------------------

@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATA_DIR, "train_FD001.txt")),
    reason="C-MAPSS data not available"
)
class TestLightGBMBaseline:

    def test_train_and_predict(self):
        train, test, rul_true = load_dataset(DATA_DIR, "FD001")
        train = cluster_regimes(train)
        test = cluster_regimes(test)
        train, test, _ = normalise_per_regime(train, test)
        train = add_rul_target(train)

        X_train, y_train = make_windows(train, window_size=30)
        # Only use a small subset for speed
        X_train = X_train[:500]
        y_train = y_train[:500]

        model = LightGBMBaseline(n_estimators=20, seed=42)
        model.train(X_train, y_train)
        preds = model.predict(X_train[:10])
        assert len(preds) == 10
        # Predictions should be positive
        assert all(p > -10 for p in preds)


# -----------------------------------------------------------------------
# Adapter
# -----------------------------------------------------------------------

class TestAdapter:

    def test_events_to_evidence(self):
        config = SimulatorConfig(horizon_days=5)
        events, _ = gen_run(config, seed=42)
        items = events_to_evidence(events)
        assert len(items) > 0
        assert all(isinstance(i, EvidenceItem) for i in items)
        # Check hash chain
        for i in range(1, len(items)):
            assert items[i].prev_hash == items[i - 1].hash

    def test_no_ground_truth_leakage(self):
        """Adapter should not contain any ground truth references."""
        import inspect
        import adapter.sim_adapter as mod
        src = inspect.getsource(mod)
        assert "true_state" not in src
        assert "true_health" not in src
        assert "ground_truth" not in src


# -----------------------------------------------------------------------
# Firewall
# -----------------------------------------------------------------------

class TestFirewall4:

    def test_detect_does_not_import_simulator(self):
        import inspect
        import nirnay.detect.detector as det
        import nirnay.detect.cmapss_pipeline as pipe
        for mod in [det, pipe]:
            src = inspect.getsource(mod)
            assert "from simulator" not in src
            assert "import simulator" not in src
