"""
Stage 2b tests — Observation layers.

Covers: sensors (with fault injection), maintenance records (with corruption),
teardown grades (confusion matrix accuracy + NFF), spoofed alerts, confounding
model, data partitioning, twin-world divergence, second generator family,
and the simulator firewall.
"""

import numpy as np
import random
import pytest
from collections import Counter

from simulator.config import SimulatorConfig
from simulator.generator import run as gen_run, Simulator
from simulator.observations import (
    SensorModel, SensorReading,
    RecordGenerator, MaintenanceRecord,
    TeardownModel, TeardownResult,
    AlertSpoofer, SpoofedAlert,
    ConfoundingModel,
    partition_data,
    run_twin_worlds,
    AltGeneratorFamily,
)


# -----------------------------------------------------------------------
# Sensor model
# -----------------------------------------------------------------------

class TestSensorModel:

    def test_reading_has_correct_structure(self):
        rng = np.random.RandomState(0)
        sm = SensorModel(rng)
        r = sm.read(t=100.0, part_sn="SN-1", true_state=1, channel="vibration")
        assert isinstance(r, SensorReading)
        assert r.t == 100.0
        assert r.part_sn == "SN-1"
        assert r.channel == "vibration"
        assert isinstance(r.value, float)

    def test_readings_differ_by_state(self):
        rng = np.random.RandomState(42)
        sm = SensorModel(rng)
        vals = {}
        for state in [1, 2, 3, 4]:
            readings = [sm.read(100.0, "SN-X", state, "vibration").value
                        for _ in range(200)]
            vals[state] = np.mean(readings)
        # Mean vibration should increase with state
        assert vals[1] < vals[2] < vals[3] < vals[4]

    def test_stuck_fault(self):
        rng = np.random.RandomState(7)
        sm = SensorModel(rng)
        sm.inject_fault("sensor_1", "stuck")
        readings = [sm.read(float(t), "SN-X", 2, "vibration", "sensor_1")
                    for t in range(50)]
        values = {r.value for r in readings}
        assert len(values) == 1  # all identical (stuck)
        assert all(r.fault_mode == "stuck" for r in readings)

    def test_bias_fault(self):
        rng = np.random.RandomState(8)
        sm = SensorModel(rng)
        sm.inject_fault("sensor_2", "bias")
        r = sm.read(100.0, "SN-X", 1, "vibration", "sensor_2")
        assert r.fault_mode == "bias"

    def test_drift_fault_increases_over_time(self):
        rng = np.random.RandomState(9)
        sm = SensorModel(rng)
        sm.inject_fault("sensor_3", "drift")
        early = np.mean([sm.read(10.0, "SN-X", 1, "vibration", "sensor_3").value
                         for _ in range(50)])
        # Reset rng to get comparable noise
        sm2 = SensorModel(np.random.RandomState(9))
        sm2.inject_fault("sensor_3", "drift")
        late = np.mean([sm2.read(10000.0, "SN-X", 1, "vibration", "sensor_3").value
                        for _ in range(50)])
        assert late > early  # drift pushes values up over time

    def test_all_channels(self):
        rng = np.random.RandomState(0)
        sm = SensorModel(rng)
        readings = sm.read_all_channels(100.0, "SN-X", 2, "pfx")
        assert len(readings) == 3
        channels = {r.channel for r in readings}
        assert channels == {"vibration", "temperature", "pressure"}


# -----------------------------------------------------------------------
# Maintenance records
# -----------------------------------------------------------------------

class TestRecordGenerator:

    def test_generates_record(self):
        rng = random.Random(0)
        gen = RecordGenerator(rng, corruption_rate=0.0)
        rec = gen.generate(100.0, "SN-1", "engine_starter", "J-01", "Base-0")
        assert isinstance(rec, MaintenanceRecord)
        assert "SN-1" in rec.text
        assert rec.is_corrupted is False

    def test_corruption_rate(self):
        rng = random.Random(42)
        gen = RecordGenerator(rng, corruption_rate=0.50)
        records = [gen.generate(float(i), f"SN-{i}", "pump", "J-01", "Base-0")
                   for i in range(200)]
        corrupted = sum(1 for r in records if r.is_corrupted)
        # With 50% rate, we expect roughly 100 ± 30
        assert 40 < corrupted < 160

    def test_wrong_sn_corruption(self):
        rng = random.Random(99)
        gen = RecordGenerator(rng, corruption_rate=1.0)  # always corrupt
        # Try many to find a wrong_sn corruption
        for i in range(100):
            rec = gen.generate(float(i), "SN-target", "pump", "J-01", "Base-0")
            if rec.corruption_type == "wrong_sn":
                assert "SN-target" not in rec.text
                break
        else:
            pytest.fail("No wrong_sn corruption found in 100 attempts")

    def test_copy_paste_corruption(self):
        rng = random.Random(77)
        gen = RecordGenerator(rng, corruption_rate=1.0)
        for i in range(100):
            rec = gen.generate(float(i), "SN-X", "pump", "J-01", "Base-0")
            if rec.corruption_type == "copy_paste":
                assert " // " in rec.text
                break
        else:
            pytest.fail("No copy_paste corruption found in 100 attempts")


# -----------------------------------------------------------------------
# Teardown grading
# -----------------------------------------------------------------------

class TestTeardownModel:

    def test_confusion_matrix_accuracy(self):
        """Observed grades should statistically match the confusion matrix."""
        config = SimulatorConfig()
        rng = np.random.RandomState(42)
        model = TeardownModel(config.teardown_confusion, rng, nff_prob_good=0.0,
                              nff_prob_degraded=0.0)  # disable NFF for clean CM test

        N = 2000
        for true_state in [1, 2, 3, 4]:
            grades = [model.grade(100.0, "SN-X", true_state).observed_grade
                      for _ in range(N)]
            counts = Counter(grades)
            for g in [1, 2, 3, 4]:
                expected = config.teardown_confusion[true_state - 1][g - 1]
                observed_frac = counts.get(g, 0) / N
                assert abs(observed_frac - expected) < 0.05, \
                    f"State {true_state}, grade {g}: expected {expected:.2f}, got {observed_frac:.2f}"

    def test_nff_occurs_for_good_parts(self):
        config = SimulatorConfig()
        rng = np.random.RandomState(42)
        model = TeardownModel(config.teardown_confusion, rng,
                              nff_prob_good=0.30, nff_prob_degraded=0.10)
        N = 1000
        results = [model.grade(100.0, "SN-X", 1) for _ in range(N)]
        nff_count = sum(1 for r in results if r.is_nff)
        # Expect ~30% NFF for state 1
        assert 200 < nff_count < 400, f"NFF count {nff_count} outside expected range"

    def test_nff_does_not_occur_for_failed_parts(self):
        config = SimulatorConfig()
        rng = np.random.RandomState(42)
        model = TeardownModel(config.teardown_confusion, rng)
        results = [model.grade(100.0, "SN-X", 4) for _ in range(500)]
        nff_count = sum(1 for r in results if r.is_nff)
        assert nff_count == 0


# -----------------------------------------------------------------------
# Spoofed alerts
# -----------------------------------------------------------------------

class TestAlertSpoofer:

    def test_phantom_rate(self):
        rng = random.Random(42)
        spoofer = AlertSpoofer(rng, phantom_rate=0.20, suppression_rate=0.0)
        phantoms = sum(1 for _ in range(1000)
                       if spoofer.maybe_phantom(100.0, "SN-X") is not None)
        assert 130 < phantoms < 270  # ~20%

    def test_suppression_rate(self):
        rng = random.Random(42)
        spoofer = AlertSpoofer(rng, phantom_rate=0.0, suppression_rate=0.15)
        suppressed = sum(1 for _ in range(1000)
                         if spoofer.should_suppress(100.0, "SN-X"))
        assert 100 < suppressed < 220  # ~15%

    def test_phantom_structure(self):
        rng = random.Random(0)
        spoofer = AlertSpoofer(rng, phantom_rate=1.0)
        alert = spoofer.maybe_phantom(100.0, "SN-X")
        assert alert is not None
        assert alert.alert_type == "phantom"
        assert "channel" in alert.details


# -----------------------------------------------------------------------
# Confounding by indication
# -----------------------------------------------------------------------

class TestConfoundingModel:

    def test_no_alerts_no_boost(self):
        rng = random.Random(0)
        cm = ConfoundingModel(rng, confound_strength=0.5)
        assert cm.priority_boost("SN-X", 0) == 1.0

    def test_more_alerts_more_boost(self):
        rng = random.Random(0)
        cm = ConfoundingModel(rng, confound_strength=0.5)
        b1 = cm.priority_boost("SN-X", 1)
        b5 = cm.priority_boost("SN-X", 5)
        b20 = cm.priority_boost("SN-X", 20)
        assert 1.0 < b1 < b5 < b20


# -----------------------------------------------------------------------
# Data partitioning
# -----------------------------------------------------------------------

class TestPartitioning:

    def test_no_overlap(self):
        config = SimulatorConfig(horizon_days=30)
        events, gt = gen_run(config, seed=42)
        parts = partition_data(events, gt, train_frac=0.7, seed=99)

        train_sns = {g["sn"] for g in parts["train"]["ground_truth"]}
        eval_sns = {e["sn"] for e in parts["eval"]["events"] if e["sn"]}
        test_sns = {e["sn"] for e in parts["test"]["events"] if e["sn"]}

        assert train_sns.isdisjoint(eval_sns)
        assert train_sns.isdisjoint(test_sns)
        assert eval_sns.isdisjoint(test_sns)

    def test_eval_test_gt_sealed(self):
        config = SimulatorConfig(horizon_days=10)
        events, gt = gen_run(config, seed=42)
        parts = partition_data(events, gt)
        assert parts["eval"]["ground_truth"] == []
        assert parts["test"]["ground_truth"] == []

    def test_deterministic(self):
        config = SimulatorConfig(horizon_days=10)
        events, gt = gen_run(config, seed=42)
        p1 = partition_data(events, gt, seed=123)
        p2 = partition_data(events, gt, seed=123)
        assert p1["train"]["ground_truth"] == p2["train"]["ground_truth"]


# -----------------------------------------------------------------------
# Twin-world
# -----------------------------------------------------------------------

class TestTwinWorld:

    def test_identical_without_intervention(self):
        config = SimulatorConfig(horizon_days=5)
        eA, gtA, eB, gtB = run_twin_worlds(config, seed=42)
        # Without any intervention function, worlds should be identical
        assert len(eA) == len(eB)
        assert len(gtA) == len(gtB)

    def test_diverges_with_intervention(self):
        config = SimulatorConfig(horizon_days=10)

        def reset_all_health(sim, events, t):
            """Aggressive intervention: reset every part's health."""
            for sn in sim.part_health:
                sim.part_health[sn] = 0.0
                sim.part_states[sn] = 1

        eA, gtA, eB, gtB = run_twin_worlds(config, seed=42,
                                            apply_alerts_fn=reset_all_health)
        # World A should have fewer failures (everything gets reset)
        failures_a = sum(1 for e in eA if e.event_type == "failure_abort")
        failures_b = sum(1 for e in eB if e.event_type == "failure_abort")
        assert failures_a < failures_b


# -----------------------------------------------------------------------
# Second generator family
# -----------------------------------------------------------------------

class TestAltGeneratorFamily:

    def test_reproducibility(self):
        config = SimulatorConfig(horizon_days=5)
        g1 = AltGeneratorFamily(config, seed=42)
        e1, gt1 = g1.run()
        g2 = AltGeneratorFamily(config, seed=42)
        e2, gt2 = g2.run()
        assert e1 == e2
        assert gt1 == gt2

    def test_different_from_primary(self):
        config = SimulatorConfig(horizon_days=30)
        _, gt_primary = gen_run(config, seed=42)
        alt = AltGeneratorFamily(config, seed=42)
        _, gt_alt = alt.run()
        # Serial numbers differ (ALT- prefix vs SN- prefix)
        primary_sns = {g["sn"] for g in gt_primary}
        alt_sns = {g["sn"] for g in gt_alt}
        assert primary_sns.isdisjoint(alt_sns)

    def test_jet_frailty_varies(self):
        config = SimulatorConfig(horizon_days=5)
        alt = AltGeneratorFamily(config, seed=42)
        frailties = list(alt.jet_frailty.values())
        # Not all jets should have the same frailty
        assert len(set(round(f, 2) for f in frailties)) > 1


# -----------------------------------------------------------------------
# Firewall (reinforced for new module)
# -----------------------------------------------------------------------

class TestFirewall2b:

    def test_observations_does_not_import_nirnay(self):
        import inspect
        import simulator.observations as obs
        src = inspect.getsource(obs)
        assert "from nirnay" not in src
        assert "import nirnay" not in src
