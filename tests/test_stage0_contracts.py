"""
Stage 0 tests — Scaffold and Contracts.

Gate: All contract classes instantiate, serialise to JSON, round-trip.
      pytest discovers and runs. Simulator firewall test passes. make test exits 0.
"""

import json
import ast
import os
import sys

import pytest

from nirnay.contracts import (
    EvidenceItem, EvidenceSource, HealthState, StatusTag, PartClass,
    MissionType, Regime, Passport, Alert, Plan, Scenario,
)


# -----------------------------------------------------------------------
# Contract instantiation and round-trip
# -----------------------------------------------------------------------

class TestEvidenceItem:
    """EvidenceItem must be immutable, hashable, and JSON round-trippable."""

    def test_create_default(self):
        item = EvidenceItem()
        assert item.source == EvidenceSource.SENSOR
        assert item.weight == 1.0
        assert item.supersedes is None

    def test_create_full(self):
        item = EvidenceItem(
            id="ev-001", part_sn="SN-123", source=EvidenceSource.TEARDOWN,
            node="base-A", t=1000.0, t_logged=1001.0,
            payload={"grade": 2},
            likelihood={"per_state": [0.1, 0.2, 0.4, 0.2, 0.1]},
            weight=0.85, prev_hash="abc123", hash="", sig="",
            supersedes=None,
        )
        assert item.part_sn == "SN-123"
        assert item.source == EvidenceSource.TEARDOWN

    def test_frozen(self):
        item = EvidenceItem()
        with pytest.raises(AttributeError):
            item.weight = 0.5  # type: ignore

    def test_compute_hash_deterministic(self):
        item = EvidenceItem(id="ev-001", part_sn="SN-123", t=1000.0)
        h1 = item.compute_hash()
        h2 = item.compute_hash()
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_compute_hash_changes_with_content(self):
        a = EvidenceItem(id="ev-001", part_sn="SN-123", t=1000.0)
        b = EvidenceItem(id="ev-001", part_sn="SN-124", t=1000.0)
        assert a.compute_hash() != b.compute_hash()

    def test_with_hash(self):
        item = EvidenceItem(id="ev-001", part_sn="SN-123")
        hashed = item.with_hash()
        assert hashed.hash == hashed.compute_hash()
        assert hashed.hash != ""

    def test_json_round_trip(self):
        item = EvidenceItem(
            id="ev-rt", part_sn="SN-RT", source=EvidenceSource.LAYA,
            node="depot-D", t=2000.0, t_logged=2001.0,
            payload={"text": "vibration high"},
            likelihood={"per_state": [0.05, 0.1, 0.5, 0.3, 0.05]},
            weight=0.9, prev_hash="prev", hash="h", sig="s",
            supersedes="ev-old",
        )
        json_str = item.to_json()
        restored = EvidenceItem.from_json(json_str)
        assert restored.id == item.id
        assert restored.part_sn == item.part_sn
        assert restored.source == item.source
        assert restored.payload == item.payload
        assert restored.likelihood == item.likelihood
        assert restored.weight == item.weight
        assert restored.supersedes == item.supersedes

    def test_canonical_excludes_hash_and_sig(self):
        item = EvidenceItem(id="ev-001", hash="should_be_gone", sig="also_gone")
        cd = item.canonical_dict()
        assert "hash" not in cd
        assert "sig" not in cd


class TestPassport:
    def test_create_default(self):
        p = Passport()
        assert p.part_class == PartClass.WEAROUT

    def test_json_round_trip(self):
        p = Passport(
            part_sn="SN-P1", part_class=PartClass.RANDOM,
            lot="LOT-042", configuration="v2.1",
            installed_history=[{"aircraft_sn": "T-104", "from_t": 0.0, "to_t": None}],
            exposure=[{"t": 100.0, "hours": 50.0, "regime": "normal"}],
        )
        restored = Passport.from_json(p.to_json())
        assert restored.part_sn == p.part_sn
        assert restored.part_class == PartClass.RANDOM
        assert restored.lot == p.lot
        assert len(restored.installed_history) == 1
        assert len(restored.exposure) == 1


class TestAlert:
    def test_json_round_trip(self):
        a = Alert(part_sn="SN-A1", M=12.5, threshold_b=10.0,
                  tier=2, top_evidence_ids=["ev-1", "ev-2"], budget_left=3.0)
        restored = Alert.from_json(a.to_json())
        assert restored.part_sn == a.part_sn
        assert restored.M == a.M
        assert restored.tier == 2
        assert len(restored.top_evidence_ids) == 2


class TestPlan:
    def test_json_round_trip(self):
        p = Plan(
            horizon=72.0,
            actions=[{"aircraft_sn": "T-101", "part_sn": "SN-1",
                       "action_type": "replace", "start_t": 0, "end_t": 4}],
            frontier=[{"n_cas": 10, "n_cap": 8, "p": 0.92}],
            promise={"alpha": 0.1, "lambda": 1.5, "number": 18},
            shadow_prices=[{"resource": "hydraulic_pump",
                             "echelon": "base", "elasticity": 0.3, "price": 50000}],
        )
        restored = Plan.from_json(p.to_json())
        assert restored.horizon == 72.0
        assert len(restored.actions) == 1
        assert len(restored.frontier) == 1
        assert restored.promise["number"] == 18


class TestScenario:
    def test_json_round_trip(self):
        s = Scenario(id="sc-1", name="bad_lot_surge",
                     params={"lead_time_multiplier": 2.0, "bad_lot": "LOT-042"})
        restored = Scenario.from_json(s.to_json())
        assert restored.name == "bad_lot_surge"
        assert restored.params["lead_time_multiplier"] == 2.0


# -----------------------------------------------------------------------
# Enum coverage
# -----------------------------------------------------------------------

class TestEnums:
    def test_health_states_count(self):
        assert len(HealthState) == 5
        assert HealthState.FAILED == 4

    def test_evidence_sources(self):
        assert len(EvidenceSource) == 6

    def test_status_tags(self):
        assert StatusTag.NOT_RUN.value == "NOT_RUN"

    def test_part_classes(self):
        assert PartClass.WEAROUT.value == "wearout"
        assert PartClass.RANDOM.value == "random"

    def test_mission_types(self):
        assert MissionType.CAS.value == "close_air_support"

    def test_regimes(self):
        assert len(Regime) == 3


# -----------------------------------------------------------------------
# Simulator firewall — nirnay/ must NEVER import simulator/
# -----------------------------------------------------------------------

class TestSimulatorFirewall:
    """
    Enforced rule: the nirnay package must never import the simulator package.
    Walk every .py file in nirnay/ and check that no import references simulator.
    """

    def test_nirnay_does_not_import_simulator(self):
        nirnay_dir = os.path.join(os.path.dirname(__file__), "..", "nirnay")
        nirnay_dir = os.path.abspath(nirnay_dir)
        violations = []

        for root, _dirs, files in os.walk(nirnay_dir):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                fpath = os.path.join(root, fname)
                with open(fpath, "r") as f:
                    try:
                        tree = ast.parse(f.read(), filename=fpath)
                    except SyntaxError:
                        continue

                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name.startswith("simulator"):
                                violations.append(
                                    f"{fpath}:{node.lineno} — import {alias.name}"
                                )
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and node.module.startswith("simulator"):
                            violations.append(
                                f"{fpath}:{node.lineno} — from {node.module} import ..."
                            )

        assert violations == [], (
            "nirnay/ must NEVER import simulator/. Violations:\n"
            + "\n".join(violations)
        )


# -----------------------------------------------------------------------
# Directory structure
# -----------------------------------------------------------------------

class TestDirectoryStructure:
    """Verify the required repository scaffold exists."""

    REQUIRED_DIRS = [
        "nirnay", "nirnay/store", "nirnay/belief", "nirnay/detect",
        "nirnay/laya", "nirnay/plan", "nirnay/demo", "nirnay/api",
        "simulator", "ui", "rig", "experiments", "reports", "tests",
    ]

    REQUIRED_FILES = [
        "nirnay/__init__.py",
        "nirnay/contracts.py",
        "simulator/__init__.py",
        "requirements.txt",
        "Makefile",
        "pytest.ini",
        "LIMITATIONS.md",
        "README.md",
    ]

    def test_directories_exist(self):
        root = os.path.join(os.path.dirname(__file__), "..")
        root = os.path.abspath(root)
        missing = [d for d in self.REQUIRED_DIRS
                   if not os.path.isdir(os.path.join(root, d))]
        assert missing == [], f"Missing directories: {missing}"

    def test_files_exist(self):
        root = os.path.join(os.path.dirname(__file__), "..")
        root = os.path.abspath(root)
        missing = [f for f in self.REQUIRED_FILES
                   if not os.path.isfile(os.path.join(root, f))]
        assert missing == [], f"Missing files: {missing}"


# -----------------------------------------------------------------------
# Experiment configs
# -----------------------------------------------------------------------

class TestExperimentConfigs:
    """All experiment config files exist and are valid JSON with NOT_RUN status."""

    EXPERIMENTS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

    def test_all_configs_exist_and_valid(self):
        root = os.path.join(os.path.dirname(__file__), "..", "experiments")
        root = os.path.abspath(root)
        for exp in self.EXPERIMENTS:
            path = os.path.join(root, exp, "config.json")
            assert os.path.isfile(path), f"Missing config for experiment {exp}"
            with open(path) as f:
                data = json.load(f)
            assert data["status"] == "NOT_RUN", \
                f"Experiment {exp} should be NOT_RUN at Stage 0"
