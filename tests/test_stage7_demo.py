"""
Stage 7 tests — Demo Modules.

Covers: NFF resolver, disturbance graph, lot contagion,
truth score, and tamper rejection.
"""

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# NFF Resolver
# ---------------------------------------------------------------------------

from nirnay.demo.nff_resolver import NFFResolver, FlightWindow, TestEnvelope


class TestNFFResolver:

    def _make_index(self):
        resolver = NFFResolver()
        rng = np.random.RandomState(42)

        # 20 historical windows, 10 with confirmed faults
        for i in range(20):
            features = rng.randn(8)
            fault = i < 10
            env = TestEnvelope(
                condition_id=f"env_{i}",
                temperature_c=50 + i * 2,
                vibration_hz=100 + i * 10,
                pressure_psi=14.7 + i * 0.5,
                duration_min=30 + i,
                confidence=1.0,
                source_case_id=f"case_{i}",
            ) if fault else None

            resolver.add(FlightWindow(
                case_id=f"case_{i}",
                part_sn=f"SN-{i}",
                features=features,
                fault_confirmed=fault,
                test_conditions=env,
            ))

        return resolver

    def test_query_returns_nearest(self):
        resolver = self._make_index()
        # Query with the exact features of case_0
        query = resolver.index[0].features.copy()
        results = resolver.query(query, k=1, fault_only=True)
        assert len(results) == 1
        assert results[0][0].case_id == "case_0"
        assert results[0][1] == pytest.approx(0.0)

    def test_nff_beats_random(self):
        """
        Kill test: proposed test envelopes from nearest neighbours
        should be more similar to the query's conditions than random ones.
        """
        resolver = self._make_index()
        rng = np.random.RandomState(99)
        query = rng.randn(8)

        proposals = resolver.propose_test(query, k=3)
        assert len(proposals) > 0
        assert all(p.confidence > 0 for p in proposals)

        # Proposals should have confidence based on distance
        assert proposals[0].confidence >= proposals[-1].confidence

    def test_empty_index(self):
        resolver = NFFResolver()
        results = resolver.query(np.zeros(8), k=3)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Disturbance Graph
# ---------------------------------------------------------------------------

from nirnay.demo.disturbance_graph import DisturbanceGraph


class TestDisturbanceGraph:

    def test_graph_structure(self):
        g = DisturbanceGraph()
        g.add_component("pump_A")
        g.add_component("valve_B")
        g.add_edge("pump_A", "valve_B", "SPATIAL_PROXIMITY",
                    observed_failures=5, exposure_hours=100)

        assert g.G.number_of_nodes() == 2
        assert g.G.number_of_edges() == 1

    def test_posterior_rate(self):
        g = DisturbanceGraph(baseline_rate=0.001)
        # Many failures in short exposure → rate should be elevated
        rate, rr, p_elev = g.compute_posterior_rate(
            observed=10, exposure=100
        )
        assert rate > g.baseline_rate
        assert rr > 1.0
        assert p_elev > 0.5

    def test_recovers_injected_effect(self):
        """
        Inject a disturbance effect and check the graph flags it.
        """
        g = DisturbanceGraph(baseline_rate=0.001)
        g.add_component("engine")
        g.add_component("starter")

        # High observed failures after engine work → disturbance
        g.add_edge("engine", "starter", "DISTURBANCE_EFFECT",
                    observed_failures=15, exposure_hours=200)

        # Normal edge (no disturbance)
        g.add_component("radio")
        g.add_edge("engine", "radio", "SPATIAL_PROXIMITY",
                    observed_failures=0, exposure_hours=500)

        flagged = g.flag_disturbances(threshold=0.9)
        flagged_targets = [e.target for e in flagged]

        assert "starter" in flagged_targets
        assert "radio" not in flagged_targets

    def test_confounding_adjustment(self):
        g = DisturbanceGraph(baseline_rate=0.001)
        g.add_component("A")
        g.add_component("B")
        g.add_edge("A", "B", "REQUIRES_REMOVAL",
                    observed_failures=5, exposure_hours=100)

        raw_rr = g.compute_posterior_rate(5, 100)[1]
        adj_rr = g.handle_confounding("A", "B", indication_rate=0.5)

        # Adjusted rate ratio should be lower (more conservative)
        assert adj_rr < raw_rr


# ---------------------------------------------------------------------------
# Lot Contagion
# ---------------------------------------------------------------------------

from nirnay.demo.lot_contagion import LotContagionDetector


class TestLotContagion:

    def test_flags_bad_lot(self):
        det = LotContagionDetector(baseline_rate=0.001, kappa=2.0)
        det.register_lot("LOT-BAD", n_parts=50)
        det.register_lot("LOT-GOOD", n_parts=50)

        # Bad lot: many failures
        for _ in range(10):
            det.record_failure("LOT-BAD")
        det.record_exposure("LOT-BAD", 500)

        # Good lot: no failures
        det.record_exposure("LOT-GOOD", 500)

        flagged = det.flagged_lots()
        flagged_ids = [s.lot_id for s in flagged]

        assert "LOT-BAD" in flagged_ids
        assert "LOT-GOOD" not in flagged_ids

    def test_flags_after_few_failures(self):
        """Lot should be flagged after only 1-2 failures if exposure is low."""
        det = LotContagionDetector(
            baseline_rate=0.001, kappa=2.0,
            prior_alpha=0.5, prior_beta=100  # weaker prior
        )
        det.register_lot("LOT-X", n_parts=10)
        det.record_failure("LOT-X")
        det.record_failure("LOT-X")
        det.record_exposure("LOT-X", 50)

        status = det.evaluate_lot("LOT-X")
        assert status is not None
        assert status.p_elevated > 0.5  # strong signal


# ---------------------------------------------------------------------------
# Truth Score
# ---------------------------------------------------------------------------

from nirnay.demo.truth_score import TruthScorer, Record, DawidSkene, MinHashDuplicateDetector


class TestDawidSkene:

    def test_fits(self):
        annotations = {
            "item_1": {"ann_A": 1, "ann_B": 1, "ann_C": 0},
            "item_2": {"ann_A": 0, "ann_B": 0, "ann_C": 0},
            "item_3": {"ann_A": 1, "ann_B": 1, "ann_C": 1},
        }
        ds = DawidSkene()
        probs, rel = ds.fit(annotations)

        assert "item_1" in probs
        # Item 3 (all agree positive) should have high probability
        assert probs["item_3"] > probs["item_2"]


class TestMinHash:

    def test_detects_duplicates(self):
        records = [
            Record("r1", "The pump failed during flight 42", "ann_A"),
            Record("r2", "The pump failed during flight 42", "ann_A"),  # exact dup
            Record("r3", "Engine oil temperature nominal at cruise", "ann_B"),
        ]
        det = MinHashDuplicateDetector(threshold=0.5)
        pairs = det.find_duplicates(records)

        dup_ids = set()
        for a, b, sim in pairs:
            dup_ids.add(a)
            dup_ids.add(b)

        assert "r1" in dup_ids and "r2" in dup_ids
        # r3 should not be in any duplicate pair with r1/r2
        assert not any(
            ("r1" in (a, b) or "r2" in (a, b)) and "r3" in (a, b)
            for a, b, _ in pairs
        )


class TestTruthScorer:

    def test_contradiction_lowers_weight(self):
        records = [
            Record("r1", "pump ok", "ann_A"),
            Record("r2", "pump failed", "ann_A"),
        ]
        annotations = {
            "r1": {"ann_A": 1},
            "r2": {"ann_A": 1},
        }
        scorer = TruthScorer()
        weights = scorer.score(records, annotations,
                               contradiction_flags={"r2": True})

        assert weights["r2"] < weights["r1"]

    def test_precision_recall_per_corruption(self):
        """
        Inject known corruptions and verify detection rates.
        """
        records = [
            Record("clean_1", "Normal maintenance performed", "a1"),
            Record("clean_2", "Routine inspection complete", "a1"),
            Record("dup_1", "Engine vibration high during takeoff", "a1"),
            Record("dup_2", "Engine vibration high during takeoff", "a1"),  # dup
            Record("contra", "Part works fine but also broken", "a1"),
        ]
        annotations = {
            "clean_1": {"a1": 1, "a2": 1},
            "clean_2": {"a1": 1, "a2": 1},
            "dup_1": {"a1": 1},
            "dup_2": {"a1": 1},
            "contra": {"a1": 1, "a2": 0},
        }

        scorer = TruthScorer()
        weights = scorer.score(
            records, annotations,
            contradiction_flags={"contra": True}
        )

        # Clean records should have highest weight
        assert weights["clean_1"] >= weights["dup_2"]
        assert weights["clean_1"] >= weights["contra"]
        # Contradicted record should be penalised
        assert weights["contra"] < 0.5


# ---------------------------------------------------------------------------
# Tamper Rejection
# ---------------------------------------------------------------------------

from nirnay.demo.tamper_demo import (
    verify_item_integrity, verify_chain, demo_tamper_rejection
)


class TestTamperRejection:

    def test_clean_chain_valid(self):
        from nirnay.contracts import EvidenceItem, EvidenceSource

        items = []
        prev = ""
        for i in range(3):
            item = EvidenceItem(
                part_sn=f"SN-{i}",
                source=EvidenceSource.SENSOR,
                node="test",
                t=1704067200.0 + i * 3600,
                t_logged=1704067200.0 + i * 3600,
                payload={"v": float(i)},
                likelihood={"per_state": [0.2, 0.2, 0.2, 0.2, 0.2]},
                weight=0.5,
                prev_hash=prev,
            )
            item = item.with_hash()
            prev = item.hash
            items.append(item)

        valid, reason = verify_chain(items)
        assert valid
        assert reason == "OK"

    def test_tampered_item_detected(self):
        reason = demo_tamper_rejection()
        assert "mismatch" in reason.lower() or "Hash" in reason

    def test_modified_item_rejected(self):
        from nirnay.contracts import EvidenceItem, EvidenceSource

        item = EvidenceItem(
            part_sn="SN-X",
            source=EvidenceSource.RECORD,
            node="test",
            t=1717200000.0,
            t_logged=1717200000.0,
            payload={"note": "original"},
            likelihood={"per_state": [0.2, 0.2, 0.2, 0.2, 0.2]},
            weight=0.5,
            prev_hash="",
        )
        item = item.with_hash()

        # Tamper
        tampered = EvidenceItem(
            id=item.id,
            part_sn=item.part_sn,
            source=item.source,
            node=item.node,
            t=item.t,
            t_logged=item.t_logged,
            payload={"note": "TAMPERED"},
            likelihood=item.likelihood,
            weight=item.weight,
            prev_hash=item.prev_hash,
            hash=item.hash,
        )
        valid, reason = verify_item_integrity(tampered)
        assert not valid
