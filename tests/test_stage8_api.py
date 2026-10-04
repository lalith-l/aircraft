"""
Stage 8 tests — UI, API and Security.

Tests API endpoints, WebSocket, sync/merge, partition test, and tamper rejection.
"""

import pytest
import json
import numpy as np
from unittest.mock import MagicMock

from nirnay.api.security import SyncNode, partition_test
from nirnay.contracts import EvidenceItem, EvidenceSource
from nirnay.demo.tamper_demo import verify_chain, demo_tamper_rejection


# ---------------------------------------------------------------------------
# API Endpoint Tests
# ---------------------------------------------------------------------------

class TestAPIEndpoints:

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from nirnay.api.server import app
        return TestClient(app)

    def test_fleet_endpoint(self, client):
        resp = client.get("/api/fleet")
        assert resp.status_code == 200
        data = resp.json()
        assert "jets" in data
        assert "readiness" in data
        assert len(data["jets"]) == 12

    def test_aircraft_endpoint(self, client):
        resp = client.get("/api/aircraft/JET-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["jet"]["id"] == "JET-001"
        assert len(data["parts"]) == 12

    def test_aircraft_not_found(self, client):
        resp = client.get("/api/aircraft/NONEXISTENT")
        assert resp.status_code == 404

    def test_passport_endpoint(self, client):
        resp = client.get("/api/passport/JET-001-engine_L")
        assert resp.status_code == 200
        data = resp.json()
        assert "timeline" in data
        assert len(data["timeline"]) > 0

    def test_windtunnel_endpoint(self, client):
        resp = client.get("/api/windtunnel")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["scenarios"]) == 4

    def test_replay_endpoint(self, client):
        resp = client.get("/api/replay/5")
        assert resp.status_code == 200
        data = resp.json()
        assert data["day"] == 5
        assert "events" in data

    def test_replay_out_of_range(self, client):
        resp = client.get("/api/replay/0")
        assert resp.status_code == 400

    def test_depot_endpoint(self, client):
        resp = client.get("/api/depot")
        assert resp.status_code == 200
        data = resp.json()
        assert "teardowns" in data

    def test_commander_endpoint(self, client):
        resp = client.get("/api/commander")
        assert resp.status_code == 200
        data = resp.json()
        assert "frontier" in data
        assert "shadow_prices" in data
        assert "promise" in data

    def test_sync_status(self, client):
        resp = client.get("/api/security/sync_status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["in_sync"] is True


# ---------------------------------------------------------------------------
# Security / Sync Tests
# ---------------------------------------------------------------------------

def _make_item(part_sn: str, t: float, prev_hash: str = "",
               node: str = "test") -> EvidenceItem:
    item = EvidenceItem(
        part_sn=part_sn,
        source=EvidenceSource.SENSOR,
        node=node,
        t=t,
        t_logged=t,
        payload={"v": t},
        likelihood={"per_state": [0.2, 0.2, 0.2, 0.2, 0.2]},
        weight=0.5,
        prev_hash=prev_hash,
    )
    return item.with_hash()


class TestSyncNode:

    @pytest.fixture(autouse=True)
    def mock_verify(self, monkeypatch):
        from nirnay.store.crypto import KeyRegistry
        monkeypatch.setattr(KeyRegistry, "verify", lambda *args, **kwargs: True)

    def test_merge_adds_missing_items(self):
        node_a = SyncNode("a", ":memory:")
        node_b = SyncNode("b", ":memory:")

        item1 = _make_item("SN-1", 1000.0, node="a")
        item2 = _make_item("SN-1", 2000.0, node="b")

        node_a.append(item1)
        node_b.append(item2)

        count = node_a.merge_from(node_b)
        assert count == 1
        assert item2.id in node_a.items

    def test_merge_is_idempotent(self):
        node_a = SyncNode("a", ":memory:")
        node_b = SyncNode("b", ":memory:")

        item1 = _make_item("SN-1", 1000.0, node="a")
        node_a.append(item1)
        node_b.append(item1)

        count = node_a.merge_from(node_b)
        assert count == 0


class TestPartitionTest:

    @pytest.fixture(autouse=True)
    def mock_verify(self, monkeypatch):
        from nirnay.store.crypto import KeyRegistry
        monkeypatch.setattr(KeyRegistry, "verify", lambda *args, **kwargs: True)

    def test_beliefs_match_after_merge(self):
        items_a = [_make_item("SN-X", 1000.0 + i * 100) for i in range(3)]
        items_b = [_make_item("SN-X", 2000.0 + i * 100) for i in range(3)]

        result = partition_test(items_a, items_b)

        assert result["match"] is True
        for part, info in result["parts"].items():
            assert info["match"] is True
            # Alpha, beta, and central should have identical beliefs
            assert info["beliefs_alpha"] == info["beliefs_beta"]
            assert info["beliefs_alpha"] == info["beliefs_central"]


class TestTamperRejection8:

    def test_tamper_detected(self):
        reason = demo_tamper_rejection()
        assert "mismatch" in reason.lower() or "Hash" in reason

    def test_clean_chain_ok(self):
        items = [_make_item("SN-1", 1000.0 + i * 100) for i in range(5)]
        # Fix chain links
        chained = [items[0]]
        for i in range(1, len(items)):
            item = EvidenceItem(
                part_sn=items[i].part_sn,
                source=items[i].source,
                node=items[i].node,
                t=items[i].t,
                t_logged=items[i].t_logged,
                payload=items[i].payload,
                likelihood=items[i].likelihood,
                weight=items[i].weight,
                prev_hash=chained[-1].hash,
            )
            chained.append(item.with_hash())

        valid, reason = verify_chain(chained)
        assert valid
