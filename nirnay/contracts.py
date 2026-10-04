"""
Nirnay Data Contracts — Immutable, serialisable data structures.

These are the ONLY stored objects in the system (Section 4 of the blueprint).
Beliefs are derived, never stored as truth. Every module is a function of
passports and writes new items; no module keeps private data.

Status: [REAL]
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EvidenceSource(str, Enum):
    """Origin of an evidence item."""
    SENSOR = "sensor"
    SIBLING = "sibling"
    LAYA = "laya"
    TEARDOWN = "teardown"
    RECORD = "record"
    MANUAL = "manual"


class HealthState(int, Enum):
    """Hidden health states for the HMM belief engine (§5.2)."""
    NEW = 0
    GOOD = 1
    DEGRADED = 2
    SEVERE = 3
    FAILED = 4  # absorbing


class StatusTag(str, Enum):
    """Status labels shown on screen and in reports (§1)."""
    REAL = "REAL"
    SIMULATED = "SIMULATED"
    DEMO = "DEMO"
    PLANNED = "PLANNED"
    NOT_RUN = "NOT_RUN"


class PartClass(str, Enum):
    """Part failure-mode classes."""
    WEAROUT = "wearout"      # Weibull shape > 1
    RANDOM = "random"        # Weibull shape ≈ 1


class MissionType(str, Enum):
    """Mission types for the fleet."""
    CAS = "close_air_support"
    CAP = "combat_air_patrol"


class Regime(str, Enum):
    """Operating regimes."""
    NORMAL = "normal"
    HOT_DUSTY = "hot_dusty"
    SURGE_HIGH_G = "surge_high_g"


# ---------------------------------------------------------------------------
# Core data contracts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvidenceItem:
    """
    Immutable, signed evidence item (§4).

    Hash chains are per originating node.  The union of all nodes' chains
    is the part's evidence set.  Corrections append (supersedes), never delete.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    part_sn: str = ""
    source: EvidenceSource = EvidenceSource.SENSOR
    node: str = ""                          # originating node id
    t: float = 0.0                          # event time (epoch seconds)
    t_logged: float = 0.0                   # when logged (epoch seconds)
    payload: dict = field(default_factory=dict)  # features, text, grade
    likelihood: dict = field(default_factory=dict)
    # likelihood is either:
    #   {"per_state": [l0, l1, l2, l3, l4]}   — continuous obs
    #   {"grade": g}                           — teardown grade
    weight: float = 1.0                     # trust weight in [0, 1]
    prev_hash: str = ""                     # previous item hash from SAME node
    hash: str = ""                          # SHA-256 of canonical form
    sig: str = ""                           # ed25519 signature by originating node
    supersedes: Optional[str] = None        # id of superseded item, or None

    def canonical_dict(self) -> dict:
        """Return the dict used for hashing — excludes hash and sig fields."""
        d = asdict(self)
        d.pop("hash", None)
        d.pop("sig", None)
        return d

    def compute_hash(self) -> str:
        """SHA-256 over canonical JSON (sorted keys, no hash/sig)."""
        canonical = json.dumps(self.canonical_dict(), sort_keys=True,
                               separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def to_json(self) -> str:
        """Serialise to JSON string."""
        d = asdict(self)
        # Convert enums to their values for clean JSON
        d["source"] = self.source.value
        return json.dumps(d, sort_keys=True, separators=(",", ":"),
                          default=str)

    @classmethod
    def from_json(cls, s: str) -> "EvidenceItem":
        """Deserialise from JSON string."""
        d = json.loads(s)
        d["source"] = EvidenceSource(d["source"])
        return cls(**d)

    def with_hash(self) -> "EvidenceItem":
        """Return a new item with the hash field computed."""
        return EvidenceItem(
            id=self.id, part_sn=self.part_sn, source=self.source,
            node=self.node, t=self.t, t_logged=self.t_logged,
            payload=self.payload, likelihood=self.likelihood,
            weight=self.weight, prev_hash=self.prev_hash,
            hash=self.compute_hash(), sig=self.sig,
            supersedes=self.supersedes,
        )


@dataclass(frozen=True)
class Passport:
    """
    Part passport — the carrier of belief (§3).
    Belief travels with the part between airframes and bases.
    """
    part_sn: str = ""
    part_class: PartClass = PartClass.WEAROUT
    lot: str = ""
    configuration: str = ""
    installed_history: list = field(default_factory=list)
    # Each entry: {"aircraft_sn": str, "from_t": float, "to_t": float|None}
    exposure: list = field(default_factory=list)
    # Each entry: {"t": float, "hours": float, "regime": str}

    def to_json(self) -> str:
        d = asdict(self)
        d["part_class"] = self.part_class.value
        return json.dumps(d, sort_keys=True, separators=(",", ":"),
                          default=str)

    @classmethod
    def from_json(cls, s: str) -> "Passport":
        d = json.loads(s)
        d["part_class"] = PartClass(d["part_class"])
        return cls(**d)


@dataclass(frozen=True)
class Alert:
    """
    Budgeted alert (§3.5).

    M is the e-detector statistic; threshold_b is the budget-derived
    threshold; tier is escalation level.
    """
    part_sn: str = ""
    M: float = 0.0
    threshold_b: float = 1.0
    tier: int = 1
    top_evidence_ids: list = field(default_factory=list)
    budget_left: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True,
                          separators=(",", ":"), default=str)

    @classmethod
    def from_json(cls, s: str) -> "Alert":
        return cls(**json.loads(s))


@dataclass(frozen=True)
class Plan:
    """
    Mission-aware readiness plan (§9).

    frontier: list of (n_cas, n_cap, p) tuples showing the feasible
    mission-mix frontier.
    promise: calibrated readiness number with alpha, lambda, number.
    shadow_prices: marginal value of each resource.
    """
    horizon: float = 48.0  # hours
    actions: list = field(default_factory=list)
    # Each action: {"aircraft_sn", "part_sn", "action_type", "start_t", "end_t"}
    frontier: list = field(default_factory=list)
    # Each point: {"n_cas": int, "n_cap": int, "p": float}
    promise: dict = field(default_factory=lambda: {
        "alpha": 0.1, "lambda": 1.0, "number": 0
    })
    shadow_prices: list = field(default_factory=list)
    # Each: {"resource": str, "echelon": str, "elasticity": float, "price": float}

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True,
                          separators=(",", ":"), default=str)

    @classmethod
    def from_json(cls, s: str) -> "Plan":
        return cls(**json.loads(s))


@dataclass(frozen=True)
class Scenario:
    """
    Scenario card for the wind-tunnel / what-if view (§4).
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    params: dict = field(default_factory=dict)
    # Example params:
    # {"lead_time_multiplier": 2.0, "base_cut": "B",
    #  "bad_lot": "LOT-042", "spoofed_alerts": 5}

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True,
                          separators=(",", ":"), default=str)

    @classmethod
    def from_json(cls, s: str) -> "Scenario":
        return cls(**json.loads(s))
