"""
Tamper Rejection Demo

Demonstrates that a modified evidence item is detected and rejected
by the cryptographic hash-chain and signature verification.

Status: [DEMO over real code]
"""

from typing import Tuple
from nirnay.contracts import EvidenceItem


def verify_item_integrity(item: EvidenceItem, expected_hash: str = ""
                          ) -> Tuple[bool, str]:
    """
    Verify that an EvidenceItem has not been tampered with.

    Checks:
    1. Recompute the canonical hash and compare.
    2. If expected_hash is given, check it matches.

    Returns: (is_valid, reason_if_invalid)
    """
    # Recompute
    recomputed = item.with_hash()

    if recomputed.hash != item.hash:
        return False, (
            f"Hash mismatch: stored={item.hash[:16]}... "
            f"recomputed={recomputed.hash[:16]}..."
        )

    if expected_hash and item.hash != expected_hash:
        return False, (
            f"Expected hash {expected_hash[:16]}... "
            f"but got {item.hash[:16]}..."
        )

    return True, "OK"


def verify_chain(items: list) -> Tuple[bool, str]:
    """
    Verify the hash chain across a sequence of EvidenceItems.

    Returns: (is_valid, reason_if_invalid)
    """
    for i, item in enumerate(items):
        # Verify individual hash
        valid, reason = verify_item_integrity(item)
        if not valid:
            return False, f"Item {i} ({item.id}): {reason}"

        # Verify chain link
        if i > 0:
            if item.prev_hash != items[i - 1].hash:
                return False, (
                    f"Chain break at item {i}: prev_hash={item.prev_hash[:16]}... "
                    f"!= items[{i-1}].hash={items[i-1].hash[:16]}..."
                )

    return True, "OK"


def demo_tamper_rejection():
    """
    Demonstrate tamper rejection.

    1. Create a valid evidence chain.
    2. Modify one item's payload.
    3. Show that verification detects the tampering.
    """
    from nirnay.contracts import EvidenceSource

    # Create a valid chain
    items = []
    prev = ""
    for i in range(3):
        item = EvidenceItem(
            part_sn=f"SN-DEMO-{i}",
            source=EvidenceSource.SENSOR,
            node="demo_node",
            t=1704067200.0 + i * 3600,
            t_logged=1704067200.0 + i * 3600,
            payload={"reading": float(i) * 10.0},
            likelihood={"per_state": [0.2, 0.2, 0.2, 0.2, 0.2]},
            weight=0.5,
            prev_hash=prev,
        )
        item = item.with_hash()
        prev = item.hash
        items.append(item)

    # Verify the clean chain
    valid, reason = verify_chain(items)
    assert valid, f"Clean chain should be valid: {reason}"

    # Tamper with the middle item
    tampered = items[1]
    # Modify the payload (this should invalidate the hash)
    tampered_payload = dict(tampered.payload)
    tampered_payload["reading"] = 999.0
    tampered = EvidenceItem(
        id=tampered.id,
        part_sn=tampered.part_sn,
        source=tampered.source,
        node=tampered.node,
        t=tampered.t,
        t_logged=tampered.t_logged,
        payload=tampered_payload,
        likelihood=tampered.likelihood,
        weight=tampered.weight,
        prev_hash=tampered.prev_hash,
        hash=tampered.hash,  # Keep the old hash
    )
    items[1] = tampered

    # Verify again — should fail
    valid, reason = verify_chain(items)
    assert not valid, "Tampered chain should be detected"

    return reason
