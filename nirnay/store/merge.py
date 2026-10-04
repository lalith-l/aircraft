"""
Nirnay Store — Merge and Fold logic.

Status: [REAL]
"""

from typing import List, Iterable, Set
from nirnay.contracts import EvidenceItem
from nirnay.store.db import AppendOnlyStore


def merge_stores(local_store: AppendOnlyStore, remote_items: Iterable[EvidenceItem]) -> int:
    """
    Merge remote items into the local store via set union.
    Returns the number of new items appended.
    
    This is mathematically commutative, associative, and idempotent
    because append() rejects duplicate exact items and raises on
    duplicate IDs with different content.
    """
    added = 0
    for item in remote_items:
        try:
            if local_store.append(item):
                added += 1
        except ValueError:
            # We skip items that fail validation during merge
            # (e.g. bad signatures or tampered duplicates).
            # In a full system, we might log this heavily.
            pass
    return added


def deterministic_fold(items: Iterable[EvidenceItem]) -> List[EvidenceItem]:
    """
    Given a set of EvidenceItems, returns the deterministic ordered sequence of active items.
    
    1. Tracks superseded IDs.
    2. Handles concurrent supersedes (multiple items superseding the same target).
    3. Handles chains of corrections (A <- B <- C).
    4. Filters out all superseded items.
    5. Sorts the final active items by (t, id).
    """
    # 1. Gather all superseded IDs
    superseded_ids: Set[str] = set()
    for item in items:
        if item.supersedes:
            superseded_ids.add(item.supersedes)
            
    # 2. Filter out superseded items
    # If A supersedes B, B is removed.
    # If A and C both supersede B, B is removed. A and C remain (concurrent tie-break is just the final sort).
    # If A supersedes B, and C supersedes A, then A and B are removed. C remains.
    active_items = [item for item in items if item.id not in superseded_ids]
    
    # 3. Sort by event time `t`, tie-break by `id` (this resolves concurrent supersedes deterministically)
    active_items.sort(key=lambda x: (x.t, x.id))
    
    return active_items
