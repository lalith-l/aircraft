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
    Given a set of EvidenceItems (e.g. all items for a specific part),
    returns the deterministic ordered sequence of active items.
    
    1. Sorts items by (t, id). This guarantees deterministic ordering
       even if two items arrive with the exact same timestamp.
    2. Filters out superseded items.
    
    The belief engine takes this sorted sequence and folds it into a belief.
    """
    # 1. Gather all superseded IDs
    superseded_ids: Set[str] = set()
    for item in items:
        if item.supersedes:
            superseded_ids.add(item.supersedes)
            
    # 2. Filter out superseded items
    active_items = [item for item in items if item.id not in superseded_ids]
    
    # 3. Sort by event time `t`, tie-break by `id`
    active_items.sort(key=lambda x: (x.t, x.id))
    
    return active_items
