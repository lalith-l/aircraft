"""
Stage 1 follow-up tests: associativity, k-way partition, verify_store, supersedes edge cases.
"""

import time
import sqlite3
import pytest
from hypothesis import given, settings, strategies as st, HealthCheck
import random
from nirnay.contracts import EvidenceItem
from nirnay.store.crypto import KeyRegistry, NodeIdentity, global_registry
from nirnay.store.db import AppendOnlyStore
from nirnay.store.merge import merge_stores, deterministic_fold
from tests.test_stage1_store import create_item, item_sequences

# -----------------------------------------------------------------------
# Setup helpers
# -----------------------------------------------------------------------

@pytest.fixture
def registry():
    reg = KeyRegistry()
    return reg

@pytest.fixture
def node_a(registry):
    node = NodeIdentity("nodeA")
    registry.register(node.node_id, node.public_key_hex())
    return node

# -----------------------------------------------------------------------
# Hypothesis Properties: Associativity & k-way partition
# -----------------------------------------------------------------------

class TestMergePropertiesExtended:

    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(item_sequences())
    def test_associative(self, tmp_path, items):
        """(A merge B) merge C == A merge (B merge C)"""
        # Split into A, B, C
        lst = list(items)
        if len(lst) < 3:
            return # Skip if too small
            
        part1 = len(lst) // 3
        part2 = 2 * len(lst) // 3
        A = lst[:part1]
        B = lst[part1:part2]
        C = lst[part2:]
        
        reg = global_registry
        
        # (A merge B) merge C
        s1 = AppendOnlyStore(str(tmp_path / "s1.db"), reg, "main")
        merge_stores(s1, A)
        merge_stores(s1, B)
        merge_stores(s1, C)
        fold1 = deterministic_fold(s1.get_all())
        
        # A merge (B merge C)
        s2 = AppendOnlyStore(str(tmp_path / "s2.db"), reg, "main")
        merge_stores(s2, B)
        merge_stores(s2, C)
        merge_stores(s2, A)
        fold2 = deterministic_fold(s2.get_all())
        
        assert [i.id for i in fold1] == [i.id for i in fold2]

    @settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    @given(item_sequences())
    def test_k_way_partition(self, tmp_path, items):
        """Random k-way partitions merged in arbitrary order must yield identical fold."""
        lst = list(items)
        if not lst:
            return
            
        k = random.randint(2, max(3, min(len(lst), 10)))
        random.shuffle(lst)
        
        partitions = []
        chunk_size = max(1, len(lst) // k)
        for i in range(0, len(lst), chunk_size):
            partitions.append(lst[i:i+chunk_size])
            
        reg = global_registry
        
        # Method 1: Ordered
        s1 = AppendOnlyStore(str(tmp_path / "k1.db"), reg, "main")
        for p in partitions:
            merge_stores(s1, p)
        fold1 = deterministic_fold(s1.get_all())
        
        # Method 2: Reverse ordered
        s2 = AppendOnlyStore(str(tmp_path / "k2.db"), reg, "main")
        for p in reversed(partitions):
            merge_stores(s2, p)
        fold2 = deterministic_fold(s2.get_all())
        
        assert [i.id for i in fold1] == [i.id for i in fold2]


# -----------------------------------------------------------------------
# verify_store
# -----------------------------------------------------------------------

class TestVerifyStore:
    def test_verify_store_success(self, tmp_path, registry, node_a):
        s = AppendOnlyStore(str(tmp_path / "v1.db"), registry, node_a.node_id)
        i1 = create_item(node_a, "", id="1")
        i2 = create_item(node_a, i1.hash, id="2")
        s.append(i1)
        s.append(i2)
        assert s.verify_store() is True

    def test_verify_store_detects_tamper(self, tmp_path, registry, node_a):
        db_path = tmp_path / "v2.db"
        s = AppendOnlyStore(str(db_path), registry, node_a.node_id)
        i1 = create_item(node_a, "", id="1")
        s.append(i1)
        s.close()
        
        # Tamper directly by opening SQLite, dropping trigger, mutating, and recreating trigger
        conn = sqlite3.connect(db_path)
        conn.execute("DROP TRIGGER prevent_update")
        # Change payload JSON but keep hash/sig the same
        conn.execute("UPDATE evidence SET item_json = replace(item_json, '\"weight\":1.0', '\"weight\":0.5') WHERE id = '1'")
        conn.commit()
        conn.close()
        
        s2 = AppendOnlyStore(str(db_path), registry, node_a.node_id)
        with pytest.raises(ValueError, match="Tamper detected: Hash mismatch"):
            s2.verify_store()


# -----------------------------------------------------------------------
# Supersedes edge cases
# -----------------------------------------------------------------------

class TestSupersedesEdgeCases:
    
    def test_superseder_arrives_first(self, node_a):
        # Even if the correction arrives before the original item (via sync),
        # deterministic_fold should handle it by filtering out the original ID.
        i1 = create_item(node_a, "", id="ev-orig", t=100.0)
        i2 = create_item(node_a, i1.hash, id="ev-corr", t=150.0, supersedes="ev-orig")
        
        folded = deterministic_fold([i2, i1])  # i2 arrives first
        assert [i.id for i in folded] == ["ev-corr"]
        
    def test_chains_of_corrections(self, node_a):
        # A <- B <- C
        iA = create_item(node_a, "", id="A", t=100.0)
        iB = create_item(node_a, iA.hash, id="B", t=110.0, supersedes="A")
        iC = create_item(node_a, iB.hash, id="C", t=120.0, supersedes="B")
        
        folded = deterministic_fold([iA, iB, iC])
        # A and B are superseded
        assert [i.id for i in folded] == ["C"]
        
    def test_concurrent_supersedes_tie_break(self, node_a):
        # A is superseded by both B and C (e.g., partitioned nodes correct at same time)
        iA = create_item(node_a, "", id="A", t=100.0)
        iB = create_item(node_a, iA.hash, id="B", t=110.0, supersedes="A")
        iC = create_item(node_a, iB.hash, id="C", t=110.0, supersedes="A") # same timestamp!
        
        # A is superseded. B and C remain active.
        # Deterministic fold will sort by (t, id), so B then C.
        folded = deterministic_fold([iA, iB, iC])
        assert [i.id for i in folded] == ["B", "C"]


# -----------------------------------------------------------------------
# Detailed Performance Test
# -----------------------------------------------------------------------

class TestPerformanceDetailed:
    def test_10k_detailed(self, tmp_path, registry, node_a):
        db_path = tmp_path / "perf_detailed.db"
        store = AppendOnlyStore(str(db_path), registry, node_a.node_id)
        
        items = []
        prev_hash = ""
        for i in range(10000):
            item = create_item(node_a, prev_hash, id=f"ev-{i}", t=float(i))
            prev_hash = item.hash
            items.append(item)
            
        for item in items:
            store.append(item)
            
        store.close()
        
        # Test 1: DB Load
        start = time.time()
        store2 = AppendOnlyStore(str(db_path), registry, node_a.node_id)
        loaded = store2.get_all()
        t_load = time.time() - start
        
        # Test 2: Signature Verify (verify_store does hash, sig, chain)
        start = time.time()
        store2.verify_store()
        t_verify = time.time() - start
        
        # Test 3: Sort/Fold
        start = time.time()
        folded = deterministic_fold(loaded)
        t_fold = time.time() - start
        
        # Write to file
        with open("reports/perf_stage1_detailed.txt", "w") as f:
            f.write(f"DB Load 10k items: {t_load:.3f}s\n")
            f.write(f"Verify Store (Hash+Sig) 10k items: {t_verify:.3f}s\n")
            f.write(f"Sort/Fold 10k items: {t_fold:.3f}s\n")
