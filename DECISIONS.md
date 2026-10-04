# Nirnay — Architecture and Design Decisions

This document records the architectural choices made for the Nirnay prototype and the rationale behind them.

## General
- **Python 3.11 with Virtual Environment**: Selected for standard modern Python features while ensuring full reproducibility and isolation.
- **Strict Separation of Simulator**: `simulator/` acts as an independent referee. `nirnay/` must never import it. This guarantees we don't accidentally leak ground truth into the belief system.
- **CI is `make test`**: The primary gate for all changes is running the full test suite locally via `make test`.

## Stage 0: Contracts
- **Frozen Dataclasses for Contracts**: `EvidenceItem`, `Passport`, etc. are frozen to prevent accidental mutation of evidence.
- **Deep-Immutability for Collections**: Lists and dicts (payload, likelihood) in `EvidenceItem` are deep-copied in `__post_init__` so external changes to the original dicts don't affect the item.
- **Deterministic Hashing**: JSON serialisation for hashing (`canonical_dict`) normalizes Enums to strings and sorts keys, preventing hash divergence across different Python versions or execution runs. Float formatting and NaN/Inf are also restricted for stability.

## Stage 1: Evidence Store and Merge
- **SQLite Append-Only Table**: Evidence is stored in a single table. To guarantee the append-only nature at the database level, we use SQLite `CREATE TRIGGER` to block `UPDATE` and `DELETE` operations.
- **Ed25519 Signatures & Hash Chains**: Every item is cryptographically signed and hash-chained to the previous item *from the same node*. This allows per-node auditing and makes tamper-detection trivial.
- **Public-Key Registry**: A simple dictionary/registry mapping node IDs to their public keys, allowing nodes to verify signatures of incoming items.
- **Merge as Set Union**: When two nodes sync, merge is a pure set-union operation based on `(source, id)`. This provides mathematically guaranteed commutative, associative, and idempotent properties, validated by Hypothesis property tests.
- **Deterministic Ordered Fold**: To reconstruct a belief, the entire set of evidence for a part is folded in order of `(t, id)`. If an item arrives late (e.g. from an offline node), the final set is identical, so the resulting fold is identical.
- **Supersedes Handling**: Corrections never overwrite data (due to append-only rule). Instead, a new item points to the old item via `supersedes`. The fold logic skips superseded items.
- **Event-Time Clock-Skew Risk**: Sorting by `t` (event time) is mathematically deterministic, but if nodes have skewed clocks, evidence might be interleaved in a non-causal physical order. The tie-breaker by `id` handles identical timestamps. In a production system, bounded clock-sync or Lamport timestamps would be required, but for this prototype, UTC epoch seconds are used.
