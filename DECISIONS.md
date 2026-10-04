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

## Stage 2: Simulator
- **Separate Observations Module**: Sensor readings, maintenance records, teardown grades, and alert spoofing live in `simulator/observations.py`, cleanly separated from the ground-truth physics in `generator.py`. This makes it easy to swap observation noise profiles without touching the causal model.
- **Sensor Fault Injection**: Three fault modes (stuck, bias, drift) are pre-programmed per sensor ID so the detection module can be tested against realistic failure patterns.
- **Teardown Confusion Matrix + NFF**: The depot test bench applies a configurable confusion matrix (P(grade | true_state)) and injects No-Fault-Found stochastically for good/degraded parts. This replicates a major real-world source of label noise.
- **Record Corruption by Design**: Maintenance text records are corrupted at a configurable rate with wrong serial numbers, copy-paste duplicates, and heavy abbreviations, ensuring the NLP pipeline is stress-tested.
- **Confounding by Indication**: Sicker-looking parts get priority maintenance, which paradoxically makes them appear healthier in survival analysis. This confounder is deliberately injected so the belief engine can be tested for robustness.
- **Twin-World Mode**: Two simulation copies share the same seed but diverge when `apply_alerts_fn` modifies world A. World B serves as the untreated control. This enables causal effect estimation (Experiment G).
- **Alt Generator Family**: A second generator uses gamma-increment damage with jet-level frailty (no lot effect) to provide cross-family validation. Any pattern the belief engine finds must survive testing against both families.
- **Partitioning**: Data is split by part serial number (not by time) into train/eval/test. Eval and test ground truth are sealed—the simulator referee never reveals them to the nirnay system.
