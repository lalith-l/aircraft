"""
Simulator-to-Evidence Adapter

This module sits outside both the `simulator` and `nirnay` packages.
It reads the simulator's event stream (but never the ground truth stream)
and translates those events into cryptographically signed `EvidenceItem`
objects, which it then appends to the `AppendOnlyStore`.

This is built in Stage 4 (Detection) when we start translating simulation
output into belief-engine input.
"""

def dummy_adapter():
    pass
