import sys
from nirnay.contracts import EvidenceItem, EvidenceSource

item = EvidenceItem(
    id="cross-hash-1", part_sn="SN-HASH", source=EvidenceSource.SENSOR,
    node="node-X", t=123456789.0, t_logged=123456790.0,
    payload={"foo": "bar", "val": 1.23}, weight=1.0, prev_hash="abc"
)
print(item.compute_hash())
