"""
Simulator-to-Evidence Adapter

Sits OUTSIDE both simulator/ and nirnay/.
Reads only the event stream (never ground truth).
Translates simulator events into EvidenceItem objects
and appends them to the AppendOnlyStore.

Status: [REAL]
"""

from typing import List, Dict
from nirnay.contracts import EvidenceItem, EvidenceSource


def events_to_evidence(events: List[dict], node_id: str = "sim_adapter"
                       ) -> List[EvidenceItem]:
    """
    Convert simulator event dicts into EvidenceItem objects.
    
    This function sees ONLY the event stream — never ground truth.
    It constructs per_state likelihoods from observable signals only.
    """
    items = []
    prev_hash = ""

    for event in events:
        if event["type"] == "sortie":
            # Sorties generate exposure evidence
            item = EvidenceItem(
                part_sn=event.get("sn", ""),
                source=EvidenceSource.SENSOR,
                node=node_id,
                t=event["t"],
                t_logged=event["t"],
                payload={
                    "event_type": "sortie",
                    "jet": event.get("jet", ""),
                    "hours": event.get("details", {}).get("hours", 0),
                    "regime": event.get("details", {}).get("regime", "normal"),
                },
                # Uniform likelihood — a sortie by itself doesn't tell us health
                likelihood={"per_state": [0.2, 0.2, 0.2, 0.2, 0.2]},
                weight=0.3,
                prev_hash=prev_hash,
            )
            item = item.with_hash()
            prev_hash = item.hash
            items.append(item)

        elif event["type"] == "failure_abort":
            # Strong evidence of state 4
            item = EvidenceItem(
                part_sn=event.get("sn", ""),
                source=EvidenceSource.SENSOR,
                node=node_id,
                t=event["t"],
                t_logged=event["t"],
                payload={
                    "event_type": "failure_abort",
                    "jet": event.get("jet", ""),
                    "part": event.get("part", ""),
                },
                likelihood={"per_state": [0.01, 0.02, 0.07, 0.20, 0.70]},
                weight=0.9,
                prev_hash=prev_hash,
            )
            item = item.with_hash()
            prev_hash = item.hash
            items.append(item)

        elif event["type"] == "maintenance":
            action = event.get("details", {}).get("action", "")
            item = EvidenceItem(
                part_sn=event.get("sn", ""),
                source=EvidenceSource.RECORD,
                node=node_id,
                t=event["t"],
                t_logged=event["t"],
                payload={
                    "event_type": "maintenance",
                    "jet": event.get("jet", ""),
                    "part": event.get("part", ""),
                    "action": action,
                },
                # Maintenance tells us the part was bad enough to remove
                likelihood={"per_state": [0.05, 0.10, 0.25, 0.35, 0.25]},
                weight=0.7,
                prev_hash=prev_hash,
            )
            item = item.with_hash()
            prev_hash = item.hash
            items.append(item)

    return items


def sensor_readings_to_evidence(readings: list, node_id: str = "sensor_adapter"
                                 ) -> List[EvidenceItem]:
    """
    Convert SensorReading objects (from observations) into EvidenceItems.
    Maps sensor values to per-state likelihoods using channel-specific thresholds.
    """
    from simulator.observations import SensorModel

    items = []
    prev_hash = ""

    for r in readings:
        # Convert sensor value to a likelihood vector based on channel
        ch_info = SensorModel.CHANNELS.get(r.channel, None)
        if ch_info is None:
            continue

        baseline, offsets, noise_std = ch_info
        # Compute likelihood for each state using Gaussian emission
        per_state = []
        for state_idx in range(5):
            mu = baseline + offsets[min(state_idx, len(offsets) - 1)]
            # Gaussian likelihood
            diff = r.value - mu
            ll = float(
                (1.0 / (noise_std * 2.507)) *  # sqrt(2*pi) ~ 2.507
                __import__('math').exp(-0.5 * (diff / noise_std) ** 2)
            )
            per_state.append(round(max(ll, 1e-12), 8))

        item = EvidenceItem(
            part_sn=r.part_sn,
            source=EvidenceSource.SENSOR,
            node=node_id,
            t=r.t,
            t_logged=r.t,
            payload={
                "channel": r.channel,
                "value": r.value,
                "sensor_id": r.sensor_id,
            },
            likelihood={"per_state": per_state},
            weight=0.8 if r.fault_mode == "" else 0.3,
            prev_hash=prev_hash,
        )
        item = item.with_hash()
        prev_hash = item.hash
        items.append(item)

    return items
