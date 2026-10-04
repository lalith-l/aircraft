"""
Simulator Observations — Sensor readings, maintenance records, teardown grades,
spoofed alerts, and partitioned views.

Status: [SIMULATED]

This module sits on top of generator.py and transforms ground-truth states
into noisy, incomplete, or deliberately corrupted observations that the
nirnay system will ingest.  Everything here is inside simulator/ and never
imported by nirnay/.
"""

import numpy as np
import random as pyrandom
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any


# ---------------------------------------------------------------------------
# Sensor model
# ---------------------------------------------------------------------------

@dataclass
class SensorReading:
    """A single time-stamped sensor observation for a part."""
    t: float
    part_sn: str
    channel: str            # e.g. "vibration", "temperature", "pressure"
    value: float
    sensor_id: str = ""
    fault_mode: str = ""    # "", "bias", "stuck", "drift"


class SensorModel:
    """
    Generates synthetic degradation curves from true health state.

    Each channel has:
      - a baseline value for health-state 1
      - a per-state offset
      - additive Gaussian noise
      - optional fault injection (bias, stuck, drift)
    """

    # Channel definitions: (baseline, state_offsets[1..4], noise_std)
    CHANNELS = {
        "vibration":   (0.2, [0.0, 0.3, 0.8, 2.0], 0.05),
        "temperature": (85.0, [0.0, 5.0, 15.0, 40.0], 2.0),
        "pressure":    (3000.0, [0.0, -50.0, -200.0, -500.0], 10.0),
    }

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        self._stuck_values: Dict[str, float] = {}   # sensor_id -> stuck value
        self._bias_values: Dict[str, float] = {}
        self._drift_rates: Dict[str, float] = {}

    def inject_fault(self, sensor_id: str, fault_mode: str):
        """Pre-program a sensor fault."""
        if fault_mode == "stuck":
            self._stuck_values[sensor_id] = None  # will latch on first call
        elif fault_mode == "bias":
            self._bias_values[sensor_id] = self.rng.uniform(-10, 10)
        elif fault_mode == "drift":
            self._drift_rates[sensor_id] = self.rng.uniform(0.001, 0.01)

    def read(self, t: float, part_sn: str, true_state: int,
             channel: str = "vibration", sensor_id: str = "") -> SensorReading:
        """Generate a single noisy reading."""
        baseline, offsets, noise_std = self.CHANNELS[channel]
        true_val = baseline + offsets[true_state - 1]
        noisy = true_val + self.rng.normal(0, noise_std)

        fault_mode = ""

        # Apply faults
        if sensor_id in self._stuck_values:
            fault_mode = "stuck"
            if self._stuck_values[sensor_id] is None:
                self._stuck_values[sensor_id] = noisy
            noisy = self._stuck_values[sensor_id]
        elif sensor_id in self._bias_values:
            fault_mode = "bias"
            noisy += self._bias_values[sensor_id]
        elif sensor_id in self._drift_rates:
            fault_mode = "drift"
            noisy += self._drift_rates[sensor_id] * t

        return SensorReading(
            t=t, part_sn=part_sn, channel=channel,
            value=round(noisy, 4), sensor_id=sensor_id, fault_mode=fault_mode
        )

    def read_all_channels(self, t: float, part_sn: str, true_state: int,
                          sensor_prefix: str = "") -> List[SensorReading]:
        """Read every channel for a part at time t."""
        readings = []
        for ch in self.CHANNELS:
            sid = f"{sensor_prefix}_{ch}" if sensor_prefix else ch
            readings.append(self.read(t, part_sn, true_state, ch, sid))
        return readings


# ---------------------------------------------------------------------------
# Maintenance records  (synthetic text debriefs)
# ---------------------------------------------------------------------------

# Template pool for generating synthetic maintenance text
_DEBRIEF_TEMPLATES = [
    "Pilot rpt {symptom} during {mission}. {part_name} SN {sn} inspected at {base}.",
    "{part_name} (SN:{sn}) removed at {base} after {symptom}. Sent to depot.",
    "Sched insp: {part_name} SN {sn}, {base}. Condition: {condition}.",
    "UNS removal {part_name} SN {sn}. Symptom: {symptom}. Action: {action}.",
]

_SYMPTOMS = [
    "excessive vibration", "high temp indication", "oil leak",
    "intermittent fault code", "performance drop", "unusual noise",
    "no fault found on test bench",
]

_ACTIONS = ["R&R", "bench test", "sent to depot", "cannib from J-XX", "deferred"]

_CONDITIONS = ["serviceable", "degraded", "unserviceable", "BER"]


@dataclass
class MaintenanceRecord:
    t: float
    part_sn: str
    part_name: str
    jet_id: str
    base: str
    text: str
    is_corrupted: bool = False
    corruption_type: str = ""   # "wrong_sn", "copy_paste", "abbrev"


class RecordGenerator:
    """
    Generate synthetic maintenance text records.
    Optionally inject corruptions: wrong serial number, copy-paste duplicates,
    heavy abbreviations.
    """

    def __init__(self, rng: pyrandom.Random, corruption_rate: float = 0.10):
        self.rng = rng
        self.corruption_rate = corruption_rate

    def generate(self, t: float, part_sn: str, part_name: str,
                 jet_id: str, base: str, mission: str = "CAS") -> MaintenanceRecord:
        tmpl = self.rng.choice(_DEBRIEF_TEMPLATES)
        symptom = self.rng.choice(_SYMPTOMS)
        action = self.rng.choice(_ACTIONS)
        condition = self.rng.choice(_CONDITIONS)

        text = tmpl.format(
            symptom=symptom, mission=mission, part_name=part_name,
            sn=part_sn, base=base, condition=condition, action=action,
        )

        is_corrupted = False
        corruption_type = ""

        if self.rng.random() < self.corruption_rate:
            corruption_type = self.rng.choice(["wrong_sn", "copy_paste", "abbrev"])
            is_corrupted = True

            if corruption_type == "wrong_sn":
                # Replace the real SN with a plausible but wrong one
                fake_sn = f"SN-{part_name}-{self.rng.randint(9000, 9999)}"
                text = text.replace(part_sn, fake_sn)
            elif corruption_type == "copy_paste":
                # Duplicate the text (a common real-world error)
                text = text + " // " + text
            elif corruption_type == "abbrev":
                # Heavy abbreviation making NLP harder
                for long, short in [("excessive", "exc"), ("vibration", "vib"),
                                    ("temperature", "temp"), ("indication", "ind"),
                                    ("performance", "perf"), ("inspected", "insp'd"),
                                    ("removed", "rem'd"), ("condition", "cond")]:
                    text = text.replace(long, short)

        return MaintenanceRecord(
            t=t, part_sn=part_sn, part_name=part_name,
            jet_id=jet_id, base=base, text=text,
            is_corrupted=is_corrupted, corruption_type=corruption_type,
        )


# ---------------------------------------------------------------------------
# Teardown grading (depot)
# ---------------------------------------------------------------------------

@dataclass
class TeardownResult:
    t: float
    part_sn: str
    true_state: int
    observed_grade: int      # 1..4, corrupted by confusion matrix
    is_nff: bool             # No Fault Found


class TeardownModel:
    """
    Simulates the depot test bench.
    Given a true state, emits an observed grade drawn from the confusion matrix.
    NFF occurs stochastically for state ≤ 2 parts.
    """

    def __init__(self, confusion_matrix: List[List[float]],
                 rng: np.random.RandomState, nff_prob_good: float = 0.30,
                 nff_prob_degraded: float = 0.10):
        self.cm = np.array(confusion_matrix)
        self.rng = rng
        self.nff_prob = {1: nff_prob_good, 2: nff_prob_degraded}

    def grade(self, t: float, part_sn: str, true_state: int) -> TeardownResult:
        row = self.cm[true_state - 1]
        observed = int(self.rng.choice([1, 2, 3, 4], p=row))

        # NFF: intermittent faults on good/degraded parts
        is_nff = False
        if true_state in self.nff_prob:
            if self.rng.random() < self.nff_prob[true_state]:
                is_nff = True
                observed = 1  # test bench says "no fault found"

        return TeardownResult(
            t=t, part_sn=part_sn, true_state=true_state,
            observed_grade=observed, is_nff=is_nff,
        )


# ---------------------------------------------------------------------------
# Spoofed alerts
# ---------------------------------------------------------------------------

@dataclass
class SpoofedAlert:
    t: float
    part_sn: str
    alert_type: str     # "phantom", "suppressed"
    details: Dict[str, Any] = field(default_factory=dict)


class AlertSpoofer:
    """
    Injects phantom alerts (false positives) or suppresses real alerts
    (false negatives) to stress the detection module.
    """

    def __init__(self, rng: pyrandom.Random,
                 phantom_rate: float = 0.05, suppression_rate: float = 0.03):
        self.rng = rng
        self.phantom_rate = phantom_rate
        self.suppression_rate = suppression_rate

    def maybe_phantom(self, t: float, part_sn: str) -> Optional[SpoofedAlert]:
        """Randomly inject a phantom alert."""
        if self.rng.random() < self.phantom_rate:
            return SpoofedAlert(
                t=t, part_sn=part_sn, alert_type="phantom",
                details={"channel": self.rng.choice(["vibration", "temperature", "pressure"]),
                          "fake_value": round(self.rng.uniform(0, 100), 2)},
            )
        return None

    def should_suppress(self, t: float, part_sn: str) -> bool:
        """Decide whether to suppress a real alert."""
        return self.rng.random() < self.suppression_rate


# ---------------------------------------------------------------------------
# Confounding by indication
# ---------------------------------------------------------------------------

class ConfoundingModel:
    """
    Models confounding by indication: parts that *appear* sicker get more
    aggressive maintenance, which makes them *appear* to have better outcomes,
    biasing naïve estimates.

    Given a set of alerts, returns a priority-override that fast-tracks
    sicker-looking parts, potentially hiding the causal effect of maintenance.
    """

    def __init__(self, rng: pyrandom.Random, confound_strength: float = 0.5):
        self.rng = rng
        self.confound_strength = confound_strength

    def priority_boost(self, part_sn: str, alert_count: int) -> float:
        """
        Returns a priority multiplier.  Higher alert counts → more aggressive
        maintenance → the part gets fixed faster, confounding survival analysis.
        """
        if alert_count == 0:
            return 1.0
        boost = 1.0 + self.confound_strength * np.log1p(alert_count)
        return round(float(boost), 3)


# ---------------------------------------------------------------------------
# Data partitioning
# ---------------------------------------------------------------------------

def partition_data(events: List[dict], gt: List[dict],
                   train_frac: float = 0.7, seed: int = 42
                   ) -> Dict[str, Dict[str, list]]:
    """
    Deterministic partition of events and ground truth into train / eval / test
    splits by unique part serial numbers.

    Ground truth is *only* available for train (the simulator referee keeps
    eval/test ground truth sealed).
    """
    rng = pyrandom.Random(seed)

    # Collect unique part SNs from ground truth
    all_sns = sorted({g["sn"] for g in gt})
    rng.shuffle(all_sns)

    n_train = int(len(all_sns) * train_frac)
    n_eval = int(len(all_sns) * (1 - train_frac) / 2)

    train_sns = set(all_sns[:n_train])
    eval_sns = set(all_sns[n_train:n_train + n_eval])
    test_sns = set(all_sns[n_train + n_eval:])

    def filt(lst, key, sn_set):
        return [e for e in lst if e.get(key, e.get("sn", "")) in sn_set]

    return {
        "train": {"events": filt(events, "sn", train_sns),
                  "ground_truth": filt(gt, "sn", train_sns)},
        "eval":  {"events": filt(events, "sn", eval_sns),
                  "ground_truth": []},   # sealed
        "test":  {"events": filt(events, "sn", test_sns),
                  "ground_truth": []},   # sealed
    }


# ---------------------------------------------------------------------------
# Twin-world runner
# ---------------------------------------------------------------------------

def run_twin_worlds(config, seed: int, apply_alerts_fn=None):
    """
    Run two copies of the simulation from the same seed.
    World A applies alerts (interventions); World B ignores them.
    Returns (events_A, gt_A, events_B, gt_B).

    ``apply_alerts_fn`` is a callable(sim, events, t) that modifies world A.
    If None, both worlds are identical (baseline for testing).
    """
    from simulator.generator import Simulator

    sim_a = Simulator(config, seed, world="A")
    sim_b = Simulator(config, seed, world="B")

    for day in range(config.horizon_days):
        t = float(day * 24)
        sim_a.current_t = t
        sim_b.current_t = t
        sim_a.step_day()
        sim_b.step_day()

        # In world A, apply the intervention function
        if apply_alerts_fn is not None:
            apply_alerts_fn(sim_a, sim_a.events, t)

    return sim_a.events, sim_a.ground_truth, sim_b.events, sim_b.ground_truth


# ---------------------------------------------------------------------------
# Second generator family  (alternative causal structure for cross-validation)
# ---------------------------------------------------------------------------

class AltGeneratorFamily:
    """
    A second, structurally different generator for cross-family validation.

    Differences from the primary generator:
      - Damage model uses gamma increments instead of Weibull-proxy.
      - Parts have a shared frailty term per jet (some jets are just harder
        on their parts due to pilots / mission assignment).
      - No lot effect; instead the confounder is jet-level frailty.

    This ensures that any pattern the belief engine discovers is not an
    artifact of one specific generative model.
    """

    def __init__(self, config, seed: int):
        self.config = config
        self.rng = np.random.RandomState(seed)
        self.py_rng = pyrandom.Random(seed)

        # Jet-level frailty (gamma-distributed)
        self.jet_frailty = {
            f"J-{i:02d}": float(self.rng.gamma(2.0, 0.5))
            for i in range(config.n_jets)
        }
        # Parts: same structure, different damage model
        self.part_health: Dict[str, float] = {}
        self.part_states: Dict[str, int] = {}
        self.jets: Dict[str, Dict] = {}
        self._init_fleet()

    def _init_fleet(self):
        self.jets = {
            f"J-{i:02d}": {"base": f"Base-{i % self.config.n_bases}", "parts": {}}
            for i in range(self.config.n_jets)
        }
        ctr = 0
        for jid, jet in self.jets.items():
            for pcfg in self.config.parts:
                sn = f"ALT-{pcfg.name}-{ctr}"
                ctr += 1
                self.part_health[sn] = 0.0
                self.part_states[sn] = 1
                jet["parts"][pcfg.name] = sn

    def run(self):
        events = []
        ground_truth = []

        for day in range(self.config.horizon_days):
            t = float(day * 24)
            for jid, jet in self.jets.items():
                frailty = self.jet_frailty[jid]
                flight_hours = float(self.rng.uniform(1.5, 4.0))

                events.append({
                    "t": t, "type": "sortie", "jet": jid,
                    "part": "", "sn": "",
                    "details": {"hours": flight_hours, "family": "alt"},
                })

                for pname, sn in jet["parts"].items():
                    pcfg = next(p for p in self.config.parts if p.name == pname)
                    # Gamma increment scaled by frailty
                    damage = float(self.rng.gamma(
                        shape=pcfg.weibull_shape,
                        scale=flight_hours * frailty / pcfg.weibull_scale,
                    ))
                    self.part_health[sn] += damage
                    ratio = self.part_health[sn]
                    old_state = self.part_states[sn]

                    if ratio > 1.0:
                        state = 4
                    elif ratio > 0.7:
                        state = 3
                    elif ratio > 0.4:
                        state = 2
                    else:
                        state = 1

                    if state != old_state:
                        self.part_states[sn] = state
                        ground_truth.append({
                            "sn": sn, "t": t,
                            "true_state": state, "jet_frailty": round(frailty, 4),
                        })

        events_out = events
        gt_out = ground_truth
        return events_out, gt_out
