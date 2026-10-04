"""
Simulator Observations — Sensor readings, maintenance records, teardown grades,
spoofed alerts, partitioned views, referee scoring, twin-world, and alt family.

Status: [SIMULATED]

This module sits on top of generator.py and transforms ground-truth states
into noisy, incomplete, or deliberately corrupted observations that the
nirnay system will ingest.  Everything here is inside simulator/ and never
imported by nirnay/.
"""

import math
import numpy as np
import random as pyrandom
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any


# ===========================================================================
# Seeded Random Stream Factory  (Fix #5)
# ===========================================================================

class RandomStreamFactory:
    """
    Creates independent, deterministic random streams per (entity, purpose).
    Each stream is derived from a master seed so that adding/removing a stream
    for one entity does not affect any other entity's stream.
    """

    def __init__(self, master_seed: int):
        self.master_seed = master_seed

    def _derive_seed(self, entity: str, purpose: str) -> int:
        """Derive a deterministic sub-seed from entity + purpose string."""
        import hashlib
        h = hashlib.sha256(f"{self.master_seed}:{entity}:{purpose}".encode()).digest()
        return int.from_bytes(h[:4], "big")

    def numpy_rng(self, entity: str, purpose: str) -> np.random.RandomState:
        return np.random.RandomState(self._derive_seed(entity, purpose))

    def python_rng(self, entity: str, purpose: str) -> pyrandom.Random:
        return pyrandom.Random(self._derive_seed(entity, purpose))


# ===========================================================================
# Sensor model
# ===========================================================================

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
        self._stuck_values: Dict[str, float] = {}
        self._bias_values: Dict[str, float] = {}
        self._drift_rates: Dict[str, float] = {}

    def inject_fault(self, sensor_id: str, fault_mode: str):
        """Pre-program a sensor fault."""
        if fault_mode == "stuck":
            self._stuck_values[sensor_id] = None
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
        readings = []
        for ch in self.CHANNELS:
            sid = f"{sensor_prefix}_{ch}" if sensor_prefix else ch
            readings.append(self.read(t, part_sn, true_state, ch, sid))
        return readings


# ===========================================================================
# Maintenance records  (Fix #7: two template families, late entries,
# Hinglish, not-reproduced/contradictory closures, per-question GT labels)
# ===========================================================================

# FAMILY A — English-heavy, formal
_FAMILY_A_TEMPLATES = [
    "Pilot rpt {symptom} during {mission}. {part_name} SN {sn} inspected at {base}.",
    "{part_name} (SN:{sn}) removed at {base} after {symptom}. Sent to depot.",
    "Sched insp: {part_name} SN {sn}, {base}. Condition: {condition}.",
    "UNS removal {part_name} SN {sn}. Symptom: {symptom}. Action: {action}.",
    "Defect rectification: {part_name} SN {sn} replaced due to {symptom}. Aircraft {jet_id} RTB.",
]

# FAMILY B — Mixed Hinglish, abbreviated, realistic IAF-style
_FAMILY_B_TEMPLATES = [
    "{part_name} SN {sn} mein problem aaya during {mission}. {symptom}. {base} pe dekha.",
    "Snag: {part_name} ({sn}) - {symptom}. Aircft {jet_id} grounded. Action: {action}.",
    "{jet_id} ka {part_name} SN {sn} kharab ho gaya. {symptom} report kiya. {base}.",
    "Tech entry: {part_name} SN {sn} snag rectified at {base}. Recheck baad mein karega.",
    "AOG {jet_id}: {part_name} ({sn}) fail. {symptom}. Depot bhejne ka order aaya.",
]

_SYMPTOMS = [
    "excessive vibration", "high temp indication", "oil leak",
    "intermittent fault code", "performance drop", "unusual noise",
    "no fault found on test bench", "erratic readings on gauge",
    "smoke/fumes in cockpit area", "abnormal pressure fluctuation",
]

_ACTIONS = ["R&R", "bench test", "sent to depot", "cannib from J-XX",
            "deferred", "component swapped", "BITE test cleared"]

_CONDITIONS = ["serviceable", "degraded", "unserviceable", "BER", "suspect"]

# Closure annotations
_CLOSURE_TYPES = [
    "normal",           # standard closure
    "not_reproduced",   # fault could not be reproduced on ground
    "contradictory",    # closure contradicts the symptom
]


@dataclass
class MaintenanceRecord:
    t: float
    t_actual: float         # when the event actually happened (for late entries)
    part_sn: str
    part_name: str
    jet_id: str
    base: str
    text: str
    template_family: str    # "A" or "B"
    is_corrupted: bool = False
    corruption_type: str = ""
    is_late_entry: bool = False
    closure_type: str = "normal"
    # Per-question ground-truth labels for Laya registry
    gt_labels: Dict[str, Any] = field(default_factory=dict)


class RecordGenerator:
    """
    Generate synthetic maintenance text records with two disjoint template
    families, late entries, Hinglish tokens, closure variants, and per-record
    ground-truth labels for the Laya question registry.
    """

    def __init__(self, rng: pyrandom.Random, corruption_rate: float = 0.10,
                 late_entry_rate: float = 0.08, family: str = "mixed"):
        self.rng = rng
        self.corruption_rate = corruption_rate
        self.late_entry_rate = late_entry_rate
        self.family = family  # "A", "B", or "mixed"

    def generate(self, t: float, part_sn: str, part_name: str,
                 jet_id: str, base: str, true_state: int = 0,
                 mission: str = "CAS") -> MaintenanceRecord:

        # Choose template family
        if self.family == "mixed":
            fam = self.rng.choice(["A", "B"])
        else:
            fam = self.family
        templates = _FAMILY_A_TEMPLATES if fam == "A" else _FAMILY_B_TEMPLATES

        tmpl = self.rng.choice(templates)
        symptom = self.rng.choice(_SYMPTOMS)
        action = self.rng.choice(_ACTIONS)
        condition = self.rng.choice(_CONDITIONS)

        text = tmpl.format(
            symptom=symptom, mission=mission, part_name=part_name,
            sn=part_sn, base=base, condition=condition, action=action,
            jet_id=jet_id,
        )

        # Late entry: event happened earlier but recorded now
        is_late = False
        t_actual = t
        if self.rng.random() < self.late_entry_rate:
            is_late = True
            delay_hours = self.rng.uniform(24, 168)  # 1-7 days late
            t_actual = t - delay_hours
            text = f"[LATE ENTRY - event {delay_hours:.0f}h ago] " + text

        closure = self.rng.choices(
            _CLOSURE_TYPES,
            weights=[0.80, 0.12, 0.08], k=1
        )[0]
        if closure == "not_reproduced":
            text += " Closure: fault not reproduced on ground test."
        elif closure == "contradictory":
            text += f" Closure: item found serviceable, cleared for ops."

        # Corruptions
        is_corrupted = False
        corruption_type = ""
        if self.rng.random() < self.corruption_rate:
            corruption_type = self.rng.choice(
                ["wrong_sn", "copy_paste", "abbrev", "hinglish_extra"])
            is_corrupted = True
            if corruption_type == "wrong_sn":
                fake_sn = f"SN-{part_name}-{self.rng.randint(9000, 9999)}"
                text = text.replace(part_sn, fake_sn)
            elif corruption_type == "copy_paste":
                text = text + " // " + text
            elif corruption_type == "abbrev":
                for long, short in [("excessive", "exc"), ("vibration", "vib"),
                                    ("temperature", "temp"), ("indication", "ind"),
                                    ("performance", "perf"), ("inspected", "insp'd"),
                                    ("removed", "rem'd"), ("condition", "cond")]:
                    text = text.replace(long, short)
            elif corruption_type == "hinglish_extra":
                text += " // woh wala part change karna padega, urgent hai."

        # Per-question ground-truth labels for Laya registry
        gt_labels = {
            "part_sn": part_sn,
            "part_name": part_name,
            "jet_id": jet_id,
            "true_state": true_state,
            "symptom_mentioned": symptom,
            "action_taken": action,
            "is_late_entry": is_late,
            "closure_type": closure,
            "is_corrupted": is_corrupted,
            "corruption_type": corruption_type,
            "template_family": fam,
        }

        return MaintenanceRecord(
            t=t, t_actual=t_actual, part_sn=part_sn, part_name=part_name,
            jet_id=jet_id, base=base, text=text,
            template_family=fam,
            is_corrupted=is_corrupted, corruption_type=corruption_type,
            is_late_entry=is_late, closure_type=closure,
            gt_labels=gt_labels,
        )


# ===========================================================================
# Teardown grading  (Fix #3: NFF for states 2 and 3, not state 1)
# ===========================================================================

@dataclass
class TeardownResult:
    t: float
    part_sn: str
    true_state: int
    observed_grade: int
    is_nff: bool


class TeardownModel:
    """
    Simulates the depot test bench.

    NFF (No Fault Found) models intermittent faults.
    - State 1 (Good): no NFF — if a good part reaches the bench, the bench
      correctly says "good".  NFF only arises when a part has a *real* defect
      that the test bench fails to trigger.
    - State 2 (Degraded): NFF rate ~20%.  Intermittent faults are common at
      early degradation; the fault manifests in flight but not on the bench
      because bench conditions differ from flight loads.
    - State 3 (Severe): NFF rate ~8%.  Faults are more persistent but some
      remain load-dependent or thermal-dependent and still escape bench detection.
    - State 4 (Failed): NFF rate 0%.  A genuinely failed part will always
      fail bench testing.
    """

    def __init__(self, confusion_matrix: List[List[float]],
                 rng: np.random.RandomState,
                 nff_prob_degraded: float = 0.20,
                 nff_prob_severe: float = 0.08):
        self.cm = np.array(confusion_matrix)
        self.rng = rng
        # NFF only for intermittent faults: states 2 and 3
        self.nff_prob = {2: nff_prob_degraded, 3: nff_prob_severe}

    def grade(self, t: float, part_sn: str, true_state: int) -> TeardownResult:
        row = self.cm[true_state - 1]
        observed = int(self.rng.choice([1, 2, 3, 4], p=row))

        is_nff = False
        if true_state in self.nff_prob:
            if self.rng.random() < self.nff_prob[true_state]:
                is_nff = True
                observed = 1  # test bench says "no fault found"

        return TeardownResult(
            t=t, part_sn=part_sn, true_state=true_state,
            observed_grade=observed, is_nff=is_nff,
        )


# ===========================================================================
# Spoofed alerts
# ===========================================================================

@dataclass
class SpoofedAlert:
    t: float
    part_sn: str
    alert_type: str
    details: Dict[str, Any] = field(default_factory=dict)


class AlertSpoofer:
    def __init__(self, rng: pyrandom.Random,
                 phantom_rate: float = 0.05, suppression_rate: float = 0.03):
        self.rng = rng
        self.phantom_rate = phantom_rate
        self.suppression_rate = suppression_rate

    def maybe_phantom(self, t: float, part_sn: str) -> Optional[SpoofedAlert]:
        if self.rng.random() < self.phantom_rate:
            return SpoofedAlert(
                t=t, part_sn=part_sn, alert_type="phantom",
                details={"channel": self.rng.choice(["vibration", "temperature", "pressure"]),
                          "fake_value": round(self.rng.uniform(0, 100), 2)},
            )
        return None

    def should_suppress(self, t: float, part_sn: str) -> bool:
        return self.rng.random() < self.suppression_rate


# ===========================================================================
# Confounding by indication
# ===========================================================================

class ConfoundingModel:
    """
    Parts that *appear* sicker get more aggressive maintenance, biasing
    naïve survival estimates.
    """
    def __init__(self, rng: pyrandom.Random, confound_strength: float = 0.5):
        self.rng = rng
        self.confound_strength = confound_strength

    def priority_boost(self, part_sn: str, alert_count: int) -> float:
        if alert_count == 0:
            return 1.0
        boost = 1.0 + self.confound_strength * np.log1p(alert_count)
        return round(float(boost), 3)


# ===========================================================================
# Data partitioning  (Fix #1: whole-fleet seeds + time-based splits)
# ===========================================================================

def partition_by_fleet_and_time(
    fleet_runs: Dict[str, Tuple[List[dict], List[dict]]],
    time_split_frac: float = 0.7,
) -> Dict[str, Dict[str, list]]:
    """
    Partition strategy:
    - Each fleet is a separate seeded simulation run (different seed).
    - Training fleet: first `time_split_frac` of time used for train,
      remainder for within-fleet eval.
    - Eval fleet: entirely held out (no ground truth revealed).
    - Test fleet: entirely held out (no ground truth revealed).

    Args:
        fleet_runs: dict mapping split name to (events, ground_truth).
            Expected keys: "train", "eval", "test".
        time_split_frac: fraction of the time horizon for training within
            the training fleet.

    Returns:
        {"train": {"events": ..., "ground_truth": ...},
         "train_holdout": {"events": ..., "ground_truth": []},
         "eval": {"events": ..., "ground_truth": []},
         "test": {"events": ..., "ground_truth": []}}
    """
    result = {}

    for split_name, (events, gt) in fleet_runs.items():
        if split_name == "train":
            # Time-based split within the training fleet
            if not events:
                result["train"] = {"events": [], "ground_truth": []}
                result["train_holdout"] = {"events": [], "ground_truth": []}
                continue

            all_times = [e["t"] for e in events]
            t_max = max(all_times) if all_times else 0
            t_split = t_max * time_split_frac

            result["train"] = {
                "events": [e for e in events if e["t"] <= t_split],
                "ground_truth": [g for g in gt if g["t"] <= t_split],
            }
            result["train_holdout"] = {
                "events": [e for e in events if e["t"] > t_split],
                "ground_truth": [],  # sealed
            }
        else:
            # Eval and test fleets: entirely held out
            result[split_name] = {
                "events": events,
                "ground_truth": [],  # sealed
            }

    return result


# ===========================================================================
# Referee Scoring Module  (Fix #2)
# ===========================================================================

class Referee:
    """
    Holds sealed ground truth and returns only aggregate metrics.
    Never exposes individual ground-truth records.
    """

    def __init__(self, sealed_gt: List[dict]):
        self._gt = {(g["sn"], g["t"]): g["true_state"] for g in sealed_gt}
        self._gt_list = list(sealed_gt)

    def score_predictions(self, predictions: List[dict]) -> Dict[str, float]:
        """
        Score predictions against sealed truth.

        Each prediction is {"sn": str, "t": float, "predicted_state": int}.
        Returns aggregate metrics only (no per-item truth is revealed).
        """
        matched = 0
        correct = 0
        errors = []

        for pred in predictions:
            key = (pred["sn"], pred["t"])
            if key in self._gt:
                matched += 1
                true = self._gt[key]
                predicted = pred["predicted_state"]
                if true == predicted:
                    correct += 1
                errors.append(abs(true - predicted))

        n = max(matched, 1)
        return {
            "n_predictions": len(predictions),
            "n_matched": matched,
            "accuracy": correct / n,
            "mae": sum(errors) / n if errors else float("nan"),
            "within_1": sum(1 for e in errors if e <= 1) / n if errors else 0.0,
        }

    def score_rul(self, rul_predictions: List[dict],
                  failure_time_col: str = "t_failure") -> Dict[str, float]:
        """
        Score RUL predictions.  Each prediction has sn, t_predicted, predicted_rul.
        We compare against the actual time the part transitioned to state 4.
        """
        # Build failure times from GT
        failure_times = {}
        for g in self._gt_list:
            if g["true_state"] == 4:
                sn = g["sn"]
                if sn not in failure_times or g["t"] < failure_times[sn]:
                    failure_times[sn] = g["t"]

        errors = []
        for pred in rul_predictions:
            sn = pred["sn"]
            if sn in failure_times:
                actual_rul = failure_times[sn] - pred["t_predicted"]
                error = pred["predicted_rul"] - actual_rul
                errors.append(error)

        n = max(len(errors), 1)
        return {
            "n_scored": len(errors),
            "mean_error": sum(errors) / n if errors else float("nan"),
            "rmse": (sum(e**2 for e in errors) / n)**0.5 if errors else float("nan"),
        }


# ===========================================================================
# Twin-world runner  (Fix #5: independent streams, no-op safety)
# ===========================================================================

def run_twin_worlds(config, seed: int, apply_alerts_fn=None):
    """
    Run two copies of the simulation from the same seed.
    Uses independent per-entity random streams so that a no-op intervention
    that consumes randomness in world A does not desynchronise world B.
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

        if apply_alerts_fn is not None:
            apply_alerts_fn(sim_a, sim_a.events, t)

    return sim_a.events, sim_a.ground_truth, sim_b.events, sim_b.ground_truth


# ===========================================================================
# Second generator family  (Fix #4: continuous latent, heavy-tailed noise,
# regime-dependent sensor gain, correlated channels)
# ===========================================================================

class AltSensorModel:
    """
    Alt family observation model with:
    - Continuous latent degradation (not discrete states)
    - Heavy-tailed Student-t noise
    - Regime-dependent sensor gain
    - Correlated channels (shared latent factor)
    """

    CHANNELS = ["vibration", "temperature", "pressure"]
    # Per-channel: (baseline, degradation_sensitivity, noise_scale)
    PARAMS = {
        "vibration":   (0.2,  1.5,  0.1),
        "temperature": (85.0, 12.0, 3.0),
        "pressure":    (3000.0, -300.0, 15.0),
    }
    REGIME_GAIN = {"normal": 1.0, "hot_dusty": 1.3, "surge_high_g": 1.8}
    DF = 3  # degrees of freedom for Student-t (heavy tails)

    def __init__(self, rng: np.random.RandomState):
        self.rng = rng
        # Correlation structure: shared latent factor loading
        self._corr_loading = np.array([0.7, 0.5, 0.3])  # per channel

    def read_all(self, t: float, part_sn: str, continuous_health: float,
                 regime: str = "normal") -> List[SensorReading]:
        """
        continuous_health: 0.0 (new) to 1.0+ (failed), a continuous latent.
        """
        gain = self.REGIME_GAIN.get(regime, 1.0)
        # Shared latent factor for correlated noise
        shared_factor = float(self.rng.standard_t(self.DF))

        readings = []
        for i, ch in enumerate(self.CHANNELS):
            base, sensitivity, noise_scale = self.PARAMS[ch]
            signal = base + sensitivity * continuous_health * gain
            # Correlated heavy-tailed noise
            idiosyncratic = float(self.rng.standard_t(self.DF)) * noise_scale
            correlated = shared_factor * self._corr_loading[i] * noise_scale
            noisy = signal + idiosyncratic + correlated
            readings.append(SensorReading(
                t=t, part_sn=part_sn, channel=ch,
                value=round(noisy, 4), sensor_id=f"alt_{ch}",
            ))
        return readings


class AltGeneratorFamily:
    """
    A second, structurally different generator for cross-family validation.

    Differences from the primary generator:
      - Continuous latent degradation (gamma increments), states derived
        by discretising the latent.
      - Jet-level gamma frailty instead of lot effect.
      - AltSensorModel with heavy-tailed noise, regime-dependent gain,
        and correlated channels.
    """

    def __init__(self, config, seed: int):
        self.config = config
        self.stream_factory = RandomStreamFactory(seed)
        self.rng = self.stream_factory.numpy_rng("global", "damage")
        self.py_rng = self.stream_factory.python_rng("global", "logic")
        self.sensor_rng = self.stream_factory.numpy_rng("global", "sensors")
        self.alt_sensor = AltSensorModel(self.sensor_rng)

        self.jet_frailty = {
            f"J-{i:02d}": float(self.rng.gamma(2.0, 0.5))
            for i in range(config.n_jets)
        }
        self.part_health: Dict[str, float] = {}   # continuous latent
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

    @staticmethod
    def _discretise(continuous: float) -> int:
        """Map continuous latent to discrete state."""
        if continuous > 1.0:
            return 4
        elif continuous > 0.7:
            return 3
        elif continuous > 0.4:
            return 2
        return 1

    def run(self):
        events = []
        ground_truth = []
        sensor_readings = []

        for day in range(self.config.horizon_days):
            t = float(day * 24)
            regime = self.py_rng.choice(["normal", "hot_dusty", "surge_high_g"])

            for jid, jet in self.jets.items():
                frailty = self.jet_frailty[jid]
                flight_hours = float(self.rng.uniform(1.5, 4.0))

                events.append({
                    "t": t, "type": "sortie", "jet": jid,
                    "part": "", "sn": "",
                    "details": {"hours": flight_hours, "family": "alt",
                                "regime": regime},
                })

                for pname, sn in jet["parts"].items():
                    pcfg = next(p for p in self.config.parts if p.name == pname)
                    damage = float(self.rng.gamma(
                        shape=pcfg.weibull_shape,
                        scale=flight_hours * frailty / pcfg.weibull_scale,
                    ))
                    self.part_health[sn] += damage
                    old_state = self.part_states[sn]
                    state = self._discretise(self.part_health[sn])

                    if state != old_state:
                        self.part_states[sn] = state
                        ground_truth.append({
                            "sn": sn, "t": t,
                            "true_state": state,
                            "continuous_health": round(self.part_health[sn], 6),
                            "jet_frailty": round(frailty, 4),
                        })

                    # Alt sensor readings with correlated heavy-tailed noise
                    readings = self.alt_sensor.read_all(
                        t, sn, self.part_health[sn], regime)
                    sensor_readings.extend(readings)

        return events, ground_truth, sensor_readings


# ===========================================================================
# Simulator-to-Evidence Adapter location declaration  (Fix #8)
# ===========================================================================
#
# The adapter that converts simulator event streams into EvidenceItems lives
# at:  adapter/sim_adapter.py
#
# It is OUTSIDE nirnay/ (so the nirnay package has no dependency on
# simulator internals) and OUTSIDE simulator/ (so no ground truth leaks).
# The adapter reads only the event stream (not ground truth), signs each
# item, and writes it into an AppendOnlyStore.
#
# This module does NOT implement the adapter — it only declares the
# architectural boundary.  The adapter will be built in Stage 4 (Detection).
