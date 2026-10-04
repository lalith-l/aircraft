"""
Truth Score Module

Computes a trust weight per maintenance record using:
1. Dawid-Skene annotator reliability estimation.
2. MinHash near-duplicate detection.
3. Laya contradiction question flags.

Output: trust weight ∈ (0, 1] per record.

Status: [DEMO over real code]
"""

import numpy as np
import hashlib
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass, field


@dataclass
class Record:
    """A maintenance record to be scored."""
    record_id: str
    text: str
    annotator_id: str
    labels: Dict[str, bool] = field(default_factory=dict)
    trust_weight: float = 1.0


# ---------------------------------------------------------------------------
# Dawid-Skene Reliability
# ---------------------------------------------------------------------------

class DawidSkene:
    """
    Simplified Dawid-Skene estimator for annotator reliability.

    Models each annotator's confusion matrix and iteratively
    re-estimates true labels and annotator quality.
    """

    def __init__(self, n_classes: int = 2, max_iter: int = 20):
        self.n_classes = n_classes
        self.max_iter = max_iter

    def fit(self, annotations: Dict[str, Dict[str, int]]
            ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """
        Fit the model.

        Args:
            annotations: {item_id: {annotator_id: label (0 or 1)}}

        Returns:
            (item_estimates, annotator_reliability)
            item_estimates: {item_id: P(true_label=1)}
            annotator_reliability: {annotator_id: (2x2) confusion matrix}
        """
        items = list(annotations.keys())
        annotators: Set[str] = set()
        for item_anns in annotations.values():
            annotators.update(item_anns.keys())
        annotators = sorted(annotators)

        # Initialise: majority vote
        item_probs = {}
        for item_id, anns in annotations.items():
            votes = list(anns.values())
            item_probs[item_id] = np.mean(votes) if votes else 0.5

        # EM iterations
        ann_reliability = {a: np.eye(self.n_classes) * 0.8 + 0.1
                           for a in annotators}

        for _ in range(self.max_iter):
            # M-step: update annotator confusion matrices
            for ann in annotators:
                confusion = np.ones((self.n_classes, self.n_classes)) * 0.01
                for item_id, anns in annotations.items():
                    if ann not in anns:
                        continue
                    label = anns[ann]
                    p1 = item_probs[item_id]
                    # Weight by estimated true class probability
                    confusion[1, label] += p1
                    confusion[0, label] += (1.0 - p1)

                # Normalise rows
                row_sums = confusion.sum(axis=1, keepdims=True)
                row_sums = np.maximum(row_sums, 1e-12)
                ann_reliability[ann] = confusion / row_sums

            # E-step: update item probabilities
            for item_id, anns in annotations.items():
                log_p0, log_p1 = 0.0, 0.0
                for ann, label in anns.items():
                    cm = ann_reliability[ann]
                    log_p0 += np.log(max(cm[0, label], 1e-12))
                    log_p1 += np.log(max(cm[1, label], 1e-12))

                # Prior: uniform
                max_log = max(log_p0, log_p1)
                p0 = np.exp(log_p0 - max_log)
                p1 = np.exp(log_p1 - max_log)
                total = p0 + p1
                item_probs[item_id] = p1 / total if total > 0 else 0.5

        return item_probs, ann_reliability

    @staticmethod
    def annotator_quality(confusion: np.ndarray) -> float:
        """Summarise annotator quality as accuracy from confusion matrix."""
        return float(np.trace(confusion) / max(confusion.sum(), 1e-12))


# ---------------------------------------------------------------------------
# MinHash Near-Duplicate Detection
# ---------------------------------------------------------------------------

class MinHashDuplicateDetector:
    """
    Detects near-duplicate records using MinHash signatures.
    """

    def __init__(self, n_hashes: int = 100, shingle_size: int = 3,
                 threshold: float = 0.5):
        self.n_hashes = n_hashes
        self.shingle_size = shingle_size
        self.threshold = threshold
        # Random hash parameters
        rng = np.random.RandomState(42)
        self.a = rng.randint(1, 2**31, size=n_hashes)
        self.b = rng.randint(0, 2**31, size=n_hashes)
        self.p = 2**31 - 1  # large prime

    def _shingle(self, text: str) -> Set[str]:
        """Create character shingles from text."""
        text = text.lower().strip()
        if len(text) < self.shingle_size:
            return {text}
        return {text[i:i + self.shingle_size]
                for i in range(len(text) - self.shingle_size + 1)}

    def _hash_shingle(self, shingle: str) -> int:
        return int(hashlib.md5(shingle.encode()).hexdigest()[:8], 16)

    def signature(self, text: str) -> np.ndarray:
        """Compute MinHash signature for a text."""
        shingles = self._shingle(text)
        if not shingles:
            return np.zeros(self.n_hashes)

        shingle_hashes = np.array([self._hash_shingle(s) for s in shingles])

        sig = np.full(self.n_hashes, np.iinfo(np.int64).max)
        for sh in shingle_hashes:
            h = (self.a * sh + self.b) % self.p
            sig = np.minimum(sig, h)

        return sig

    def similarity(self, sig_a: np.ndarray, sig_b: np.ndarray) -> float:
        """Estimate Jaccard similarity from signatures."""
        return float(np.mean(sig_a == sig_b))

    def find_duplicates(self, records: List[Record]
                         ) -> List[Tuple[str, str, float]]:
        """
        Find near-duplicate pairs among records.

        Returns: List of (id_a, id_b, similarity) where similarity >= threshold.
        """
        sigs = [(r.record_id, self.signature(r.text)) for r in records]
        pairs = []

        for i in range(len(sigs)):
            for j in range(i + 1, len(sigs)):
                sim = self.similarity(sigs[i][1], sigs[j][1])
                if sim >= self.threshold:
                    pairs.append((sigs[i][0], sigs[j][0], sim))

        return pairs


# ---------------------------------------------------------------------------
# Truth Scorer (combines all signals)
# ---------------------------------------------------------------------------

class TruthScorer:
    """
    Combines Dawid-Skene reliability, MinHash duplicates, and
    contradiction flags into a single trust weight per record.
    """

    def __init__(self):
        self.ds = DawidSkene()
        self.minhash = MinHashDuplicateDetector()

    def score(self, records: List[Record],
              annotations: Dict[str, Dict[str, int]],
              contradiction_flags: Dict[str, bool] = None
              ) -> Dict[str, float]:
        """
        Compute trust weights for all records.

        Args:
            records: list of Record objects.
            annotations: Dawid-Skene input.
            contradiction_flags: {record_id: True if contradictory}.

        Returns: {record_id: trust_weight}
        """
        if contradiction_flags is None:
            contradiction_flags = {}

        # 1. Dawid-Skene: annotator-quality-weighted item reliability
        item_probs, ann_rel = self.ds.fit(annotations)

        # 2. MinHash: penalise duplicates
        dup_pairs = self.minhash.find_duplicates(records)
        dup_penalty: Dict[str, float] = {}
        for id_a, id_b, sim in dup_pairs:
            # The later duplicate gets penalised
            dup_penalty[id_b] = min(dup_penalty.get(id_b, 1.0), 1.0 - sim)

        # 3. Combine
        weights = {}
        for r in records:
            w = 1.0

            # Dawid-Skene: item certainty
            if r.record_id in item_probs:
                certainty = max(item_probs[r.record_id],
                                1.0 - item_probs[r.record_id])
                w *= certainty

            # MinHash duplicate penalty
            if r.record_id in dup_penalty:
                w *= dup_penalty[r.record_id]

            # Contradiction penalty
            if contradiction_flags.get(r.record_id, False):
                w *= 0.3

            weights[r.record_id] = max(w, 0.01)  # floor at 1%

        return weights
