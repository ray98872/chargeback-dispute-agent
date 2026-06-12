"""
NLP & Math Pipeline: DistilBERT classifier + Conformal Prediction wrapper
=========================================================================
Loads the fine-tuned DistilBERT from `model/` and wraps its softmax outputs
in split conformal prediction (Vovk et al.), turning raw network scores
into *calibrated* uncertainty with a finite-sample coverage guarantee.

Key quantities returned for a dispute text:

  p-value (per class)
      (#{calibration scores >= score_class} + 1) / (n_calib + 1)
      where score_class = 1 - softmax_class. The fraction of held-out
      calibration examples that look at least as "non-conforming" as this
      one would, were the class label true.

  credibility = largest p-value
      How well the best class fits. Low credibility means the input looks
      unlike ANYTHING seen in calibration (e.g. out-of-distribution text).

  confidence = 1 - second largest p-value
      How decisively every OTHER class is rejected. This is the calibrated
      score the agent thresholds on: confidence > 0.95 means that, with
      statistical guarantee, fewer than 5% of calibration-like inputs
      would be assigned a competing class.

  prediction_set (at alpha = 0.05)
      All classes with p-value > alpha. Singleton set == unambiguous.

Usage:
    from ai_classifier import ConformalDisputeClassifier
    clf = ConformalDisputeClassifier()
    result = clf.classify("I never made this payment, my card was stolen")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(__file__).parent / "model"
MAX_LENGTH = 96
DEFAULT_ALPHA = 0.05


@dataclass
class ClassificationResult:
    predicted_class: str
    confidence: float          # 1 - second largest p-value  (calibrated)
    credibility: float         # largest p-value
    softmax: dict[str, float]  # raw network probabilities (uncalibrated)
    p_values: dict[str, float]
    prediction_set: list[str] = field(default_factory=list)

    def is_decisive(self, threshold: float = 0.95) -> bool:
        """Auto-resolution rule: calibrated confidence above threshold AND
        the conformal prediction set contains exactly one class."""
        return self.confidence > threshold and len(self.prediction_set) == 1


class ConformalDisputeClassifier:
    def __init__(self, model_dir: Path | str = MODEL_DIR,
                 alpha: float = DEFAULT_ALPHA):
        model_dir = Path(model_dir)
        if not (model_dir / "conformal.json").exists():
            raise FileNotFoundError(
                f"No trained model found in {model_dir}/. "
                "Run `python train_model.py` first."
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(model_dir)
            .to(self.device)
            .eval()
        )

        with open(model_dir / "conformal.json") as f:
            calib = json.load(f)
        self.labels: list[str] = calib["labels"]
        # Sorted ascending so p-values can be computed via searchsorted.
        self.calibration_scores = np.sort(np.array(calib["calibration_scores"]))
        self.n_calib = len(self.calibration_scores)
        self.alpha = alpha
        self.metadata = {
            k: calib[k]
            for k in ("test_accuracy", "conformal_coverage_at_95",
                      "avg_prediction_set_size", "n_train", "n_calibration",
                      "n_test")
            if k in calib
        }

    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _softmax(self, text: str) -> np.ndarray:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**enc).logits
        return torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    def _p_value(self, nonconformity: float) -> float:
        """Conformal p-value:
        (#{calibration scores >= s} + 1) / (n + 1)."""
        idx = np.searchsorted(self.calibration_scores, nonconformity, side="left")
        n_geq = self.n_calib - idx
        return (n_geq + 1) / (self.n_calib + 1)

    # ------------------------------------------------------------------ #

    def classify(self, dispute_text: str) -> ClassificationResult:
        probs = self._softmax(dispute_text)

        # Hypothesis test per class: "would this input conform to the
        # calibration distribution if class k were the true label?"
        p_values = np.array([self._p_value(1.0 - probs[k])
                             for k in range(len(self.labels))])

        order = np.argsort(p_values)[::-1]
        best, second = order[0], order[1]

        prediction_set = [self.labels[k]
                          for k in range(len(self.labels))
                          if p_values[k] > self.alpha]

        return ClassificationResult(
            predicted_class=self.labels[best],
            confidence=float(1.0 - p_values[second]),
            credibility=float(p_values[best]),
            softmax={l: float(probs[i]) for i, l in enumerate(self.labels)},
            p_values={l: float(p_values[i]) for i, l in enumerate(self.labels)},
            prediction_set=prediction_set,
        )


if __name__ == "__main__":
    clf = ConformalDisputeClassifier()
    print(f"Model loaded on {clf.device} | calibration n={clf.n_calib}")
    print(f"Training metadata: {clf.metadata}\n")

    samples = [
        "I have never shopped at this store, my card was stolen last week "
        "and someone used it without my permission.",
        "The merchant charged me twice for one order and support is ignoring "
        "my refund request.",
        "They kept billing my card after I cancelled the subscription, which "
        "their own terms say should stop immediately.",
        "The weather was nice on Tuesday and I enjoyed a walk in the park.",
    ]
    for text in samples:
        r = clf.classify(text)
        print(f"text:        {text[:70]}...")
        print(f"  predicted: {r.predicted_class}")
        print(f"  confidence (calibrated): {r.confidence:.4f}")
        print(f"  credibility:             {r.credibility:.4f}")
        print(f"  prediction set:          {r.prediction_set}")
        print(f"  decisive @95%:           {r.is_decisive()}\n")
