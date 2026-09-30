"""
compare_decision_rules.py
--------------------------
Recompute RISE detection thresholds under three decision rules, from the
already-collected full probability trajectories (no retraining, no new
inference):

  1. raw       — argmax(P(class)), the original rule. Sanity-checked against
                 results/raw/thresholds_gray*.npz (should match exactly).
  2. debiased  — argmax(log P(class) - log prior(class)), where prior(class)
                 is THIS model's own mean predicted probability for that
                 class at RISE step 1 (its "no real signal" default rate).
                 Directly targets response-bias / default-class collapse.
  3. reject    — argmax(P(class)) but only if max P(class) > CONF_THRESHOLD,
                 else "no detection" (never counts as detected, regardless
                 of step). Tests whether requiring genuine confidence, not
                 just being the biggest of four numbers, changes the picture.

Usage:
    .venv/bin/python scripts/compare_decision_rules.py
"""

from __future__ import annotations
import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)  # repo root, so data/, models/, results/ resolve from anywhere
import numpy as np
from scipy import stats

CLASSES = ["bird", "cat", "fish", "snake"]
N_STEPS = 20
CONF_THRESHOLD = 0.4  # meaningfully above 0.25 chance level for 4 classes

MODELS = {
    "original":        "results/raw/probs_original.npz",
    "seed1":           "results/raw/probs_seed1.npz",
    "seed2":           "results/raw/probs_seed2.npz",
    "seed3":           "results/raw/probs_seed3.npz",
    "pretrained_gray": "results/raw/probs_pretrained_gray.npz",
}


def first_and_sustained(preds_correct: np.ndarray) -> int:
    """preds_correct: bool array (20,). Returns 1-indexed step of first
    correct prediction that stays correct through step 20, else N_STEPS+1."""
    for i in range(N_STEPS):
        if preds_correct[i] and preds_correct[i:].all():
            return i + 1
    return N_STEPS + 1


def compute_thresholds(probs: np.ndarray, true_idx: np.ndarray, rule: str,
                        prior: np.ndarray | None = None) -> dict:
    """probs: (n, 20, 4). Returns {class: array of thresholds}."""
    n = probs.shape[0]
    thresholds = {cls: [] for cls in CLASSES}

    for i in range(n):
        cls = CLASSES[true_idx[i]]
        seq = probs[i]  # (20, 4)

        if rule == "raw":
            pred = seq.argmax(axis=1)
            correct = (pred == true_idx[i])
        elif rule == "debiased":
            corrected = np.log(seq + 1e-9) - np.log(prior + 1e-9)
            pred = corrected.argmax(axis=1)
            correct = (pred == true_idx[i])
        elif rule == "reject":
            pred = seq.argmax(axis=1)
            maxprob = seq.max(axis=1)
            correct = (pred == true_idx[i]) & (maxprob > CONF_THRESHOLD)
        else:
            raise ValueError(rule)

        thresholds[cls].append(first_and_sustained(correct))

    return {cls: np.array(v) for cls, v in thresholds.items()}


def summarize(name: str, thresholds: dict):
    print(f"  [{name}]")
    for cls in CLASSES:
        arr = thresholds[cls]
        never = int((arr > N_STEPS).sum())
        print(f"    {cls:6s}  mean={arr.mean():6.2f}  median={np.median(arr):5.1f}  "
              f"sd={arr.std():5.2f}  never_detected={never}/{len(arr)}")


def main():
    for model_name, path in MODELS.items():
        data = np.load(path)
        probs, true_idx = data["probs"], data["true_idx"]

        print(f"\n{'='*70}\n{model_name}\n{'='*70}")

        # Rule 1: raw (sanity check vs. existing results)
        raw = compute_thresholds(probs, true_idx, "raw")
        summarize("raw (argmax)", raw)

        # Rule 2: debiased — prior = this model's mean prob per class at step 1
        prior = probs[:, 0, :].mean(axis=0)  # (4,)
        prior = prior / prior.sum()
        print(f"  step-1 prior (this model's default bias): "
              + ", ".join(f"{c}={p:.3f}" for c, p in zip(CLASSES, prior)))
        debiased = compute_thresholds(probs, true_idx, "debiased", prior=prior)
        summarize("debiased (equal-prior argmax)", debiased)

        # Rule 3: reject
        reject = compute_thresholds(probs, true_idx, "reject")
        summarize(f"reject (confidence > {CONF_THRESHOLD})", reject)

        # Quick omnibus test on debiased result, excl. nothing this time —
        # de-biasing is specifically meant to remove the need to exclude a
        # collapsed class, so test all 4.
        arrs = [debiased[c] for c in CLASSES]
        h, p = stats.kruskal(*arrs)
        print(f"  Kruskal-Wallis on debiased thresholds (all 4 classes): H={h:.2f}  p={p:.4f}")


if __name__ == "__main__":
    main()
