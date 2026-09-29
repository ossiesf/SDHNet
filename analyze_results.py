"""
analyze_results.py
-------------------
Lightweight significance testing on RISE detection thresholds — the MVP-scope
stats, not the full PLAN.md rigor protocol (no seed-averaging, no Cliff's
delta, no bootstrap CIs, no pre-registration — those are v2).

Fish is excluded from the comparison: ~98% of fish images are "detected" at
RISE step 1 (maximum scrambling), which is a response-bias artifact (the
model defaults to "fish" under high uncertainty), not a real perceptual
result. See MVP_PLAN.md for the full explanation.

Usage:
    python analyze_results.py [--results results/thresholds_gray.npz]
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from scipy import stats


def holm_correct(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted = [None] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = pvals[idx] * (m - rank)
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/thresholds_gray.npz")
    args = ap.parse_args()

    data = np.load(args.results)
    bird, cat, snake = data["bird"], data["cat"], data["snake"]

    print("=== Descriptives (excl. fish — flagged as response-bias artifact) ===")
    for name, arr in [("bird", bird), ("cat", cat), ("snake", snake)]:
        print(f"{name:6s} n={len(arr)}  mean={arr.mean():.2f}  "
              f"median={np.median(arr):.1f}  sd={arr.std(ddof=1):.2f}")

    print("\n=== Omnibus: Kruskal-Wallis (bird, cat, snake) ===")
    h, p_omnibus = stats.kruskal(bird, cat, snake)
    print(f"H={h:.3f}  p={p_omnibus:.6f}")

    print("\n=== Pairwise: Mann-Whitney U, snake vs each, Holm-corrected (m=2) ===")
    pairs = [("snake", "bird", snake, bird), ("snake", "cat", snake, cat)]
    raw = []
    for name_a, name_b, a, b in pairs:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        rank_biserial = 1 - (2 * u) / (len(a) * len(b))
        raw.append((name_a, name_b, u, p, rank_biserial))

    holm_p = holm_correct([r[3] for r in raw])
    for (name_a, name_b, u, p, rb), hp in zip(raw, holm_p):
        verdict = "SIGNIFICANT (a=.05)" if hp < 0.05 else "not significant"
        print(f"{name_a} vs {name_b}: U={u:.1f}  raw_p={p:.4f}  "
              f"holm_p={hp:.4f}  rank_biserial_r={rb:.3f}  {verdict}")


if __name__ == "__main__":
    main()
