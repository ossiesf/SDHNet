"""
run_baseline_seeds.py
----------------------
Run the RISE experiment through several DIFFERENT random-weight
initializations of the untrained model, to check whether the strong
non-flat bias seen in the single-init baseline (bird/fish essentially
never detected, cat/snake favored, snake median threshold = step 1) is
a property of the architecture in general, or a fluke of that one seed.

Each seed gets its own results/raw/thresholds_random_seed{N}.npz so the
original (unseeded) baseline run is left untouched for comparison.

Usage:
    .venv/bin/python scripts/run_baseline_seeds.py --seeds 1 2 3
"""

from __future__ import annotations
import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)  # repo root, so data/, models/, results/ resolve from anywhere
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from fastai.vision.all import *
from torchvision.models import resnet18

from rise import run_experiment, N_STEPS

DATA_PATH = Path("data/sdh_500_clean")
CLASSES = ["bird", "cat", "fish", "snake"]
VALID_DIR = DATA_PATH / "valid"


def build_dls():
    dblock = DataBlock(
        blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        splitter=GrandparentSplitter(train_name="train", valid_name="valid"),
        item_tfms=RandomResizedCrop(224, min_scale=0.75),
        batch_tfms=[Flip(), Brightness(max_lighting=0.2),
                    Normalize.from_stats([0.449], [0.226])],
    )
    return dblock.dataloaders(DATA_PATH, bs=32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    os.makedirs("results/raw", exist_ok=True)
    dls = build_dls()

    for seed in args.seeds:
        print(f"\n=== Random-weights baseline, seed={seed} ===")
        torch.manual_seed(seed)
        learn = vision_learner(dls, resnet18, metrics=[accuracy], pretrained=False, n_in=1)
        # no learn.load() — weights stay at this seed's random init

        results = run_experiment(learn, VALID_DIR, classes=CLASSES, verbose=False)
        thresholds = {cls: [] for cls in CLASSES}
        for r in results:
            t = r["threshold"] if r["threshold"] is not None else N_STEPS + 1
            thresholds[r["class"]].append(t)

        out_path = f"results/raw/thresholds_random_seed{seed}.npz"
        np.savez(out_path, **{cls: np.array(v) for cls, v in thresholds.items()})
        print(f"Saved -> {out_path}")
        for cls in CLASSES:
            vals = thresholds[cls]
            print(f"  {cls:6s}  mean={np.mean(vals):.2f}  median={np.median(vals):.1f}  sd={np.std(vals):.2f}")


if __name__ == "__main__":
    main()
