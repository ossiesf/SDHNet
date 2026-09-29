"""
run_rise_on_model.py
---------------------
Run the RISE experiment (same as rise_experiment.ipynb cell 11) against an
arbitrary saved model checkpoint, for checking reliability of the RISE
result across independently trained seeds.

Usage:
    .venv/bin/python scripts/run_rise_on_model.py --model sdhnet-4class-gray-seed1-best --out results/thresholds_gray_seed1.npz
"""

from __future__ import annotations
import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)  # repo root, so data/, models/, results/ resolve from anywhere
import argparse
from pathlib import Path

import numpy as np
from fastai.vision.all import *
from torchvision.models import resnet18

from rise import run_experiment, N_STEPS

DATA_PATH = Path("data/sdh_500_clean")
CLASSES = ["bird", "cat", "fish", "snake"]
VALID_DIR = DATA_PATH / "valid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint name under models/, no .pth")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dblock_gray = DataBlock(
        blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        splitter=GrandparentSplitter(train_name="train", valid_name="valid"),
        item_tfms=RandomResizedCrop(224, min_scale=0.75),
        batch_tfms=[Flip(), Brightness(max_lighting=0.2),
                    Normalize.from_stats([0.449], [0.226])],
    )
    dls = dblock_gray.dataloaders(DATA_PATH, bs=32)
    learn = vision_learner(dls, resnet18, metrics=[accuracy], pretrained=False, n_in=1)
    learn.load(args.model)

    print(f"Running RISE experiment on {args.model}...")
    results = run_experiment(learn, VALID_DIR, classes=CLASSES, verbose=True)

    thresholds = {cls: [] for cls in CLASSES}
    for r in results:
        t = r["threshold"] if r["threshold"] is not None else N_STEPS + 1
        thresholds[r["class"]].append(t)

    np.savez(args.out, **{cls: np.array(v) for cls, v in thresholds.items()})
    print(f"Saved -> {args.out}")
    for cls in CLASSES:
        vals = thresholds[cls]
        print(f"  {cls:6s}  mean={np.mean(vals):.2f}  median={np.median(vals):.1f}  sd={np.std(vals):.2f}")


if __name__ == "__main__":
    main()
