"""
train_seed.py
--------------
Retrain the grayscale SDHNet classifier from scratch with a different random
seed, using the exact same recipe as sdhnet_4class_gray.ipynb (architecture,
augmentation, LR strategy, callbacks) — the only thing that changes is the
seed, which affects both weight initialization and the augmentation stream.
Data split (train/valid) is unchanged: GrandparentSplitter is folder-based,
not random.

This exists to check whether the RISE snake-vs-bird/cat result from the
original single trained run replicates across independent training runs —
same logic as run_baseline_seeds.py, applied to the trained model instead of
the untrained baseline.

Usage:
    .venv/bin/python scripts/train_seed.py --seed 1
"""

from __future__ import annotations
import os
from pathlib import Path
os.chdir(Path(__file__).resolve().parent.parent)  # repo root, so data/, models/, results/ resolve from anywhere
import argparse
import time
from pathlib import Path

import torch
from fastai.vision.all import *
from torchvision.models import resnet18

DATA_PATH = Path("data/sdh_500_clean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    dblock_gray = DataBlock(
        blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
        get_items=get_image_files,
        get_y=parent_label,
        splitter=GrandparentSplitter(train_name="train", valid_name="valid"),
        item_tfms=RandomResizedCrop(224, min_scale=0.75),
        batch_tfms=[Flip(), Brightness(max_lighting=0.2),
                    Normalize.from_stats([0.449], [0.226])],
    )
    dls_gray = dblock_gray.dataloaders(DATA_PATH, batch_size=32)

    learn_gray = vision_learner(
        dls_gray, resnet18,
        metrics=[accuracy, F1Score(average="macro")],
        pretrained=False, n_in=1,
    )

    t0 = time.time()
    suggested = learn_gray.lr_find(suggest_funcs=valley)
    lr = float(suggested.valley)
    print(f"[seed {args.seed}] Suggested LR (valley): {lr:.6g}  "
          f"({time.time()-t0:.1f}s for lr_find)")

    fname = f"sdhnet-4class-gray-seed{args.seed}-best"
    cbs = [
        EarlyStoppingCallback(monitor="valid_loss", min_delta=0.005, patience=8),
        SaveModelCallback(monitor="valid_loss", fname=fname),
    ]

    t1 = time.time()
    learn_gray.fit_one_cycle(args.epochs, lr, cbs=cbs)
    print(f"[seed {args.seed}] Training done in {time.time()-t1:.1f}s "
          f"(total incl. lr_find: {time.time()-t0:.1f}s)")
    print(f"[seed {args.seed}] Best model saved -> models/{fname}.pth")


if __name__ == "__main__":
    main()
