"""
train_pretrained.py
--------------------
Train a pretrained (ImageNet) comparison arm — color and/or grayscale — to
compare against the scratch-trained models. Mirrors the recipe in
sdhnet_4class_color.ipynb / sdhnet_4class_gray.ipynb, but with
pretrained=True and fine_tune() instead of fit_one_cycle(), matching that
project's own noted reasoning (fine_tune assumes pretrained early layers;
fit_one_cycle is for scratch).

Deliberately deferred in PLAN.md until the scratch baseline was solid — see
MVP_PLAN.md "Corrected result" section for why (ImageNet contains snake
synsets, so pretrained weights would otherwise confound the "is this
learned" question; this arm exists to quantify, not replace, that).

Usage:
    .venv/bin/python train_pretrained.py --mode gray
    .venv/bin/python train_pretrained.py --mode color
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

from fastai.vision.all import *
from torchvision.models import resnet18

DATA_PATH = Path("data/sdh_500_clean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["color", "gray"], required=True)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()

    if args.mode == "gray":
        # 0.449/0.226 == mean of the three ImageNet RGB channel stats —
        # the standard grayscale-equivalent normalization, already used for
        # the scratch-gray model, so pretrained-gray stays comparable to it.
        dblock = DataBlock(
            blocks=(ImageBlock(cls=PILImageBW), CategoryBlock),
            get_items=get_image_files, get_y=parent_label,
            splitter=GrandparentSplitter(train_name="train", valid_name="valid"),
            item_tfms=RandomResizedCrop(224, min_scale=0.75),
            batch_tfms=[Flip(), Brightness(max_lighting=0.2),
                        Normalize.from_stats([0.449], [0.226])],
        )
        n_in = 1
        fname = "sdhnet-4class-gray-pretrained-best"
    else:
        dblock = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files, get_y=parent_label,
            splitter=GrandparentSplitter(train_name="train", valid_name="valid"),
            item_tfms=RandomResizedCrop(224, min_scale=0.75),
            batch_tfms=[Flip(), Brightness(max_lighting=0.2),
                        Normalize.from_stats(*imagenet_stats)],
        )
        n_in = 3
        fname = "sdhnet-4class-color-pretrained-best"

    dls = dblock.dataloaders(DATA_PATH, batch_size=32)
    print(f"[{args.mode}] classes: {dls.vocab}  train={len(dls.train_ds)}  valid={len(dls.valid_ds)}")

    learn = vision_learner(
        dls, resnet18,
        metrics=[accuracy, F1Score(average="macro")],
        pretrained=True, n_in=n_in,
    )

    cbs = [
        EarlyStoppingCallback(monitor="valid_loss", min_delta=0.005, patience=8),
        SaveModelCallback(monitor="valid_loss", fname=fname),
    ]

    t0 = time.time()
    learn.fine_tune(args.epochs, cbs=cbs)
    print(f"[{args.mode}] Training done in {time.time()-t0:.1f}s")
    print(f"[{args.mode}] Best model saved -> models/{fname}.pth")


if __name__ == "__main__":
    main()
