"""
run_rise_full_probs.py
-----------------------
Like run_rise_on_model.py, but saves the FULL per-image, per-step
probability trajectory (not just the final threshold), so alternate
decision rules (de-biasing, rejection) can be computed post-hoc without
re-running inference.

Output: results/probs_<model>.npz with:
  probs      — float array (n_images, 20, n_classes)
  true_idx   — int array (n_images,) index into CLASSES
  filenames  — array of str (n_images,)
  classes    — CLASSES, for reference

Usage:
    .venv/bin/python run_rise_full_probs.py --model sdhnet-4class-gray-best --out results/probs_original.npz
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
from fastai.vision.all import *
from torchvision.models import resnet18

from rise import generate_rise_sequence, stable_hash, N_STEPS

DATA_PATH = Path("data/sdh_500_clean")
CLASSES = ["bird", "cat", "fish", "snake"]
VALID_DIR = DATA_PATH / "valid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
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
    vocab = list(learn.dls.vocab)
    assert vocab == CLASSES, f"vocab mismatch: {vocab} vs {CLASSES}"

    exts = {".jpg", ".jpeg", ".png"}
    all_probs, all_true, all_files = [], [], []

    t0 = time.time()
    n = 0
    for cls in CLASSES:
        cls_dir = VALID_DIR / cls
        imgs = sorted([p for p in cls_dir.iterdir() if p.suffix.lower() in exts])
        for img_path in imgs:
            seed = stable_hash(img_path.name) % (2**31)
            frames = generate_rise_sequence(img_path, seed=seed)
            step_probs = []
            for frame in frames:
                with learn.no_bar():
                    _, _, probs = learn.predict(frame)
                step_probs.append(probs.numpy())
            all_probs.append(np.stack(step_probs))  # (20, 4)
            all_true.append(CLASSES.index(cls))
            all_files.append(img_path.name)
            n += 1
            if n % 50 == 0:
                print(f"  {n}/400  ({time.time()-t0:.0f}s elapsed)")

    probs_arr = np.stack(all_probs)          # (n, 20, 4)
    true_arr = np.array(all_true)            # (n,)
    files_arr = np.array(all_files)

    np.savez(args.out, probs=probs_arr, true_idx=true_arr,
             filenames=files_arr, classes=np.array(CLASSES))
    print(f"Saved -> {args.out}  probs shape={probs_arr.shape}  "
          f"({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
