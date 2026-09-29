"""
prepare_dataset.py
------------------
Builds the 4-class SDH dataset (snake / bird / cat / fish) from existing
downloaded data, creating a balanced train/val split ready for fastai.

Output structure:
    data/sdh/
        train/
            snake/   bird/   cat/   fish/
        valid/
            snake/   bird/   cat/   fish/

Sources (pooled per class):
    snake  — data/animals_extracted/.../snake/  (60)
           + data/downloaded/snake/**           (~181 downloaded)
    cat    — data/animals_extracted/.../cat/    (60)
           + data/downloaded/cat/**             (~163 downloaded)
    bird   — data/animals_extracted/.../<bird subcategories>/  (960)
           + data/animals/imagenette2-160/.../birds/           (960)
    fish   — data/animals_extracted/.../goldfish/ + .../seahorse/  (120)
           + data/downloaded/fish/**                               (~160 downloaded)

All classes capped at IMAGES_PER_CLASS for balance.
Train/val split: 80/20.
"""

import shutil
import random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SEED             = 42
IMAGES_PER_CLASS = 500
TRAIN_RATIO      = 0.8

BASE = Path(__file__).parent / "data"
OUT  = BASE / "sdh_500"

EXTRACTED = BASE / "animals_extracted/animals/animals"
DOWNLOADED = BASE / "downloaded"

BIRD_SUBCATEGORIES = [
    "crow", "duck", "eagle", "flamingo", "goose", "hornbill",
    "hummingbird", "owl", "parrot", "pelecaniformes", "pigeon",
    "sandpiper", "sparrow", "swan", "turkey", "woodpecker",
]

# Flat folder sources (direct children are image files)
FLAT_SOURCES = {
    "snake": [EXTRACTED / "snake"],
    "cat":   [EXTRACTED / "cat"],
    "bird":  [EXTRACTED / b for b in BIRD_SUBCATEGORIES] + [
                 BASE / "animals/imagenette2-160/train/birds",
                 BASE / "animals/imagenette2-160/valid/birds",
             ],
    "fish":  [EXTRACTED / "goldfish", EXTRACTED / "seahorse"],
}

# Nested/downloaded sources (use rglob to find images in subdirs)
NESTED_SOURCES = {
    "snake": [DOWNLOADED / "snake"],
    "cat":   [DOWNLOADED / "cat"],
    "fish":  [DOWNLOADED / "fish"],
}

VALID_EXTS = {".jpg", ".jpeg", ".png"}

# ── Helpers ───────────────────────────────────────────────────────────────────
def collect_images(class_name: str) -> list[Path]:
    imgs = []
    # Flat sources
    for folder in FLAT_SOURCES.get(class_name, []):
        if folder.exists():
            imgs += [p for p in folder.iterdir()
                     if p.is_file() and p.suffix.lower() in VALID_EXTS]
    # Nested/downloaded sources
    for folder in NESTED_SOURCES.get(class_name, []):
        if folder.exists():
            imgs += [p for p in folder.rglob("*")
                     if p.is_file() and p.suffix.lower() in VALID_EXTS]
    return imgs


def balanced_sample(imgs: list[Path], n: int, rng: random.Random) -> list[Path]:
    """Sample n images, trying to draw equally from each source folder."""
    if len(imgs) <= n:
        return imgs
    return rng.sample(imgs, n)


def copy_split(imgs: list[Path], class_name: str, rng: random.Random) -> dict:
    n_train = int(len(imgs) * TRAIN_RATIO)
    shuffled = imgs[:]
    rng.shuffle(shuffled)
    train_imgs, valid_imgs = shuffled[:n_train], shuffled[n_train:]

    counts = {}
    for split, split_imgs in [("train", train_imgs), ("valid", valid_imgs)]:
        dest_dir = OUT / split / class_name
        dest_dir.mkdir(parents=True, exist_ok=True)
        for i, src in enumerate(split_imgs):
            ext = src.suffix.lower()
            dest = dest_dir / f"{class_name}_{i:04d}{ext}"
            shutil.copy2(src, dest)
            dest.chmod(0o644)  # ensure writable for future rebuilds
        counts[split] = len(split_imgs)

    return counts


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rng = random.Random(SEED)

    # Clean output dir if it exists
    if OUT.exists():
        print(f"Removing existing {OUT} ...")
        shutil.rmtree(OUT)

    print(f"\nBuilding SDH dataset → {OUT}\n{'─'*50}")

    total = {}
    for class_name in FLAT_SOURCES:
        imgs = collect_images(class_name)
        sampled = balanced_sample(imgs, IMAGES_PER_CLASS, rng)
        counts = copy_split(sampled, class_name, rng)
        total[class_name] = counts
        print(
            f"  {class_name:<8} "
            f"found {len(imgs):>4}  "
            f"sampled {len(sampled):>3}  "
            f"→  train {counts['train']:>2} / valid {counts['valid']:>2}"
        )

    print(f"\n{'─'*50}")
    print("Dataset ready.\n")

    # Sanity check
    for split in ("train", "valid"):
        for cls in FLAT_SOURCES:
            n = len(list((OUT / split / cls).glob("*")))
            assert n > 0, f"Empty dir: {OUT / split / cls}"

    print("Verification passed — all class/split dirs are non-empty.")
    print(f"\nOutput: {OUT.resolve()}")


if __name__ == "__main__":
    main()
