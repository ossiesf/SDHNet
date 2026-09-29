"""
remove_backgrounds.py
---------------------
Strips backgrounds from all images in data/sdh_500/ using rembg (U2Net),
producing data/sdh_500_clean/ with the same structure.

Animals are composited onto a flat gray background so the model learns
animal features only — no habitat shortcuts.

Install:
    pip install rembg[gpu]   # if CUDA available
    pip install rembg        # CPU-only (slower but works)

Usage:
    cd ~/Documents/Claude/Projects/SDH/sdhnet
    python remove_backgrounds.py

The script is resumable — already-processed files are skipped.
Failed files are logged to data/rembg_failures.txt.

Why gray background (128,128,128) rather than white:
    Grayscale training normalises with mean≈0.449. A white background
    (255) is far from the mean and creates a strong edge signal at the
    animal boundary. Gray sits near the dataset mean, so the boundary
    is much softer and the model focuses on internal animal texture.
"""

from pathlib import Path
from PIL import Image
import io, sys, time

try:
    from rembg import remove, new_session
except ImportError:
    print("rembg not installed. Run: pip install rembg")
    sys.exit(1)

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

# ── Config ────────────────────────────────────────────────────────────────────
SRC  = Path("data/sdh_500")
DST  = Path("data/sdh_500_clean")
BG   = (128, 128, 128)   # neutral gray background
CLASSES = ["snake", "cat", "bird", "fish"]
SPLITS  = ["train", "valid"]
EXTS    = {".jpg", ".jpeg", ".png"}

# ── Setup ─────────────────────────────────────────────────────────────────────
DST.mkdir(parents=True, exist_ok=True)
failures = []

# Load rembg model once (downloads ~170MB U2Net on first run)
print("Loading rembg model (downloads ~170MB on first run)...")
session = new_session("u2net")
print("Model ready.\n")

# ── Count total work ──────────────────────────────────────────────────────────
all_imgs = [
    (split, cls, p)
    for split in SPLITS
    for cls in CLASSES
    for p in (SRC / split / cls).glob("*")
    if p.suffix.lower() in EXTS
]
total = len(all_imgs)
print(f"Total images to process: {total}")
print(f"Output → {DST.resolve()}\n{'─'*50}")

# ── Process ───────────────────────────────────────────────────────────────────
t0 = time.time()
done = 0
skipped = 0

iterator = tqdm(all_imgs, unit="img") if USE_TQDM else all_imgs

for split, cls, src_path in iterator:
    dest_dir = DST / split / cls
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / (src_path.stem + ".jpg")

    # Resumable: skip already processed
    if dest_path.exists():
        skipped += 1
        continue

    try:
        # Remove background → RGBA
        with open(src_path, "rb") as f:
            raw = f.read()
        result_bytes = remove(raw, session=session)
        rgba = Image.open(io.BytesIO(result_bytes)).convert("RGBA")

        # Composite onto gray background
        bg = Image.new("RGBA", rgba.size, (*BG, 255))
        composite = Image.alpha_composite(bg, rgba).convert("RGB")
        composite.save(dest_path, "JPEG", quality=92)
        done += 1

    except Exception as e:
        failures.append((str(src_path), str(e)))
        if not USE_TQDM:
            print(f"  FAILED: {src_path.name} — {e}")

    if not USE_TQDM and (done + skipped) % 50 == 0:
        elapsed = time.time() - t0
        rate = (done + 1) / elapsed
        remaining = (total - done - skipped) / rate
        print(f"  {done+skipped}/{total}  ({done} processed, {skipped} skipped)  "
              f"~{remaining/60:.1f}min remaining")

# ── Summary ───────────────────────────────────────────────────────────────────
elapsed = time.time() - t0
print(f"\n{'─'*50}")
print(f"Done in {elapsed/60:.1f} min")
print(f"  Processed: {done}")
print(f"  Skipped (already done): {skipped}")
print(f"  Failed: {len(failures)}")

if failures:
    fail_log = Path("data/rembg_failures.txt")
    fail_log.write_text("\n".join(f"{p}\t{e}" for p, e in failures))
    print(f"  Failure log → {fail_log}")

print(f"\nClean dataset → {DST.resolve()}")
print("Next: update sdhnet_4class.ipynb to point DATA_PATH at 'data/sdh_500_clean'")
print("      and retrain.")
