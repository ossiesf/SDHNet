"""
rise.py
-------
Random Image Structure Evolution (Sadr & Sinha, 2004) as used in
Kawai & He (2016) "Breaking Snake Camouflage".

The algorithm progressively restores a phase-scrambled image toward its
original over 20 steps. At step 1 the phase is 95% random; at step 20 it
is identical to the original. Magnitude spectrum (and therefore overall
luminance / contrast) is held constant throughout.

Matching Kawai's procedure:
  - All stimuli converted to grayscale before scrambling
  - 20 steps, α = linspace(0.95, 0.0, 20)
  - Circular phase interpolation via complex-unit mixing (avoids wrap artefacts)
  - After IFFT, output normalised to original mean luminance and RMS contrast

Usage (generate stimuli):
    cd ~/Documents/Claude/Projects/SDH/sdhnet
    python scripts/rise.py --src data/sdh_500_clean/valid --dst data/sdh_rise

Usage (run inference from another script):
    from rise import generate_rise_sequence, run_model_on_sequence
"""

from __future__ import annotations
import argparse
import time
import zlib
from pathlib import Path


def stable_hash(s: str) -> int:
    """Deterministic string hash, stable across processes and runs.

    Python's builtin hash() is randomized per-process for str/bytes by
    default (PYTHONHASHSEED) — using it for a reproducibility-critical seed
    means every separate run generates different RISE stimuli for the "same"
    image. zlib.crc32 is deterministic across processes and platforms.
    """
    return zlib.crc32(s.encode("utf-8"))

import numpy as np
from PIL import Image

# ── Core RISE algorithm ───────────────────────────────────────────────────────

N_STEPS    = 20
MAX_ALPHA  = 0.95   # step 1: 95% random phase
ALPHAS     = np.linspace(MAX_ALPHA, 0.0, N_STEPS)   # shape (20,)


def generate_rise_sequence(
    img: Image.Image | str | Path,
    seed: int | None = None,
    size: int = 224,
) -> list[Image.Image]:
    """
    Generate a 20-step RISE sequence for one image.

    Parameters
    ----------
    img   : PIL Image (any mode) or path to image file
    seed  : optional RNG seed for reproducibility
    size  : resize to (size, size) before scrambling

    Returns
    -------
    List of 20 grayscale PIL Images, step 1 (most scrambled) → step 20 (original).
    """
    if not isinstance(img, Image.Image):
        img = Image.open(img)

    # Convert to grayscale and resize
    gray = img.convert("L").resize((size, size), Image.LANCZOS)
    arr = np.array(gray, dtype=np.float64)

    # Store luminance statistics for post-IFFT normalisation
    orig_mean = arr.mean()
    orig_std  = arr.std()

    # FFT — work in frequency domain
    fft        = np.fft.fft2(arr)
    magnitude  = np.abs(fft)
    orig_phase = np.angle(fft)

    # Random phase drawn once per sequence (fixed for the 20 steps)
    rng          = np.random.default_rng(seed)
    rand_phase   = rng.uniform(-np.pi, np.pi, orig_phase.shape)

    # Pre-compute complex unit phasors for circular interpolation
    # Mixing exp(iθ) rather than θ avoids artefacts at the ±π wrap boundary
    orig_phasor = np.exp(1j * orig_phase)
    rand_phasor = np.exp(1j * rand_phase)

    frames: list[Image.Image] = []

    for alpha in ALPHAS:
        # Circular interpolation of phase
        mixed_phasor = (1.0 - alpha) * orig_phasor + alpha * rand_phasor
        mixed_phase  = np.angle(mixed_phasor)   # back to [-π, π]

        # Reconstruct with original magnitude + mixed phase
        fft_mixed     = magnitude * np.exp(1j * mixed_phase)
        reconstructed = np.real(np.fft.ifft2(fft_mixed))

        # Normalise to original mean and RMS contrast
        rec_std = reconstructed.std()
        if rec_std > 1e-8 and orig_std > 1e-8:
            reconstructed = (reconstructed - reconstructed.mean()) / rec_std
            reconstructed = reconstructed * orig_std + orig_mean

        reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
        frames.append(Image.fromarray(reconstructed, "L"))

    return frames   # frames[0] = most scrambled, frames[-1] = original


# ── Model inference helpers ───────────────────────────────────────────────────

def run_model_on_sequence(
    learn,
    frames: list[Image.Image],
    true_class: str,
) -> dict:
    """
    Run a fastai learner over a RISE sequence and return per-step results.

    Parameters
    ----------
    learn      : trained fastai Learner (grayscale or color)
    frames     : list of 20 PIL Images from generate_rise_sequence()
    true_class : ground-truth class label (e.g. 'snake')

    Returns
    -------
    dict with keys:
      'probs'      : np.ndarray shape (20, n_classes) — softmax probabilities
      'preds'      : list of 20 predicted class strings
      'threshold'  : int (1-indexed step of first correct prediction), or None
      'true_class' : str
    """
    import torch

    vocab = learn.dls.vocab   # e.g. ['bird', 'cat', 'fish', 'snake']
    true_idx = vocab.o2i[true_class]

    probs_list = []
    preds_list = []

    for frame in frames:
        with learn.no_bar():
            pred_class, pred_idx, probs = learn.predict(frame)
        probs_list.append(probs.numpy())
        preds_list.append(str(pred_class))

    probs_arr = np.stack(probs_list)   # (20, n_classes)

    # Detection threshold: first step (1-indexed) where model predicts correctly
    threshold = None
    for i, pred in enumerate(preds_list):
        if pred == true_class:
            threshold = i + 1   # 1-indexed to match Kawai's reporting
            break

    return {
        "probs":      probs_arr,
        "preds":      preds_list,
        "threshold":  threshold,   # None if never detected
        "true_class": true_class,
    }


# ── Batch processing for the full validation set ──────────────────────────────

def run_experiment(
    learn,
    src_dir: str | Path,
    classes: list[str] | None = None,
    seed_offset: int = 0,
    verbose: bool = True,
) -> list[dict]:
    """
    Run RISE over every image in src_dir and return a list of result dicts.

    src_dir should have the structure:  src_dir/<class>/<image files>

    Each result dict contains everything from run_model_on_sequence() plus:
      'class'    : true class
      'filename' : image filename
      'step_probs': same as 'probs'
    """
    src_dir = Path(src_dir)
    if classes is None:
        classes = sorted([d.name for d in src_dir.iterdir() if d.is_dir()])

    results = []
    exts    = {".jpg", ".jpeg", ".png"}
    t0      = time.time()

    for cls in classes:
        cls_dir = src_dir / cls
        imgs    = [p for p in cls_dir.iterdir() if p.suffix.lower() in exts]
        if verbose:
            print(f"\n{cls}: {len(imgs)} images")

        for i, img_path in enumerate(sorted(imgs)):
            seed = seed_offset + stable_hash(img_path.name) % (2**31)
            frames = generate_rise_sequence(img_path, seed=seed)
            result = run_model_on_sequence(learn, frames, true_class=cls)
            result["class"]     = cls
            result["filename"]  = img_path.name
            result["step_probs"] = result.pop("probs")
            results.append(result)

            if verbose and (i + 1) % 10 == 0:
                elapsed = time.time() - t0
                print(f"  {i+1}/{len(imgs)}  ({elapsed:.0f}s elapsed)")

    return results


# ── CLI: generate and save RISE stimuli to disk ───────────────────────────────

def save_rise_stimuli(
    src_dir: Path,
    dst_dir: Path,
    classes: list[str] | None = None,
    seed_offset: int = 0,
    n_steps: int = N_STEPS,
) -> None:
    """
    Pre-generate and save all RISE sequences as JPEG files.

    Output structure:
        dst_dir/<class>/<stem>_step01.jpg  …  <stem>_step20.jpg

    This is optional — you can also generate on-the-fly during inference.
    Saving to disk is useful for visual inspection and for running Claude.
    """
    exts = {".jpg", ".jpeg", ".png"}
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)

    if classes is None:
        classes = sorted([d.name for d in src_dir.iterdir() if d.is_dir()])

    total = sum(
        len([p for p in (src_dir / cls).iterdir() if p.suffix.lower() in exts])
        for cls in classes
    )
    print(f"Saving RISE stimuli: {total} images × {n_steps} steps → {dst_dir}")
    done = 0
    t0   = time.time()

    for cls in classes:
        cls_src = src_dir / cls
        cls_dst = dst_dir / cls
        cls_dst.mkdir(parents=True, exist_ok=True)

        imgs = sorted([p for p in cls_src.iterdir() if p.suffix.lower() in exts])
        for img_path in imgs:
            seed   = seed_offset + stable_hash(img_path.name) % (2**31)
            frames = generate_rise_sequence(img_path, seed=seed)

            for step_idx, frame in enumerate(frames):
                out_name = f"{img_path.stem}_step{step_idx+1:02d}.jpg"
                frame.save(cls_dst / out_name, "JPEG", quality=92)

            done += 1
            if done % 20 == 0:
                elapsed = time.time() - t0
                rate    = done / elapsed
                eta     = (total - done) / rate
                print(f"  {done}/{total}  ~{eta/60:.1f}min remaining")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min → {dst_dir.resolve()}")


# ── Quick visual sanity check ─────────────────────────────────────────────────

def make_contact_sheet(
    frames: list[Image.Image],
    title: str = "",
    out_path: str | Path | None = None,
) -> Image.Image:
    """
    Lay out all 20 RISE steps in a single row for visual inspection.
    Saves to out_path if provided, always returns the PIL Image.
    """
    from PIL import ImageDraw

    thumb  = 112
    pad    = 4
    W      = N_STEPS * (thumb + pad) + pad
    H      = thumb + pad * 2 + 20
    canvas = Image.new("RGB", (W, H), (20, 20, 20))
    draw   = ImageDraw.Draw(canvas)

    for i, frame in enumerate(frames):
        x = pad + i * (thumb + pad)
        y = pad
        canvas.paste(frame.resize((thumb, thumb)).convert("RGB"), (x, y))
        draw.text((x + 2, y + thumb + 2), f"{i+1}", fill=(160, 160, 160))

    if title:
        draw.text((pad, H - 14), title, fill=(200, 200, 200))

    if out_path:
        canvas.save(out_path)

    return canvas


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate RISE stimulus sequences")
    parser.add_argument("--src",  default="data/sdh_500_clean/valid",
                        help="Source directory with class subfolders")
    parser.add_argument("--dst",  default="data/sdh_rise",
                        help="Output directory for saved RISE sequences")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed offset for reproducibility")
    parser.add_argument("--check", action="store_true",
                        help="Generate one contact sheet per class for visual QC")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)

    if args.check:
        # Quick QC: one contact sheet per class using the first image
        print("Generating QC contact sheets...")
        dst.mkdir(parents=True, exist_ok=True)
        classes = sorted([d.name for d in src.iterdir() if d.is_dir()])
        for cls in classes:
            imgs = sorted((src / cls).glob("*.jpg"))
            if not imgs:
                continue
            frames = generate_rise_sequence(imgs[0], seed=args.seed)
            out    = dst / f"qc_{cls}.jpg"
            make_contact_sheet(frames, title=f"{cls}: {imgs[0].name}", out_path=out)
            print(f"  {out}")
        print("Done. Open the qc_*.jpg files to verify scrambling looks correct.")
    else:
        save_rise_stimuli(src, dst, seed_offset=args.seed)
