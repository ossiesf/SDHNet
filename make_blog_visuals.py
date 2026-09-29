"""
make_blog_visuals.py
---------------------
Generates the two visuals needed for the blog post / LinkedIn post:

1. rise_contact_sheet.png — one snake image, all 20 RISE steps side by side.
2. rise_sequence.gif — the same sequence, animated (shareable on its own).
3. clincher_plot.png — debiased bird/cat/snake mean threshold across the 4
   scratch seeds and the pretrained model, visualizing the tight scratch-only
   pattern (Friedman p=.018) breaking down once pretrained.

Usage:
    .venv/bin/python make_blog_visuals.py
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from rise import generate_rise_sequence, N_STEPS

DATA_PATH = Path("data/sdh_500_clean")
RESULTS = Path("results")


def make_rise_visuals():
    # Pick a representative snake image from the validation set.
    snake_dir = DATA_PATH / "valid" / "snake"
    img_path = snake_dir / "snake_0005.jpg"  # checked visually: unambiguous full-body snake shot
    frames = generate_rise_sequence(img_path, seed=42)

    # --- Contact sheet: all 20 steps in a row ---
    fig, axes = plt.subplots(1, N_STEPS, figsize=(N_STEPS * 1.1, 1.6))
    for i, (ax, frame) in enumerate(zip(axes, frames)):
        ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
    fig.suptitle("RISE: step 1 (95% scrambled) → step 20 (original)", y=1.05, fontsize=11)
    fig.tight_layout()
    out_path = RESULTS / "rise_contact_sheet.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}  (source: {img_path.name})")

    # --- Animated GIF ---
    gif_frames = [f.convert("RGB").resize((300, 300)) for f in frames]
    # hold on the final (clearest) frame a bit longer, then loop
    durations = [120] * (N_STEPS - 1) + [1200]
    gif_path = RESULTS / "rise_sequence.gif"
    gif_frames[0].save(
        gif_path, save_all=True, append_images=gif_frames[1:],
        duration=durations, loop=0,
    )
    print(f"Saved -> {gif_path}")


def make_clincher_plot():
    # Debiased seed-level means (bird, cat, snake), scratch + pretrained.
    conditions = ["original", "seed1", "seed2", "seed3", "pretrained"]
    data = {
        "bird":  [11.06,  9.43, 10.33,  9.89, 9.68],
        "cat":   [11.88, 13.32, 10.91, 11.67, 9.82],
        "snake": [12.73, 17.68, 12.19, 14.74, 9.30],
    }
    colors = {"bird": "#4C72B0", "cat": "#DD8452", "snake": "#C44E52"}
    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, cls in enumerate(["bird", "cat", "snake"]):
        ax.bar(x + (i - 1) * width, data[cls], width, label=cls, color=colors[cls])

    ax.axvline(3.5, color="gray", linestyle="--", linewidth=1)
    ax.text(3.5, ax.get_ylim()[1] * 0.97, "  scratch  |  pretrained  ",
            ha="center", va="top", fontsize=9, color="gray",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none"))

    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylabel("Debiased detection threshold (step)\nlower = detected earlier")
    ax.set_title("Snake is consistently hardest to detect in every scratch-trained\n"
                  "model (4/4, p=.018) — but the pattern vanishes once pretrained")
    ax.legend()
    fig.tight_layout()
    out_path = RESULTS / "clincher_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    make_rise_visuals()
    make_clincher_plot()
