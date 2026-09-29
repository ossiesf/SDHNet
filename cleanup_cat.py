"""
cleanup_cat.py
--------------
Approach 1: surfaces suspect cat training images using the trained gray model's
own confidence scores. Images the model can't fit (prob < threshold) are likely
mislabeled, non-photo, or wildly out-of-distribution.

Run this from the sdhnet/ directory with your local fastai environment:
    cd ~/Documents/Claude/Projects/SDH/sdhnet
    python cleanup_cat.py

Outputs:
    data/suspect_cat_model.csv   — all training images ranked by model confidence
    data/suspect_cat_grid2.png   — visual grid of bottom 56 (model's worst)

After reviewing the grid, call prune_suspects() at the bottom of this file
or manually delete images from data/sdh_500/train/cat/ and re-run
prepare_dataset.py to rebuild the dataset.
"""

from fastai.vision.all import *
import pandas as pd
from PIL import Image, ImageDraw
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = Path('data/sdh_500')
MODEL_PATH  = Path('models/sdhnet-4class-gray-best.pth')
OUT_CSV     = Path('data/suspect_cat_model.csv')
OUT_GRID    = Path('data/suspect_cat_grid2.png')
PROB_THRESH = 0.25   # flag images where model gives < 25% to true label
TOP_N       = 56     # images to show in the grid

# ── Load model ────────────────────────────────────────────────────────────────
dblock = DataBlock(
    blocks     = (ImageBlock(cls=PILImageBW), CategoryBlock),
    get_items  = get_image_files,
    get_y      = parent_label,
    splitter   = GrandparentSplitter(train_name='train', valid_name='valid'),
    item_tfms  = RandomResizedCrop(224, min_scale=0.75),
    batch_tfms = [Flip(), Brightness(max_lighting=0.2),
                  Normalize.from_stats([0.449], [0.226])]
)
dls = dblock.dataloaders(DATA_PATH, batch_size=32)

learn = vision_learner(dls, resnet18, metrics=[accuracy], pretrained=False, n_in=1)
learn.load(MODEL_PATH.stem)
print(f'Loaded model from {MODEL_PATH}')

# ── Run inference on training set ─────────────────────────────────────────────
# Important: shuffle=False, drop_last=False to keep 1-to-1 correspondence
train_dl = dls.train.new(shuffle=False, drop_last=False)
preds, targets = learn.get_preds(dl=train_dl)
probs = preds.softmax(dim=-1)
prob_true = probs.gather(1, targets.unsqueeze(1)).squeeze()

# Map back to file paths
items = train_dl.items  # fastai DataLoader exposes .items

df = pd.DataFrame({
    'path':       [str(p) for p in items],
    'filename':   [Path(p).name for p in items],
    'true_label': [dls.vocab[t] for t in targets.tolist()],
    'prob_true':  prob_true.numpy().round(4),
    'pred_label': [dls.vocab[p.argmax()] for p in probs],
})

df.sort_values('prob_true', inplace=True)
df.to_csv(OUT_CSV, index=False)
print(f'\nAll training images ranked → {OUT_CSV}')

# ── Summary ───────────────────────────────────────────────────────────────────
cat_df = df[df['true_label'] == 'cat']
flagged = cat_df[cat_df['prob_true'] < PROB_THRESH]
print(f'\nCat images with model confidence < {PROB_THRESH}: {len(flagged)} / {len(cat_df)}')
print(f'\nBottom 20 cat images by model confidence:')
print(cat_df.head(20)[['filename', 'prob_true', 'pred_label']].to_string(index=False))

# ── Visual grid of worst cat images ──────────────────────────────────────────
worst_cat = cat_df.head(TOP_N)

THUMB = 112
COLS  = 8
rows  = (len(worst_cat) + COLS - 1) // COLS
W = COLS * (THUMB + 4) + 4
H = rows  * (THUMB + 22) + 4
canvas = Image.new('RGB', (W, H), (30, 30, 30))
draw   = ImageDraw.Draw(canvas)

for i, (_, row) in enumerate(worst_cat.iterrows()):
    c = i % COLS
    r = i // COLS
    x = 4 + c * (THUMB + 4)
    y = 4 + r * (THUMB + 22)

    try:
        thumb = Image.open(row['path']).convert('RGB').resize((THUMB, THUMB))
    except:
        thumb = Image.new('RGB', (THUMB, THUMB), (80, 80, 80))

    # Border: red if predicted wrong, orange if correct but low confidence
    if row['pred_label'] != row['true_label']:
        border = (220, 50, 50)   # red: misclassified
    else:
        border = (220, 140, 40)  # orange: correct but uncertain

    b = Image.new('RGB', (THUMB + 2, THUMB + 2), border)
    b.paste(thumb, (1, 1))
    canvas.paste(b, (x, y))

    label = f"{row['filename'][:11]} {row['prob_true']:.2f}"
    pred  = row['pred_label'][:3] if row['pred_label'] != 'cat' else ''
    draw.text((x, y + THUMB + 3), f"{label} {pred}", fill=(200, 200, 200))

canvas.save(OUT_GRID)
print(f'\nVisual grid → {OUT_GRID}')
print('  Red border  = model predicted wrong class (label shown)')
print('  Orange border = correct prediction but low confidence')

# ── Optional: auto-prune the very worst ───────────────────────────────────────
def prune_suspects(threshold=0.10, dry_run=True):
    """
    Move cat images where prob_true < threshold to data/_to_delete/cat/.
    Set dry_run=False to actually move them.
    """
    to_prune = cat_df[cat_df['prob_true'] < threshold]
    dest = Path('data/_to_delete/cat')
    dest.mkdir(parents=True, exist_ok=True)
    print(f'\n{"[DRY RUN] " if dry_run else ""}Pruning {len(to_prune)} cat images with prob < {threshold}:')
    for _, row in to_prune.iterrows():
        src = Path(row['path'])
        print(f'  {src.name}  (prob={row["prob_true"]:.3f}, pred={row["pred_label"]})')
        if not dry_run:
            src.rename(dest / src.name)
    if not dry_run:
        print(f'Moved to {dest}. Re-run prepare_dataset.py to rebuild.')

# Dry run by default — inspect the grid first, then call with dry_run=False
prune_suspects(threshold=0.10, dry_run=True)

print('\nDone. Review the grid, then call prune_suspects(dry_run=False) to move junk files.')
print('After pruning, re-run prepare_dataset.py and retrain.')
