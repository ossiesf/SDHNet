"""
download_more_images.py
-----------------------
Downloads additional images for the categories that are under-represented
(snake, cat, fish) to bring all classes up to ~TARGET images before
prepare_dataset.py is re-run.

Downloads go into data/downloaded/<class>/ so the original extracted data
is never touched. prepare_dataset.py pools all sources together.

Usage:
    python download_more_images.py
"""

import os
import logging
from pathlib import Path

# Suppress icrawler's verbose logging
logging.getLogger("icrawler").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from icrawler.builtin import BingImageCrawler

BASE = Path(__file__).parent / "data"
OUT  = BASE / "downloaded"

# How many to download per query. We run multiple queries per class and
# deduplicate, so set this comfortably above what you actually need.
TARGET_PER_CLASS = 200

DOWNLOAD_JOBS = {
    "snake": [
        "snake on ground wildlife",
        "wild snake close up",
        "snake coiled grass",
        "python snake nature",
        "rattlesnake desert",
        "garter snake leaves",
        "viper snake habitat",
    ],
    "cat": [
        "domestic cat portrait",
        "cat sitting outdoors",
        "tabby cat photo",
        "cat in nature",
        "kitten outside",
        "cat on grass",
        "wild cat photograph",
    ],
    "fish": [
        "fish underwater photo",
        "goldfish clear water",
        "tropical fish aquarium",
        "freshwater fish nature",
        "salmon swimming",
        "bass fish underwater",
        "trout river",
    ],
}

PER_QUERY = 120  # images per search query


def download_class(class_name: str, queries: list[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(out_dir.glob("*.*")))
    print(f"\n  {class_name}: {existing} already downloaded")

    for i, query in enumerate(queries):
        # icrawler numbers files sequentially per run, prefix by query index
        sub_dir = out_dir / f"q{i}"
        sub_dir.mkdir(exist_ok=True)

        crawler = BingImageCrawler(
            storage={"root_dir": str(sub_dir)},
            feeder_threads=1,
            parser_threads=1,
            downloader_threads=4,
        )
        crawler.crawl(
            keyword=query,
            max_num=PER_QUERY,
            min_size=(100, 100),
            file_idx_offset=0,
        )

    # Count total after all queries
    total = sum(len(list(d.glob("*.*"))) for d in out_dir.iterdir() if d.is_dir())
    print(f"  {class_name}: {total} images available after download")


def main():
    print(f"Downloading images to {OUT}\n{'─'*50}")
    for class_name, queries in DOWNLOAD_JOBS.items():
        download_class(class_name, queries, OUT / class_name)
    print(f"\n{'─'*50}")
    print("Done. Re-run prepare_dataset.py to rebuild the SDH dataset.")


if __name__ == "__main__":
    main()
