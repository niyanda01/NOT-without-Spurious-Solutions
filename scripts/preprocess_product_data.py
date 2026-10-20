"""Build the data/cache/ arrays used by run_handbags_to_shoes.py: every
edges2handbags / edges2shoes photo is background-masked, resized and reduced
to its kept channel once, instead of on every training batch.

Usage (from the repo root, after scripts/download_product_data.sh):
    python scripts/preprocess_product_data.py --workers 16
    python scripts/preprocess_product_data.py --keep-background
"""
import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from src.datasets import build_handbags_to_shoes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-root", default=os.path.join(REPO_ROOT, "data"))
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--keep-background", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--min-border-white", type=float, default=0.5)
    args = parser.parse_args()

    start = time.time()
    source, target = build_handbags_to_shoes(
        args.data_root, img_size=args.img_size,
        remove_background=not args.keep_background,
        use_cache=True, num_workers=args.workers,
        min_border_white=args.min_border_white)
    print(f"handbags: {len(source):,}  shoes: {len(target):,}  "
          f"({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
