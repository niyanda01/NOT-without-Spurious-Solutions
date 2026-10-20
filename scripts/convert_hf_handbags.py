"""Rebuild data/edges2handbags/{train,val}/*_AB.jpg from the Hugging Face
mirror luethan2025/edges2handbags, for when the Berkeley server is too slow.

The mirror stores the edge (imageA) and photo (imageB) halves as separate
256x256 JPEGs; they are pasted back into one 512x256 edge|photo pair and
re-encoded as JPEG (quality 95, no chroma subsampling), so the result drops
into Pix2PixProductPhotoDataset like the original archive.

Needs pyarrow (pip install pyarrow). Usage:
    curl -L -O https://huggingface.co/datasets/luethan2025/edges2handbags/resolve/main/data/<shard>.parquet
    python scripts/convert_hf_handbags.py --parquet-dir data/hf_edges2handbags
"""
import argparse
import io
import os
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def convert_shard(job):
    parquet_path, out_root = job
    split = "val" if parquet_path.name.startswith("val") else "train"
    out_dir = out_root / split
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=512):
        for row in batch.to_pylist():
            edge = Image.open(io.BytesIO(row["imageA"]["bytes"])).convert("RGB")
            photo = Image.open(io.BytesIO(row["imageB"]["bytes"])).convert("RGB")
            pair = Image.new("RGB", (edge.width + photo.width, edge.height))
            pair.paste(edge, (0, 0))
            pair.paste(photo, (edge.width, 0))
            pair.save(out_dir / row["imageA"]["path"], quality=95, subsampling=0)
            count += 1
    return parquet_path.name, split, count


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parquet-dir", default=REPO_ROOT / "data" / "hf_edges2handbags", type=Path)
    parser.add_argument("--out-dir", default=REPO_ROOT / "data" / "edges2handbags", type=Path)
    args = parser.parse_args()

    shards = sorted(args.parquet_dir.glob("*.parquet"))
    if not shards:
        raise SystemExit(f"No parquet shards in {args.parquet_dir}")
    with Pool(min(len(shards), os.cpu_count() or 1)) as pool:
        for name, split, count in pool.imap_unordered(
                convert_shard, [(shard, args.out_dir) for shard in shards]):
            print(f"{name}: {count:,} -> {split}/", flush=True)
    (args.out_dir / "SOURCE.txt").write_text(
        "Converted from the Hugging Face mirror luethan2025/edges2handbags\n"
        "(https://huggingface.co/datasets/luethan2025/edges2handbags) by\n"
        "scripts/convert_hf_handbags.py, not from the Berkeley pix2pix archive.\n"
        "Halves re-joined and re-encoded as JPEG q95 (4:4:4).\n")


if __name__ == "__main__":
    main()
