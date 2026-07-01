"""Download Habitat navigation episode datasets into sharedata.

Examples:
  python scripts/download_nav_datasets.py --dataset objectnav_hm3d
  python scripts/download_nav_datasets.py --dataset objectnav_mp3d
  python scripts/download_nav_datasets.py --dataset all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.nav_datasets import DATASET_SPECS, dataset_dir, resolve_data_root


def run(cmd):
    print("+ " + " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True)


def download_and_extract(dataset_name, data_root):
    spec = DATASET_SPECS[dataset_name]
    root = resolve_data_root(data_root)
    root.mkdir(parents=True, exist_ok=True)
    archive_path = root / spec.archive_name
    out_dir = dataset_dir(spec, data_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    run(["wget", "-c", spec.url, "-O", str(archive_path)])
    print(f"Extracting {archive_path} -> {out_dir}")
    with zipfile.ZipFile(archive_path) as zf:
        zf.extractall(out_dir)
    nested = out_dir / archive_path.stem
    if nested.is_dir():
        for child in nested.iterdir():
            target = out_dir / child.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(child), str(target))
        nested.rmdir()
    print(f"Done: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", choices=["all", *sorted(DATASET_SPECS)])
    parser.add_argument("--data-root", default="sharedata")
    args = parser.parse_args()

    names = sorted(DATASET_SPECS) if args.dataset == "all" else [args.dataset]
    for name in names:
        download_and_extract(name, args.data_root)


if __name__ == "__main__":
    main()
