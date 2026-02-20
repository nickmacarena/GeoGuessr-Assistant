#!/usr/bin/env python3
"""
Convert MTSD fully-annotated dataset to YOLO format for binary sign detection.

Reads annotation JSONs + official train/val splits → writes:
  data/mapillary/yolo_detection/
    images/train/   (symlinks to full_ds/images/)
    images/val/
    labels/train/   (YOLO .txt: "0 cx cy w h" per sign, normalized)
    labels/val/
    dataset.yaml

Filtering applied:
  - Skip images flagged as panoramic (ispano=true)
  - Skip boxes with out-of-frame=true or ambiguous=true
  - All remaining signs → class 0 (binary: sign / no-sign)
  - Images with no valid boxes get an empty label file (negatives)

Usage:
    python prepare_detection_data.py
    python prepare_detection_data.py --splits train val   # only specific splits
    python prepare_detection_data.py --no-symlinks        # copy instead of symlink
"""

import argparse
import json
import os
import yaml
from pathlib import Path
from tqdm import tqdm


# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent.parent
DATA_ROOT   = REPO_ROOT / "data" / "mapillary"
FULL_DS     = DATA_ROOT / "full_ds"
IMAGES_DIR  = FULL_DS / "images"
ANN_DIR     = FULL_DS / "mtsd_v2_fully_annotated" / "annotations"
SPLITS_DIR  = FULL_DS / "mtsd_v2_fully_annotated" / "splits"
OUT_DIR     = DATA_ROOT / "yolo_detection"


def load_split(split_name):
    """Load image keys from a split .txt file."""
    path = SPLITS_DIR / f"{split_name}.txt"
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def load_annotation(image_key):
    """Load and return the annotation dict for an image key, or None."""
    path = ANN_DIR / f"{image_key}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def boxes_to_yolo(ann):
    """
    Convert MTSD annotation dict to YOLO-format lines.

    Returns list of "0 cx cy w h" strings (normalized to [0,1]).
    Filters out out-of-frame and ambiguous boxes.
    Returns empty list for panoramic images (caller should skip).
    """
    if ann.get("ispano", False):
        return None  # signal to skip this image entirely

    img_w = ann["width"]
    img_h = ann["height"]
    lines = []

    for obj in ann.get("objects", []):
        props = obj.get("properties", {})
        if props.get("out-of-frame", False):
            continue
        if props.get("ambiguous", False):
            continue

        bbox = obj["bbox"]
        xmin, ymin, xmax, ymax = bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]

        # Clamp to image bounds
        xmin = max(0.0, xmin)
        ymin = max(0.0, ymin)
        xmax = min(float(img_w), xmax)
        ymax = min(float(img_h), ymax)

        if xmax <= xmin or ymax <= ymin:
            continue  # degenerate box after clamping

        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        w  = (xmax - xmin) / img_w
        h  = (ymax - ymin) / img_h

        lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

    return lines


def prepare_split(split_name, use_symlinks=True):
    """Process one split (train/val/test) and write images + labels."""
    image_keys = load_split(split_name)

    img_out_dir   = OUT_DIR / "images" / split_name
    label_out_dir = OUT_DIR / "labels" / split_name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    label_out_dir.mkdir(parents=True, exist_ok=True)

    stats = {"total": 0, "skipped_pano": 0, "skipped_no_ann": 0,
             "negatives": 0, "boxes": 0}

    for key in tqdm(image_keys, desc=f"  {split_name}", unit="img"):
        stats["total"] += 1
        src_img = IMAGES_DIR / f"{key}.jpg"

        # Skip if image file doesn't exist on disk
        if not src_img.exists():
            stats["skipped_no_ann"] += 1
            continue

        ann = load_annotation(key)
        if ann is None:
            stats["skipped_no_ann"] += 1
            continue

        lines = boxes_to_yolo(ann)
        if lines is None:
            stats["skipped_pano"] += 1
            continue

        # Image: symlink or copy
        dst_img = img_out_dir / f"{key}.jpg"
        if not dst_img.exists():
            if use_symlinks:
                dst_img.symlink_to(src_img.resolve())
            else:
                import shutil
                shutil.copy2(src_img, dst_img)

        # Label file (empty = negative image, still useful for training)
        label_path = label_out_dir / f"{key}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
            if lines:
                f.write("\n")

        if not lines:
            stats["negatives"] += 1
        else:
            stats["boxes"] += len(lines)

    return stats


def write_dataset_yaml():
    """Write dataset.yaml for ultralytics YOLO."""
    cfg = {
        "path": str(OUT_DIR.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["sign"],
    }
    out = OUT_DIR / "dataset.yaml"
    with open(out, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"  Wrote {out}")


def main():
    parser = argparse.ArgumentParser(description="Prepare MTSD → YOLO detection dataset")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=["train", "val", "test"],
                        help="Which splits to process (default: train val)")
    parser.add_argument("--no-symlinks", action="store_true",
                        help="Copy images instead of symlinking (slower, uses more disk)")
    args = parser.parse_args()

    use_symlinks = not args.no_symlinks
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {OUT_DIR}")
    print(f"Mode: {'symlinks' if use_symlinks else 'copies'}")
    print()

    all_stats = {}
    for split in args.splits:
        print(f"Processing split: {split}")
        stats = prepare_split(split, use_symlinks=use_symlinks)
        all_stats[split] = stats
        valid = stats["total"] - stats["skipped_pano"] - stats["skipped_no_ann"]
        print(f"    {stats['total']:,} images in split")
        print(f"    {stats['skipped_no_ann']:,} skipped (missing image/annotation)")
        print(f"    {stats['skipped_pano']:,} skipped (panoramic)")
        print(f"    {valid:,} written  ({stats['negatives']:,} negatives, {stats['boxes']:,} boxes)")
        print()

    write_dataset_yaml()

    print("\nDone. Directory layout:")
    for split in args.splits:
        n_labels = len(list((OUT_DIR / "labels" / split).glob("*.txt")))
        n_images = len(list((OUT_DIR / "images" / split).iterdir()))
        print(f"  {split}: {n_images:,} images, {n_labels:,} label files")

    print(f"\nNext step:")
    print(f"  python scripts/training/train_sign_detector.py")


if __name__ == "__main__":
    main()
