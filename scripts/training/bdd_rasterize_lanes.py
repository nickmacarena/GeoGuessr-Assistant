#!/usr/bin/env python3
"""
Rasterize BDD100K lane marking polylines into segmentation masks.

Converts poly2d annotations (with bezier curves) into pixel-level masks
where each pixel is labeled with its lane line class.

Scope: lane lines only (center and edge). Curb, crosswalk, and transverse
(vertical) markings are dropped — this model is purely for lane-line
classification across color/style/multiplicity, which is the geo signal.

Class mapping (9 classes):
    0 = background
    1 = single white solid
    2 = single white dashed
    3 = double white solid
    4 = double white dashed
    5 = single yellow solid
    6 = single yellow dashed
    7 = double yellow solid
    8 = double yellow dashed

Usage:
    python bdd_rasterize_lanes.py
    python bdd_rasterize_lanes.py --preview 5
    python bdd_rasterize_lanes.py --split val
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# ── Class mapping (category, style) → class_id ───────────────────────────────
# Only parallel-direction lane lines in these 4 categories are rasterized.
LANE_CLASSES = {
    ("single white",  "solid"):  1,
    ("single white",  "dashed"): 2,
    ("double white",  "solid"):  3,
    ("double white",  "dashed"): 4,
    ("single yellow", "solid"):  5,
    ("single yellow", "dashed"): 6,
    ("double yellow", "solid"):  7,
    ("double yellow", "dashed"): 8,
}

CLASS_NAMES = [
    "bg",
    "s_white_solid", "s_white_dashed",
    "d_white_solid", "d_white_dashed",
    "s_yellow_solid", "s_yellow_dashed",
    "d_yellow_solid", "d_yellow_dashed",
]

# Visualization colors (BGR for cv2.imwrite)
CLASS_COLORS = {
    0: (0, 0, 0),         # background
    1: (255, 255, 255),    # single white solid — bright white
    2: (160, 160, 160),    # single white dashed — gray
    3: (255, 200, 200),    # double white solid — light blue-white
    4: (180, 140, 140),    # double white dashed — dusty
    5: (0, 255, 255),      # single yellow solid — yellow
    6: (0, 180, 180),      # single yellow dashed — dark yellow / olive
    7: (0, 165, 255),      # double yellow solid — orange
    8: (0, 100, 180),      # double yellow dashed — dark orange
}

IMG_W, IMG_H = 1280, 720
LINE_THICKNESS = 8

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "bdd100k"
MASK_DIR = DATA_DIR / "lane_masks"


def bezier_cubic(p0, p1, p2, p3, n_steps=30):
    """Evaluate cubic bezier curve at n_steps points."""
    t = np.linspace(0, 1, n_steps)
    t2 = t * t
    t3 = t2 * t
    mt = 1 - t
    mt2 = mt * mt
    mt3 = mt2 * mt

    x = mt3 * p0[0] + 3 * mt2 * t * p1[0] + 3 * mt * t2 * p2[0] + t3 * p3[0]
    y = mt3 * p0[1] + 3 * mt2 * t * p1[1] + 3 * mt * t2 * p2[1] + t3 * p3[1]

    return np.column_stack([x, y]).astype(np.int32)


def polyline_to_points(poly2d):
    """Convert BDD100K poly2d (with L and C types) to dense point array."""
    points = []
    i = 0

    while i < len(poly2d):
        x, y, ptype = poly2d[i]

        if ptype == "L":
            points.append([int(x), int(y)])
            i += 1

        elif ptype == "C":
            c_points = []
            while i < len(poly2d) and poly2d[i][2] == "C":
                c_points.append((poly2d[i][0], poly2d[i][1]))
                i += 1

            if points:
                start = tuple(points[-1])
            else:
                start = c_points[0]
                c_points = c_points[1:]

            j = 0
            while j + 2 < len(c_points):
                curve_pts = bezier_cubic(
                    start, c_points[j], c_points[j + 1], c_points[j + 2]
                )
                points.extend(curve_pts.tolist())
                start = c_points[j + 2]
                j += 3

            while j < len(c_points):
                points.append([int(c_points[j][0]), int(c_points[j][1])])
                j += 1
        else:
            points.append([int(x), int(y)])
            i += 1

    return np.array(points, dtype=np.int32) if points else np.empty((0, 2), dtype=np.int32)


def rasterize_label(label_path):
    """Rasterize a single BDD100K label file into a segmentation mask."""
    with open(label_path) as f:
        data = json.load(f)

    mask = np.zeros((IMG_H, IMG_W), dtype=np.uint8)

    for frame in data.get("frames", []):
        for obj in frame.get("objects", []):
            cat = obj.get("category", "")
            if not cat.startswith("lane/"):
                continue

            lane_type = cat.replace("lane/", "")
            attrs = obj.get("attributes") or {}

            # Filter: parallel-direction lane lines only (drop transverse / stop lines)
            if attrs.get("direction") != "parallel":
                continue

            style = attrs.get("style")
            class_id = LANE_CLASSES.get((lane_type, style), 0)
            if class_id == 0:
                continue

            poly2d = obj.get("poly2d")
            if not poly2d:
                continue

            pts = polyline_to_points(poly2d)
            if len(pts) < 2:
                continue

            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(mask, [pts], isClosed=False, color=int(class_id),
                          thickness=LINE_THICKNESS)

    return mask


def visualize_mask(mask):
    """Convert class mask to color image for visualization."""
    vis = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    for class_id, color in CLASS_COLORS.items():
        vis[mask == class_id] = color
    return vis


def rasterize_split(split, preview=0):
    """Rasterize all labels in a split."""
    label_dir = DATA_DIR / "100k" / split
    if not label_dir.exists():
        print(f"  Split directory not found: {label_dir}")
        return

    out_dir = MASK_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(label_dir.glob("*.json"))
    print(f"  {split}: {len(files)} files → {out_dir}")

    class_pixel_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    previewed = 0

    for i, f in enumerate(files):
        name = f.stem
        mask = rasterize_label(f)

        cv2.imwrite(str(out_dir / f"{name}.png"), mask)

        # Tally per-class pixel counts for the frequency report
        for c in range(len(CLASS_NAMES)):
            class_pixel_counts[c] += int((mask == c).sum())

        if previewed < preview and mask.max() > 0:
            vis = visualize_mask(mask)
            preview_dir = MASK_DIR / "preview"
            preview_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(preview_dir / f"{name}_mask.png"), vis)
            previewed += 1

        if (i + 1) % 5000 == 0 or i == len(files) - 1:
            print(f"    [{i+1}/{len(files)}] done")

    print(f"  {split} class pixel frequencies:")
    total = class_pixel_counts.sum()
    for c, name in enumerate(CLASS_NAMES):
        pct = 100 * class_pixel_counts[c] / total if total else 0
        print(f"    {c} {name:18s}: {class_pixel_counts[c]:>14,} ({pct:6.3f}%)")


def main():
    parser = argparse.ArgumentParser(description="Rasterize BDD100K lane polylines")
    parser.add_argument("--split", choices=["train", "val", "test"])
    parser.add_argument("--preview", type=int, default=0)
    args = parser.parse_args()

    print(f"Data dir: {DATA_DIR}")
    print(f"Output:   {MASK_DIR}")
    print(f"Classes:  {len(CLASS_NAMES)} ({', '.join(CLASS_NAMES[1:])})")

    splits = [args.split] if args.split else ["train", "val"]
    for split in splits:
        rasterize_split(split, preview=args.preview)

    print("\nDone.")


if __name__ == "__main__":
    main()
