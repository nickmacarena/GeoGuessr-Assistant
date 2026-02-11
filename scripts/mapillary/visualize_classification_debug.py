#!/usr/bin/env python3
"""
Diagnostic visualization showing WHY an image was classified as yellow/white/unclassified.

For each image, produces a 2x2 panel:
  [Original]               [SegFormer class 0 road overlay]
  [HSV colour masks]       [Text summary]
"""

import cv2
import numpy as np
import json
import argparse
import random
from pathlib import Path

YELLOW_LOWER = np.array([20, 100, 100])
YELLOW_UPPER = np.array([40, 255, 255])
WHITE_LOWER  = np.array([0,  0,   200])
WHITE_UPPER  = np.array([180, 30, 255])

ROAD_CLASS_ID = 0


def run_segformer(image_path, detector):
    seg_map, _ = detector.segment_image(image_path)
    return seg_map


def build_debug_image(image_path, classification, seg_map, target_h=400):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        return None

    H, W = img_bgr.shape[:2]
    scale = target_h / H
    tW = int(W * scale)
    tH = target_h

    def rs(im):
        return cv2.resize(im, (tW, tH))

    # ── Panel 0: original ────────────────────────────────────────────────────
    p0 = rs(img_bgr.copy())
    _label(p0, "Original")

    # ── Panel 1: SegFormer road overlay ──────────────────────────────────────
    seg_vis = np.zeros((H, W, 3), dtype=np.uint8)
    if seg_map is not None:
        seg_vis[seg_map == ROAD_CLASS_ID] = (128, 64, 128)   # road - purple
        seg_vis[seg_map != ROAD_CLASS_ID] = (50, 50, 50)     # other - dark grey

    p1 = rs(cv2.addWeighted(img_bgr, 0.35, seg_vis, 0.65, 0))
    cv2.rectangle(p1, (5, tH-35), (145, tH-5), (0, 0, 0), -1)
    cv2.rectangle(p1, (8, tH-32), (22, tH-12), (128, 64, 128), -1)
    cv2.putText(p1, "class 0 road", (26, tH-14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    _label(p1, "SegFormer road mask")

    # ── Panel 2: HSV colour masks within road ────────────────────────────────
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    white_mask  = cv2.inRange(hsv, WHITE_LOWER,  WHITE_UPPER)

    if seg_map is not None:
        road_mask = (seg_map == ROAD_CLASS_ID).astype(np.uint8) * 255
        yellow_mask = cv2.bitwise_and(yellow_mask, road_mask)
        white_mask  = cv2.bitwise_and(white_mask,  road_mask)

    overlay = img_bgr.copy()
    overlay[yellow_mask > 0] = (0, 0, 255)      # bright red
    overlay[white_mask  > 0] = (255, 0, 0)      # bright blue

    p2 = rs(cv2.addWeighted(img_bgr, 0.5, overlay, 0.5, 0))
    y_px = int(np.sum(yellow_mask > 0))
    w_px = int(np.sum(white_mask  > 0))
    cv2.putText(p2, f"Y:{y_px}px  W:{w_px}px", (8, tH - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    _label(p2, "HSV: yellow / white on road")

    # ── Panel 3: text summary ─────────────────────────────────────────────────
    p3 = _build_summary_panel(classification, tW, tH)

    row0 = np.hstack([p0, p1])
    row1 = np.hstack([p2, p3])
    return np.vstack([row0, row1])


def _label(img, text, pos=(8, 22)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)


def _build_summary_panel(c, W, H):
    panel = np.zeros((H, W, 3), dtype=np.uint8)
    panel[:] = (20, 20, 20)

    line_color = c.get('line_color', '?')
    header_colour = {
        'yellow_lines': (0, 200, 255),
        'white_lines':  (200, 200, 255),
        'both_lines':   (0, 200, 100),
        'unclassified': (80, 80, 80),
    }.get(line_color, (150, 150, 150))

    cv2.rectangle(panel, (0, 0), (W, 28), header_colour, -1)
    _label(panel, f"  {line_color.upper()}", (6, 20))

    lines = [
        f"Used seg: {c.get('used_segmentation', False)}",
        "",
        f"Yellow px:     {c.get('yellow_pixels', 0)}",
        f"  of road:     {c.get('yellow_pct_of_road', 0):.2f}%",
        f"  share:       {c.get('yellow_share', 0):.1f}%",
        "",
        f"White px:      {c.get('white_pixels', 0)}",
        f"  of road:     {c.get('white_pct_of_road', 0):.2f}%",
        f"  share:       {c.get('white_share', 0):.1f}%",
        "",
        f"Reason: {c.get('reason', '') or 'classified'}",
    ]

    y = 50
    for line in lines:
        cv2.putText(panel, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
        y += 22
        if y > H - 10:
            break

    return panel


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic viz: why was each road classified yellow/white/unclassified?"
    )
    parser.add_argument('--input', default='road_line_classifications_confident_full.json')
    parser.add_argument('--output', default='debug_viz')
    parser.add_argument('--categories', nargs='+',
                        default=['yellow_lines', 'white_lines', 'both_lines', 'unclassified'])
    parser.add_argument('--n-per-category', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"Loading {args.input}...")
    with open(args.input) as f:
        data = json.load(f)
    classifications = data['classifications']

    print("Loading SegFormer model...")
    from detect_road_markings import RoadMarkingDetector
    detector = RoadMarkingDetector()

    out_dir = Path(args.output)
    out_dir.mkdir(exist_ok=True)

    for cat in args.categories:
        subset = [c for c in classifications
                  if c.get('success') and c.get('line_color') == cat]
        if not subset:
            print(f"  No images for category: {cat}")
            continue

        samples = random.sample(subset, min(args.n_per_category, len(subset)))
        print(f"\n[{cat}] {len(subset)} total → showing {len(samples)}")

        for i, c in enumerate(samples, 1):
            img_path = c['image_path']
            print(f"  {i}/{len(samples)} {Path(img_path).name} ...", end=' ', flush=True)

            try:
                seg_map = run_segformer(img_path, detector)
            except Exception as e:
                print(f"seg failed: {e}")
                seg_map = None

            debug_img = build_debug_image(img_path, c, seg_map)
            if debug_img is None:
                print("image load failed")
                continue

            out_file = out_dir / f"{cat}_{i}_{Path(img_path).stem}.jpg"
            cv2.imwrite(str(out_file), debug_img)
            print(f"saved → {out_file.name}")

    print(f"\nDone. All debug images saved to {out_dir}/")


if __name__ == "__main__":
    main()
