#!/usr/bin/env python3
"""
Automatically classify road line colors.

Uses SegFormer to identify the road surface (class 0), then applies HSV
color analysis on those pixels to detect yellow vs white lane markings.

Output: two categories — 'yellow_lines' or 'white_lines' (or 'unclassified').
"""

import cv2
import numpy as np
import json
from pathlib import Path
import argparse
from tqdm import tqdm


class RoadLineClassifier:
    """Classify road line colours using SegFormer road mask + HSV analysis."""

    # HSV color ranges for line detection
    # Yellow: Hue 20-40, Saturation 100-255, Value 100-255
    YELLOW_LOWER = np.array([20, 100, 100])
    YELLOW_UPPER = np.array([40, 255, 255])

    # White: Hue any, Saturation 0-30, Value 200-255
    WHITE_LOWER = np.array([0, 0, 200])
    WHITE_UPPER = np.array([180, 30, 255])

    # Cityscapes class IDs (standard 19-class set)
    ROAD_CLASS_ID = 0

    def __init__(self, use_segmentation=True):
        """
        Initialize classifier.

        Args:
            use_segmentation: If True, use SegFormer road mask (class 0).
                            If False, fall back to simple bottom 40% method.
        """
        self.use_segmentation = use_segmentation
        if use_segmentation:
            from detect_road_markings import RoadMarkingDetector
            print("Loading SegFormer model for road segmentation...")
            self.segmenter = RoadMarkingDetector()
        else:
            self.segmenter = None
            print("Using simple bottom-40% method (no segmentation)")

    def detect_line_colors(self, image, seg_map=None):
        """
        Detect yellow and white pixels within the road surface.

        Args:
            image: Input image (BGR)
            seg_map: Optional segmentation map from SegFormer

        Returns:
            Dictionary with color detection results including masks
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(hsv, self.YELLOW_LOWER, self.YELLOW_UPPER)
        white_mask  = cv2.inRange(hsv, self.WHITE_LOWER,  self.WHITE_UPPER)

        if seg_map is not None:
            # Restrict to road surface pixels (class 0)
            road_mask = (seg_map == self.ROAD_CLASS_ID).astype(np.uint8) * 255
            yellow_mask = cv2.bitwise_and(yellow_mask, road_mask)
            white_mask  = cv2.bitwise_and(white_mask,  road_mask)
            total_pixels = int(np.sum(road_mask > 0))
        else:
            # Fallback: use bottom 40% of image
            h, w = image.shape[:2]
            start_row = int(h * 0.6)
            yellow_mask[:start_row, :] = 0
            white_mask[:start_row, :] = 0
            total_pixels = (h - start_row) * w

        yellow_pixels = int(np.sum(yellow_mask > 0))
        white_pixels  = int(np.sum(white_mask  > 0))

        yellow_pct = (yellow_pixels / total_pixels * 100) if total_pixels > 0 else 0
        white_pct  = (white_pixels  / total_pixels * 100) if total_pixels > 0 else 0

        return {
            'yellow_pixels': yellow_pixels,
            'white_pixels':  white_pixels,
            # pct_of_road: fraction of total road pixels that are this colour
            'yellow_pct_of_road': float(yellow_pct),
            'white_pct_of_road':  float(white_pct),
            'total_road_pixels': total_pixels,
            'yellow_mask': yellow_mask,
            'white_mask':  white_mask,
        }

    def classify_image(self, image_path, min_road_pixels=1000,
                       dominant_color_ratio=0.85, min_colored_pct_of_road=1.0):
        """
        Classify road line colour in image.

        Args:
            image_path: Path to image
            min_road_pixels: Minimum road pixels required to classify at all
            dominant_color_ratio: One colour must be >= this fraction of all
                                  coloured pixels to be the sole classification (default 0.85)
            min_colored_pct_of_road: Minimum coloured pixels as % of road pixels (default 1.0%)

        Returns:
            Classification results dict. line_color is one of:
              'yellow_lines', 'white_lines', 'both_lines', 'unclassified'
        """
        image = cv2.imread(str(image_path))
        if image is None:
            return {'success': False, 'error': 'Could not load image'}

        seg_map = None
        if self.use_segmentation and self.segmenter is not None:
            try:
                seg_map, _ = self.segmenter.segment_image(image_path)
            except Exception as e:
                print(f"Warning: Segmentation failed for {image_path}: {e}")

        color_results = self.detect_line_colors(image, seg_map)

        yellow_pixels = color_results['yellow_pixels']
        white_pixels  = color_results['white_pixels']
        total_road    = color_results['total_road_pixels']
        colored_pixels = yellow_pixels + white_pixels
        colored_pct_of_road = (colored_pixels / total_road * 100) if total_road > 0 else 0

        # Not enough road pixels or coloured pixels are less than 1% of road surface
        if total_road < min_road_pixels or colored_pct_of_road < min_colored_pct_of_road:
            return {
                'success': True,
                'image_path': str(image_path),
                'line_color': 'unclassified',
                'yellow_pixels': yellow_pixels,
                'white_pixels':  white_pixels,
                'yellow_pct_of_road': color_results['yellow_pct_of_road'],
                'white_pct_of_road':  color_results['white_pct_of_road'],
                'yellow_share': 0.0,
                'white_share':  0.0,
                'used_segmentation': seg_map is not None,
                'reason': 'insufficient_road_or_color_pixels',
            }

        # Zero out any individual colour below the per-colour threshold
        # (e.g. 0.38% yellow shouldn't count as "yellow lines present")
        yellow_pct = color_results['yellow_pct_of_road']
        white_pct  = color_results['white_pct_of_road']
        if yellow_pct < min_colored_pct_of_road:
            yellow_pixels = 0
        if white_pct < min_colored_pct_of_road:
            white_pixels = 0
        colored_pixels = yellow_pixels + white_pixels

        # If zeroing colours left nothing, unclassified
        if colored_pixels == 0:
            return {
                'success': True,
                'image_path': str(image_path),
                'line_color': 'unclassified',
                'yellow_pixels': color_results['yellow_pixels'],
                'white_pixels':  color_results['white_pixels'],
                'yellow_pct_of_road': yellow_pct,
                'white_pct_of_road':  white_pct,
                'yellow_share': 0.0,
                'white_share':  0.0,
                'used_segmentation': seg_map is not None,
                'reason': 'no_color_meets_per_color_threshold',
            }

        yellow_ratio = yellow_pixels / colored_pixels
        white_ratio  = white_pixels  / colored_pixels

        if yellow_ratio >= dominant_color_ratio:
            line_color = 'yellow_lines'
        elif white_ratio >= dominant_color_ratio:
            line_color = 'white_lines'
        else:
            line_color = 'both_lines'

        return {
            'success': True,
            'image_path': str(image_path),
            'line_color': line_color,
            'yellow_pixels': yellow_pixels,
            'white_pixels':  white_pixels,
            # pct_of_road: % of road surface pixels that are this colour (small numbers, e.g. 2%)
            'yellow_pct_of_road': color_results['yellow_pct_of_road'],
            'white_pct_of_road':  color_results['white_pct_of_road'],
            # share: fraction of (yellow+white) pixels that are this colour (what dominance is based on)
            'yellow_share': float(yellow_ratio * 100),
            'white_share':  float(white_ratio  * 100),
            'used_segmentation': seg_map is not None,
            'reason': '',
        }


def classify_from_json(json_file, output_file, n_sample=None, use_segmentation=True):
    """
    Classify road line colours from a detection results JSON.

    Args:
        json_file: Path to road marking detections JSON
        output_file: Path to save classification results
        n_sample: Number of images to process (None = all)
        use_segmentation: Whether to use SegFormer road mask
    """
    print(f"Loading detections from {json_file}...")
    with open(json_file) as f:
        data = json.load(f)

    images_with_markings = data['images_with_markings']
    print(f"Found {len(images_with_markings)} images with road markings")

    if n_sample and n_sample < len(images_with_markings):
        import random
        random.seed(42)
        images_with_markings = random.sample(images_with_markings, n_sample)
        print(f"Sampling {n_sample} images")

    classifier = RoadLineClassifier(use_segmentation=use_segmentation)

    results = []
    print(f"\nClassifying {len(images_with_markings)} images...")
    for img_data in tqdm(images_with_markings):
        results.append(classifier.classify_image(img_data['image_path']))

    # Statistics
    total = len([r for r in results if r.get('success')])
    yellow_count = len([r for r in results if r.get('line_color') == 'yellow_lines'])
    white_count  = len([r for r in results if r.get('line_color') == 'white_lines'])
    both_count   = len([r for r in results if r.get('line_color') == 'both_lines'])
    unclass_count = len([r for r in results if r.get('line_color') == 'unclassified'])
    seg_count = len([r for r in results if r.get('used_segmentation')])

    summary = {
        'total_classified': total,
        'yellow_lines': yellow_count,
        'white_lines':  white_count,
        'both_lines':   both_count,
        'unclassified': unclass_count,
        'yellow_pct': yellow_count / total * 100 if total else 0,
        'white_pct':  white_count  / total * 100 if total else 0,
        'both_pct':   both_count   / total * 100 if total else 0,
        'used_segmentation': use_segmentation,
        'segmentation_success_count': seg_count,
    }

    with open(output_file, 'w') as f:
        json.dump({'summary': summary, 'classifications': results}, f, indent=2)

    print(f"\n{'='*60}")
    print("CLASSIFICATION SUMMARY")
    print(f"{'='*60}")
    print(f"Method: {'SegFormer road mask + HSV' if use_segmentation else 'Bottom-40% + HSV'}")
    print(f"Segmentation success: {seg_count}/{total}")
    print(f"Total classified: {total}")
    print(f"  yellow_lines:  {yellow_count} ({summary['yellow_pct']:.1f}%)")
    print(f"  white_lines:   {white_count}  ({summary['white_pct']:.1f}%)")
    print(f"  both_lines:    {both_count}   ({summary['both_pct']:.1f}%)")
    print(f"  unclassified:  {unclass_count}")
    print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Classify road line colours (yellow vs white) using SegFormer + HSV'
    )
    parser.add_argument('--input',  default='road_markings_1k.json')
    parser.add_argument('--output', default='road_line_classifications.json')
    parser.add_argument('--n-sample', type=int, default=None)
    parser.add_argument('--no-segmentation', action='store_true')
    args = parser.parse_args()

    classify_from_json(
        json_file=args.input,
        output_file=args.output,
        n_sample=args.n_sample,
        use_segmentation=not args.no_segmentation,
    )
    print("\nClassification complete!")


if __name__ == "__main__":
    main()
