#!/usr/bin/env python3
"""
Detect road markings in Mapillary images using SegFormer.

Uses pre-trained SegFormer model to segment roads and lane markings,
then filters images with visible road markings.
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
import json
from tqdm import tqdm
import argparse


class RoadMarkingDetector:
    """Detect road markings using SegFormer semantic segmentation."""

    # Cityscapes class IDs (standard 19-class set)
    ROAD_CLASS_ID = 0  # Road surface

    # HSV ranges for lane marking colours (same as classifier)
    YELLOW_LOWER = np.array([20, 100, 100])
    YELLOW_UPPER = np.array([40, 255, 255])
    WHITE_LOWER  = np.array([0,   0,  200])
    WHITE_UPPER  = np.array([180, 30, 255])

    def __init__(self, model_name="nvidia/segformer-b0-finetuned-cityscapes-1024-1024"):
        """
        Initialize SegFormer model.

        Args:
            model_name: HuggingFace model name (b0 is fastest, b5 is most accurate)
        """
        print(f"Loading SegFormer model: {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Load processor and model (use safetensors for safety)
        self.processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            use_safetensors=True
        )
        self.model.to(self.device)
        self.model.eval()

        print("Model loaded successfully!")

    def segment_image(self, image_path):
        """
        Segment image and return segmentation mask.

        Args:
            image_path: Path to image file

        Returns:
            numpy array with class IDs for each pixel
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Preprocess
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get segmentation map
        logits = outputs.logits
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False
        )

        # Get class predictions
        seg_map = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()

        return seg_map, image.size

    def _build_road_band_mask(self, seg_map, edge_fraction=0.15, center_fraction=0.25):
        """
        Build a mask covering the edge bands and center band of the road polygon.

        For each row, finds the road's left/right extent (class 0) and marks:
          - edge bands: inner edge_fraction of road width from each side
          - center band: middle (center_fraction .. 1-center_fraction) of road width

        Lane markings physically live on these bands, not the bulk road interior.

        Args:
            seg_map: SegFormer segmentation map
            edge_fraction: Fraction of road width to include from each edge (default 0.15)
            center_fraction: Inner boundary of center band (default 0.25 → 25-75%)

        Returns:
            uint8 mask (255 = band, 0 = outside band)
        """
        H, W = seg_map.shape
        road_mask = seg_map == self.ROAD_CLASS_ID
        band = np.zeros((H, W), dtype=np.uint8)

        for row in range(H):
            road_cols = np.where(road_mask[row])[0]
            if len(road_cols) < 10:
                continue
            rl, rr = int(road_cols[0]), int(road_cols[-1])
            rw = rr - rl
            if rw < 10:
                continue

            edge_w  = max(1, int(rw * edge_fraction))
            center_start = rl + int(rw * center_fraction)
            center_end   = rl + int(rw * (1.0 - center_fraction))

            # Left edge band
            band[row, rl : rl + edge_w] = 255
            # Right edge band
            band[row, rr - edge_w : rr + 1] = 255
            # Center band (only if it has meaningful width)
            if center_end > center_start:
                band[row, center_start : center_end] = 255

        return band

    def has_road_markings(self, seg_map, image_bgr,
                          min_road_pixels=5000, min_marking_pixels=50):
        """
        Check if image has visible road markings (white/yellow on road boundary bands).

        Args:
            seg_map: Segmentation map (numpy array)
            image_bgr: BGR image (numpy array) for HSV colour analysis
            min_road_pixels: Minimum road pixels required
            min_marking_pixels: Minimum coloured pixels in road bands required

        Returns:
            dict with detection results
        """
        road_pixels = int(np.sum(seg_map == self.ROAD_CLASS_ID))
        total_pixels = seg_map.size
        road_pct = (road_pixels / total_pixels) * 100

        has_road = road_pixels >= min_road_pixels
        if not has_road:
            return {
                'has_road': False,
                'has_markings': False,
                'road_pixels': road_pixels,
                'marking_pixels': 0,
                'road_percentage': float(road_pct),
                'marking_percentage': 0.0
            }

        # Build road band mask (edge + center strips of road polygon)
        band_mask = self._build_road_band_mask(seg_map)

        # HSV colour detection within bands
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        yellow_mask = cv2.bitwise_and(
            cv2.inRange(hsv, self.YELLOW_LOWER, self.YELLOW_UPPER), band_mask)
        white_mask = cv2.bitwise_and(
            cv2.inRange(hsv, self.WHITE_LOWER, self.WHITE_UPPER), band_mask)

        marking_pixels = int(np.sum(yellow_mask > 0) + np.sum(white_mask > 0))
        marking_pct = (marking_pixels / total_pixels) * 100
        has_markings = marking_pixels >= min_marking_pixels

        return {
            'has_road': True,
            'has_markings': bool(has_markings),
            'road_pixels': road_pixels,
            'marking_pixels': marking_pixels,
            'road_percentage': float(road_pct),
            'marking_percentage': float(marking_pct)
        }

    def process_image(self, image_path, min_road_pixels=5000, min_marking_pixels=50):
        """
        Process single image and detect road markings.

        Args:
            image_path: Path to image file
            min_road_pixels: Minimum road pixels required
            min_marking_pixels: Minimum coloured pixels in road bands required

        Returns:
            Detection results dictionary
        """
        try:
            seg_map, image_size = self.segment_image(image_path)
            image_bgr = cv2.imread(str(image_path))
            results = self.has_road_markings(seg_map, image_bgr, min_road_pixels, min_marking_pixels)
            results['image_path'] = str(image_path)
            results['image_size'] = image_size
            results['success'] = True
            return results
        except Exception as e:
            return {
                'image_path': str(image_path),
                'success': False,
                'error': str(e),
                'has_road': False,
                'has_markings': False
            }


def sample_and_filter_images(image_dir, output_file, n_sample=1000,
                             min_road_pixels=5000, min_marking_pixels=50):
    """
    Sample images and filter for road markings.

    Args:
        image_dir: Directory containing images
        output_file: JSON file to save results
        n_sample: Number of images to sample
        min_road_pixels: Minimum road pixels required
        min_marking_pixels: Minimum lane marking pixels required
    """
    print(f"\nScanning images in {image_dir}...")
    image_paths = list(Path(image_dir).glob("*.jpg"))
    print(f"Found {len(image_paths):,} total images")

    # Sample randomly
    if len(image_paths) > n_sample:
        import random
        random.seed(42)
        image_paths = random.sample(image_paths, n_sample)
        print(f"Randomly sampled {n_sample} images")

    # Initialize detector
    detector = RoadMarkingDetector()

    # Process images
    results = []
    images_with_markings = []

    print(f"\nProcessing {len(image_paths)} images...")
    for img_path in tqdm(image_paths):
        result = detector.process_image(img_path, min_road_pixels, min_marking_pixels)
        results.append(result)

        if result.get('has_markings', False):
            images_with_markings.append(result)

    # Summary statistics
    total_processed = len([r for r in results if r['success']])
    total_with_road = len([r for r in results if r.get('has_road', False)])
    total_with_markings = len(images_with_markings)

    print(f"\n{'='*60}")
    print("DETECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {total_processed:,}")
    print(f"Images with roads: {total_with_road:,} ({total_with_road/total_processed*100:.1f}%)")
    print(f"Images with lane markings: {total_with_markings:,} ({total_with_markings/total_processed*100:.1f}%)")

    # Save results
    output_data = {
        'summary': {
            'total_processed': total_processed,
            'total_with_road': total_with_road,
            'total_with_markings': total_with_markings,
            'min_road_pixels': min_road_pixels,
            'min_marking_pixels': min_marking_pixels
        },
        'all_results': results,
        'images_with_markings': images_with_markings
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_file}")
    print(f"Images with markings: {total_with_markings}")

    return images_with_markings


def main():
    parser = argparse.ArgumentParser(
        description='Detect road markings in Mapillary images using SegFormer'
    )
    parser.add_argument('--image-dir', default='full_ds/images',
                       help='Directory containing images')
    parser.add_argument('--output', default='road_marking_detections.json',
                       help='Output JSON file')
    parser.add_argument('--n-sample', type=int, default=1000,
                       help='Number of images to sample')
    parser.add_argument('--min-road-pixels', type=int, default=5000,
                       help='Minimum road pixels required')
    parser.add_argument('--min-marking-pixels', type=int, default=100,
                       help='Minimum lane marking pixels required')

    args = parser.parse_args()

    # Run detection
    images_with_markings = sample_and_filter_images(
        image_dir=args.image_dir,
        output_file=args.output,
        n_sample=args.n_sample,
        min_road_pixels=args.min_road_pixels,
        min_marking_pixels=args.min_marking_pixels
    )

    print("\n" + "="*60)
    print("FILTERING COMPLETE!")
    print("="*60)
    print(f"Found {len(images_with_markings)} images with road markings")
    print(f"\nTo visualize results, check: {args.output}")


if __name__ == "__main__":
    main()
