#!/usr/bin/env python3
"""
Visualize road segmentation results to verify SegFormer detection.

Shows original image, segmentation mask, and road marking overlay.
"""

import json
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
from detect_road_markings import RoadMarkingDetector


class SegmentationVisualizer:
    """Visualize road segmentation results."""

    # Cityscapes color palette
    COLORS = {
        0: [128, 64, 128],   # road - purple
        6: [255, 255, 0],    # lane marking - yellow
        1: [244, 35, 232],   # sidewalk - pink
        2: [70, 70, 70],     # building - gray
        3: [102, 102, 156],  # wall - light gray
        4: [190, 153, 153],  # fence - beige
        5: [153, 153, 153],  # pole - gray
        7: [220, 220, 0],    # traffic light - yellow
        8: [220, 20, 60],    # traffic sign - red
    }

    def __init__(self, detector=None):
        """Initialize visualizer with detector."""
        self.detector = detector or RoadMarkingDetector()

    def colorize_segmentation(self, seg_map):
        """
        Convert segmentation map to RGB colors.

        Args:
            seg_map: Segmentation map with class IDs

        Returns:
            RGB colored segmentation image
        """
        h, w = seg_map.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id, color in self.COLORS.items():
            mask = seg_map == class_id
            rgb[mask] = color

        return rgb

    def visualize_image(self, image_path, output_dir=None):
        """
        Visualize segmentation for single image.

        Args:
            image_path: Path to image
            output_dir: Directory to save visualization (optional)

        Returns:
            Combined visualization image
        """
        # Load original image
        original = cv2.imread(str(image_path))
        if original is None:
            print(f"Error loading {image_path}")
            return None

        # Get segmentation
        print(f"Segmenting {Path(image_path).name}...")
        seg_map, _ = self.detector.segment_image(image_path)

        # Check for road markings
        results = self.detector.has_road_markings(seg_map)

        # Colorize segmentation
        seg_color = self.colorize_segmentation(seg_map)
        seg_color = cv2.cvtColor(seg_color, cv2.COLOR_RGB2BGR)

        # Create overlay (50% original, 50% segmentation)
        overlay = cv2.addWeighted(original, 0.5, seg_color, 0.5, 0)

        # Resize for display
        h, w = original.shape[:2]
        max_h = 400
        if h > max_h:
            scale = max_h / h
            new_w = int(w * scale)
            original = cv2.resize(original, (new_w, max_h))
            seg_color = cv2.resize(seg_color, (new_w, max_h))
            overlay = cv2.resize(overlay, (new_w, max_h))

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        color = (255, 255, 255)

        cv2.putText(original, "Original", (10, 30), font, font_scale, color, thickness)
        cv2.putText(seg_color, "Segmentation", (10, 30), font, font_scale, color, thickness)
        cv2.putText(overlay, "Overlay", (10, 30), font, font_scale, color, thickness)

        # Add detection results
        status_text = f"Road: {'YES' if results['has_road'] else 'NO'} | Markings: {'YES' if results['has_markings'] else 'NO'}"
        cv2.putText(overlay, status_text, (10, 60), font, font_scale, (0, 255, 0) if results['has_markings'] else (0, 0, 255), thickness)

        # Combine horizontally
        combined = np.hstack([original, seg_color, overlay])

        # Save if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            output_file = output_path / f"{Path(image_path).stem}_segmentation.jpg"
            cv2.imwrite(str(output_file), combined)
            print(f"Saved: {output_file}")

        return combined, results

    def visualize_from_json(self, json_file, n_samples=10, output_dir=None, show=True):
        """
        Visualize sample images from detection results JSON.

        Args:
            json_file: JSON file with detection results
            n_samples: Number of samples to visualize
            output_dir: Directory to save visualizations
            show: Whether to display images
        """
        # Load results
        with open(json_file, 'r') as f:
            data = json.load(f)

        images_with_markings = data['images_with_markings']
        print(f"Found {len(images_with_markings)} images with road markings")

        # Sample
        import random
        samples = random.sample(images_with_markings, min(n_samples, len(images_with_markings)))

        print(f"\nVisualizing {len(samples)} samples...")

        for i, result in enumerate(samples, 1):
            image_path = result['image_path']
            print(f"\n[{i}/{len(samples)}] Processing {Path(image_path).name}")

            combined, detection_results = self.visualize_image(image_path, output_dir)

            if combined is None:
                continue

            print(f"  Road pixels: {detection_results['road_pixels']:,} ({detection_results['road_percentage']:.1f}%)")
            print(f"  Marking pixels: {detection_results['marking_pixels']:,} ({detection_results['marking_percentage']:.2f}%)")

            if show:
                cv2.imshow(f"Segmentation - {Path(image_path).name}", combined)
                print("  Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        print("\nVisualization complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize road segmentation results'
    )
    parser.add_argument('--json', default='road_marking_detections.json',
                       help='JSON file with detection results')
    parser.add_argument('--n-samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--output', default='road_segmentation_viz',
                       help='Output directory for visualizations')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display images (only save)')

    args = parser.parse_args()

    # Create visualizer
    viz = SegmentationVisualizer()

    # Visualize from JSON
    viz.visualize_from_json(
        json_file=args.json,
        n_samples=args.n_samples,
        output_dir=args.output,
        show=not args.no_show
    )


if __name__ == "__main__":
    main()
