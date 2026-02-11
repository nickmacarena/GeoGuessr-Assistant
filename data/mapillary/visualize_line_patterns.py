#!/usr/bin/env python3
"""
Visualize different road line patterns.

Shows examples of yellow center, white center, and edge line patterns.
"""

import cv2
import numpy as np
import json
from pathlib import Path
import argparse


class LinePatternVisualizer:
    """Visualize road line patterns with color overlays."""

    def __init__(self):
        """Initialize visualizer."""
        # HSV color ranges for visualization
        self.YELLOW_LOWER = np.array([20, 100, 100])
        self.YELLOW_UPPER = np.array([40, 255, 255])
        self.WHITE_LOWER = np.array([0, 0, 200])
        self.WHITE_UPPER = np.array([180, 30, 255])

    def visualize_pattern(self, image_path, classification):
        """
        Visualize road line pattern with color overlay.

        Args:
            image_path: Path to image
            classification: Classification results dict

        Returns:
            Annotated image
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Error loading {image_path}")
            return None

        # Extract road region (bottom 40%)
        h, w = img.shape[:2]
        start_row = int(h * 0.6)
        road_region = img[start_row:, :]

        # Convert to HSV
        hsv = cv2.cvtColor(road_region, cv2.COLOR_BGR2HSV)

        # Create masks
        yellow_mask = cv2.inRange(hsv, self.YELLOW_LOWER, self.YELLOW_UPPER)
        white_mask = cv2.inRange(hsv, self.WHITE_LOWER, self.WHITE_UPPER)

        # Create color overlays
        yellow_overlay = np.zeros_like(road_region)
        yellow_overlay[yellow_mask > 0] = (0, 255, 255)  # Yellow in BGR

        white_overlay = np.zeros_like(road_region)
        white_overlay[white_mask > 0] = (255, 255, 255)  # White

        # Combine overlays
        combined_overlay = cv2.addWeighted(yellow_overlay, 0.5, white_overlay, 0.5, 0)

        # Blend with original road region
        road_with_overlay = cv2.addWeighted(road_region, 0.7, combined_overlay, 0.3, 0)

        # Replace road region in original image
        result = img.copy()
        result[start_row:, :] = road_with_overlay

        # Add text labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        thickness = 2

        # Pattern info
        patterns = classification['line_pattern']
        pattern_text = ", ".join(patterns)

        # Yellow/White info
        yellow_pct = classification['yellow_info']['percentage']
        white_pct = classification['white_info']['percentage']

        # Label
        label_lines = [
            f"Pattern: {pattern_text}",
            f"Yellow: {yellow_pct:.2f}% | White: {white_pct:.2f}%"
        ]

        y_offset = 40
        for line in label_lines:
            (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            cv2.rectangle(result, (10, y_offset - text_h - 10),
                         (20 + text_w, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(result, line, (15, y_offset), font, font_scale,
                       (0, 255, 0), thickness)
            y_offset += 45

        return result

    def visualize_pattern_samples(self, classifications, pattern_filter, n_samples=5,
                                  output_dir=None, show=False):
        """
        Visualize samples of a specific pattern.

        Args:
            classifications: List of classification results
            pattern_filter: Pattern to filter for (e.g., 'yellow_center')
            n_samples: Number of samples to visualize
            output_dir: Directory to save visualizations
            show: Whether to display images
        """
        # Filter for pattern
        filtered = [c for c in classifications
                   if c.get('success', False) and pattern_filter in c['line_pattern']]

        if len(filtered) == 0:
            print(f"No samples found for pattern: {pattern_filter}")
            return

        print(f"\nFound {len(filtered)} images with pattern: {pattern_filter}")
        print(f"Visualizing {min(n_samples, len(filtered))} samples...")

        # Sample randomly
        import random
        random.seed(42)
        samples = random.sample(filtered, min(n_samples, len(filtered)))

        # Create output directory
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)

        for i, classification in enumerate(samples, 1):
            image_path = classification['image_path']
            print(f"\n[{i}/{len(samples)}] {Path(image_path).name}")

            # Visualize
            result = self.visualize_pattern(image_path, classification)
            if result is None:
                continue

            # Save
            if output_dir:
                output_file = output_path / f"{pattern_filter}_{i}_{Path(image_path).stem}.jpg"
                cv2.imwrite(str(output_file), result)
                print(f"  Saved: {output_file}")

            # Display
            if show:
                # Resize for display
                h, w = result.shape[:2]
                max_h = 800
                if h > max_h:
                    scale = max_h / h
                    result = cv2.resize(result, None, fx=scale, fy=scale)

                cv2.imshow(f"{pattern_filter} - {Path(image_path).name}", result)
                print("  Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize road line patterns'
    )
    parser.add_argument('--input', default='road_line_classifications.json',
                       help='Input JSON with classifications')
    parser.add_argument('--patterns', nargs='+',
                       default=['yellow_center', 'white_center', 'white_edges', 'yellow_edges'],
                       help='Patterns to visualize')
    parser.add_argument('--n-samples', type=int, default=5,
                       help='Number of samples per pattern')
    parser.add_argument('--output', default='line_pattern_viz',
                       help='Output directory')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display images')

    args = parser.parse_args()

    # Load classifications
    print(f"Loading classifications from {args.input}...")
    with open(args.input, 'r') as f:
        data = json.load(f)

    classifications = data['classifications']
    print(f"Loaded {len(classifications)} classifications")

    # Initialize visualizer
    viz = LinePatternVisualizer()

    # Visualize each pattern
    for pattern in args.patterns:
        print(f"\n{'='*60}")
        print(f"VISUALIZING PATTERN: {pattern.upper()}")
        print(f"{'='*60}")

        viz.visualize_pattern_samples(
            classifications=classifications,
            pattern_filter=pattern,
            n_samples=args.n_samples,
            output_dir=args.output,
            show=not args.no_show
        )

    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETE!")
    print(f"{'='*60}")
    print(f"Output saved to: {args.output}/")


if __name__ == "__main__":
    main()
