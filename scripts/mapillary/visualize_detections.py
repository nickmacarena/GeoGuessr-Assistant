#!/usr/bin/env python3
"""
Visualize traffic sign images with detected text and predicted language.

Draws bounding boxes, recognized text, and language predictions on images.
"""

import pandas as pd
import cv2
import json
import numpy as np
from pathlib import Path
import argparse
from language_detector import LanguageDetector


class DetectionVisualizer:
    """Visualize text detections and language predictions on images."""

    def __init__(self, csv_path='det_with_text.csv', detector=None, recompute_language=True):
        """
        Initialize visualizer.

        Args:
            csv_path: Path to CSV with detections (can be det_with_text.csv or det_with_text_fasttext.csv)
            detector: LanguageDetector instance (creates new one if None)
            recompute_language: If True, recompute language with current detector (recommended)
        """
        print(f"Loading data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        self.detector = detector or LanguageDetector()
        self.recompute_language = recompute_language

        # Filter to rows with text detections
        self.df_with_text = self.df[self.df['n_regions'] > 0].copy()
        print(f"Found {len(self.df_with_text):,} images with text detections")

        if recompute_language:
            print("Note: Will recompute language predictions with current detector")

    def get_random_samples(self, n=10, min_confidence=0.5, language=None):
        """
        Get random samples for visualization.

        Args:
            n: Number of samples
            min_confidence: Minimum language detection confidence
            language: Filter by specific language code (e.g., 'en', 'zh')

        Returns:
            DataFrame with samples
        """
        df = self.df_with_text.copy()

        # Filter by confidence
        if 'confidence_fasttext' in df.columns:
            df = df[df['confidence_fasttext'] >= min_confidence]

        # Filter by language
        if language and 'language_fasttext' in df.columns:
            df = df[df['language_fasttext'] == language]

        if len(df) == 0:
            print(f"No samples found with criteria: confidence>={min_confidence}, language={language}")
            return pd.DataFrame()

        # Random sample
        n_samples = min(n, len(df))
        return df.sample(n=n_samples)

    def draw_detections(self, image_path, boxes_str, texts_str, language='', confidence=0.0, script='unknown'):
        """
        Draw text detections on image.

        Args:
            image_path: Path to image file
            boxes_str: JSON string of bounding boxes
            texts_str: JSON string of recognized texts
            language: Predicted language code
            confidence: Language prediction confidence
            script: Detected script (optional)

        Returns:
            Annotated image (numpy array)
        """
        # Load image
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"Warning: Could not load {image_path}")
            return None

        # Parse boxes and texts
        try:
            boxes = json.loads(boxes_str)
            texts = json.loads(texts_str)
        except:
            print(f"Warning: Could not parse boxes/texts for {image_path}")
            return img

        # Draw each text region
        for box, text in zip(boxes, texts):
            # Convert box to numpy array
            pts = np.array(box, dtype=np.int32)

            # Draw polygon
            cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

            # Get top-left point for text placement
            x, y = pts[0]

            # Draw text above the box
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2

            # Background rectangle for text
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(img, (x, y - text_h - 10), (x + text_w, y - 5),
                         (0, 255, 0), -1)

            # Draw recognized text
            cv2.putText(img, text, (x, y - 10), font, font_scale,
                       (0, 0, 0), thickness)

        # Draw language prediction at top of image
        if language and language != 'unknown':
            lang_name = self.detector.get_language_name(language)
            label = f"Language: {lang_name} ({language}) - Confidence: {confidence:.2f}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2

            (label_w, label_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            # Background rectangle
            cv2.rectangle(img, (10, 10), (20 + label_w, 20 + label_h),
                         (255, 255, 0), -1)

            # Draw label
            cv2.putText(img, label, (15, 15 + label_h), font, font_scale,
                       (0, 0, 0), thickness)

        return img

    def visualize_samples(self, samples, output_dir=None, show=True):
        """
        Visualize sample images with detections.

        Args:
            samples: DataFrame with samples to visualize
            output_dir: Directory to save annotated images (optional)
            show: Whether to display images in windows
        """
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            print(f"Saving visualizations to {output_path}")

        for idx, row in samples.iterrows():
            image_path = row['image_path']

            # Get detection data
            boxes = row['boxes']
            texts = row['texts']

            # Parse text for recomputation
            try:
                parsed_texts = json.loads(texts)
                combined_text = ' '.join(parsed_texts)
            except:
                combined_text = "Error parsing texts"

            # Recompute language with current detector if requested
            if self.recompute_language and combined_text != "Error parsing texts":
                result = self.detector.detect(combined_text)
                language = result['language']
                confidence = result['confidence']
                script = result.get('script', 'unknown')
                script_validated = result.get('script_validated', True)
            else:
                # Use pre-computed values from CSV
                language = row.get('language_fasttext', 'unknown')
                confidence = row.get('confidence_fasttext', 0.0)
                script = 'unknown'
                script_validated = True

            # Draw detections
            img = self.draw_detections(image_path, boxes, texts, language, confidence, script)

            if img is None:
                continue

            print(f"\nImage: {Path(image_path).name}")
            print(f"  Text: '{combined_text}'")
            if self.recompute_language:
                print(f"  Script: {script}")
                print(f"  Language: {self.detector.get_language_name(language)} ({language})")
                print(f"  Confidence: {confidence:.3f}")
                print(f"  Validated: {'✓' if script_validated else '✗'}")
            else:
                print(f"  Language: {self.detector.get_language_name(language)} ({language})")
                print(f"  Confidence: {confidence:.3f}")

            # Save if output directory specified
            if output_dir:
                output_file = output_path / f"{idx}_{Path(image_path).stem}_annotated.jpg"
                cv2.imwrite(str(output_file), img)
                print(f"  Saved: {output_file}")

            # Display if requested
            if show:
                # Resize for display if too large
                h, w = img.shape[:2]
                max_dim = 1200
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    img = cv2.resize(img, None, fx=scale, fy=scale)

                cv2.imshow(f"Detection - {Path(image_path).name}", img)
                print("  Press any key to continue...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Visualize text detections and language predictions'
    )
    parser.add_argument('--csv', default='det_with_text_fasttext.csv',
                       help='Path to CSV with detections')
    parser.add_argument('--n', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--language', type=str, default=None,
                       help='Filter by language code (e.g., en, zh, ja)')
    parser.add_argument('--min-confidence', type=float, default=0.3,
                       help='Minimum language confidence threshold')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for annotated images')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display images (only save)')

    args = parser.parse_args()

    # Initialize visualizer
    print("Initializing visualizer...")
    viz = DetectionVisualizer(csv_path=args.csv)

    # Get samples
    print(f"\nGetting {args.n} random samples...")
    samples = viz.get_random_samples(
        n=args.n,
        min_confidence=args.min_confidence,
        language=args.language
    )

    if len(samples) == 0:
        print("No samples found with specified criteria!")
        return

    print(f"Found {len(samples)} samples to visualize")

    # Visualize
    viz.visualize_samples(
        samples,
        output_dir=args.output,
        show=not args.no_show
    )

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()