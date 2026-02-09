#!/usr/bin/env python3
"""
Production-ready language detection using fastText.

This module provides a simple, optimized interface for detecting languages
from short text extracted from traffic signs.

Usage:
    from language_detector import LanguageDetector

    detector = LanguageDetector()
    result = detector.detect("STOP")
    # Returns: {'language': 'en', 'confidence': 0.95, 'raw_text': 'STOP'}
"""

import fasttext
import os
import re
import warnings
from typing import Dict, List, Optional

# Suppress fastText warnings
warnings.filterwarnings("ignore", message=".*invalid.*UTF-8.*")


class LanguageDetector:
    """Fast language detection optimized for traffic sign text."""

    MODEL_PATH = "lid.176.bin"
    MIN_CONFIDENCE = 0.3  # Threshold for reliable predictions

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the language detector.

        Args:
            model_path: Path to fastText model. If None, uses MODEL_PATH in current dir.
        """
        self.model_path = model_path or self.MODEL_PATH

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"fastText model not found at {self.model_path}\n"
                f"Download it from: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            )

        # Load model
        self.model = fasttext.load_model(self.model_path)

    def _is_numeric_only(self, text: str) -> bool:
        """Check if text is only numbers and common separators."""
        return bool(re.match(r'^[\d\s.,/\-:]+$', text.strip()))

    def _is_too_short(self, text: str) -> bool:
        """Check if text is too short for reliable detection."""
        # Remove spaces and special chars for length check
        clean_text = re.sub(r'[^\w]', '', text)
        return len(clean_text) < 2

    def _normalize_text(self, text: str) -> str:
        """Normalize text for detection."""
        # Replace newlines with spaces
        text = text.replace('\n', ' ').replace('\r', ' ')
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def detect(self, text: str, threshold: float = MIN_CONFIDENCE) -> Dict[str, any]:
        """
        Detect language from text.

        Args:
            text: Input text to classify
            threshold: Minimum confidence threshold (default: 0.3)

        Returns:
            Dictionary with:
                - language: ISO 639-1 code (lowercase) or 'unknown'
                - confidence: Float between 0 and 1
                - raw_text: Original input text
                - is_numeric: Boolean indicating if text is purely numeric
                - is_short: Boolean indicating if text is very short
        """
        result = {
            'language': 'unknown',
            'confidence': 0.0,
            'raw_text': text,
            'is_numeric': False,
            'is_short': False
        }

        # Check for empty text
        if not text or not text.strip():
            return result

        # Normalize
        normalized = self._normalize_text(text)

        # Check for numeric-only content
        if self._is_numeric_only(normalized):
            result['is_numeric'] = True
            return result

        # Check for very short text
        if self._is_too_short(normalized):
            result['is_short'] = True
            return result

        # Predict language
        predictions = self.model.predict(normalized, k=1)

        # Extract language code (remove __label__ prefix)
        lang_code = predictions[0][0].replace('__label__', '')
        confidence = float(predictions[1][0])

        result['language'] = lang_code
        result['confidence'] = confidence

        # Mark as unknown if confidence is too low
        if confidence < threshold:
            result['language'] = 'unknown'

        return result

    def detect_batch(self, texts: List[str], threshold: float = MIN_CONFIDENCE) -> List[Dict[str, any]]:
        """
        Detect languages for a batch of texts.

        Args:
            texts: List of text strings
            threshold: Minimum confidence threshold

        Returns:
            List of detection result dictionaries
        """
        return [self.detect(text, threshold) for text in texts]

    def get_language_name(self, iso_code: str) -> str:
        """
        Convert ISO 639-1 code to language name.

        Args:
            iso_code: ISO 639-1 language code (e.g., 'en', 'zh')

        Returns:
            Full language name (e.g., 'English', 'Chinese')
        """
        # Common language mappings
        LANGUAGE_NAMES = {
            'en': 'English',
            'zh': 'Chinese',
            'ja': 'Japanese',
            'ko': 'Korean',
            'fr': 'French',
            'de': 'German',
            'es': 'Spanish',
            'it': 'Italian',
            'pt': 'Portuguese',
            'ru': 'Russian',
            'ar': 'Arabic',
            'hi': 'Hindi',
            'nl': 'Dutch',
            'sv': 'Swedish',
            'da': 'Danish',
            'no': 'Norwegian',
            'fi': 'Finnish',
            'pl': 'Polish',
            'tr': 'Turkish',
            'vi': 'Vietnamese',
            'th': 'Thai',
            'cs': 'Czech',
            'el': 'Greek',
            'he': 'Hebrew',
            'unknown': 'Unknown'
        }

        return LANGUAGE_NAMES.get(iso_code.lower(), iso_code.upper())


# Example usage
if __name__ == "__main__":
    # Initialize detector
    detector = LanguageDetector()

    # Test examples
    test_texts = [
        "STOP",
        "ONE WAY",
        "禁止停车",  # No parking (Chinese)
        "止まれ",     # Stop (Japanese)
        "ARRÊT",     # Stop (French)
        "35",        # Number only
        "N1",        # Very short
        "REDUCE SPEED NOW",
        "Curva peligrosa",  # Dangerous curve (Spanish)
    ]

    print("Language Detection Examples:\n" + "="*60)

    for text in test_texts:
        result = detector.detect(text)
        lang_name = detector.get_language_name(result['language'])

        print(f"\nText: '{text}'")
        print(f"  Language: {lang_name} ({result['language']})")
        print(f"  Confidence: {result['confidence']:.3f}")

        if result['is_numeric']:
            print(f"  Note: Numeric-only content")
        if result['is_short']:
            print(f"  Note: Very short text")

    print("\n" + "="*60)