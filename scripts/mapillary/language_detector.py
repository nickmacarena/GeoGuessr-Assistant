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
from pathlib import Path
from typing import Dict, List, Optional

# Suppress fastText warnings
warnings.filterwarnings("ignore", message=".*invalid.*UTF-8.*")


class LanguageDetector:
    """Fast language detection optimized for traffic sign text."""

    MODEL_PATH = str(Path(__file__).parent.parent.parent / "data" / "mapillary" / "lid.176.bin")
    MIN_CONFIDENCE = 0.3  # Threshold for reliable predictions

    # Mapping of scripts to valid languages
    SCRIPT_LANGUAGES = {
        'latin': ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'sv', 'da', 'no', 'fi',
                 'pl', 'cs', 'sk', 'hr', 'sl', 'ro', 'hu', 'et', 'lv', 'lt', 'tr',
                 'id', 'ms', 'vi', 'tl', 'sw', 'so', 'af', 'zu', 'xh', 'is', 'ca',
                 'eu', 'gl', 'cy', 'ga', 'gd', 'mt', 'sq', 'bs', 'mk', 'la', 'nn',
                 'oc', 'rm', 'sc', 'wa', 'br', 'co', 'fo', 'fy', 'lb', 'li', 'mg',
                 'mi', 'nap', 'pms', 'scn', 've', 'vec', 'vls', 'wo'],
        'cjk': ['zh', 'ja', 'ko', 'wuu', 'yue', 'nan'],
        'cyrillic': ['ru', 'uk', 'bg', 'sr', 'mk', 'be', 'kk', 'ky', 'tg', 'mn'],
        'arabic': ['ar', 'fa', 'ur', 'ps', 'sd', 'ckb', 'ku'],
        'devanagari': ['hi', 'ne', 'mr', 'sa'],
        'greek': ['el'],
        'hebrew': ['he', 'yi'],
        'thai': ['th'],
        'georgian': ['ka'],
        'armenian': ['hy'],
    }

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

    def _detect_script(self, text: str) -> str:
        """
        Detect the writing system/script of text.

        Returns:
            Script name: 'numeric', 'latin', 'cjk', 'cyrillic', 'arabic',
                        'devanagari', 'greek', 'hebrew', 'thai', 'mixed', or 'unknown'
        """
        # Unicode ranges for different scripts
        has_latin = bool(re.search(r'[a-zA-Z]', text))
        has_cjk = bool(re.search(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]', text))
        has_cyrillic = bool(re.search(r'[\u0400-\u04ff]', text))
        has_arabic = bool(re.search(r'[\u0600-\u06ff]', text))
        has_devanagari = bool(re.search(r'[\u0900-\u097f]', text))
        has_greek = bool(re.search(r'[\u0370-\u03ff]', text))
        has_hebrew = bool(re.search(r'[\u0590-\u05ff]', text))
        has_thai = bool(re.search(r'[\u0e00-\u0e7f]', text))

        # Count how many scripts are present
        scripts_present = sum([
            has_latin, has_cjk, has_cyrillic, has_arabic,
            has_devanagari, has_greek, has_hebrew, has_thai
        ])

        # Check if purely numeric
        if self._is_numeric_only(text):
            return 'numeric'

        # Mixed scripts (e.g., "STOP 停车")
        if scripts_present > 1:
            return 'mixed'

        # Single script
        if has_cjk:
            return 'cjk'
        elif has_cyrillic:
            return 'cyrillic'
        elif has_arabic:
            return 'arabic'
        elif has_devanagari:
            return 'devanagari'
        elif has_greek:
            return 'greek'
        elif has_hebrew:
            return 'hebrew'
        elif has_thai:
            return 'thai'
        elif has_latin:
            return 'latin'
        else:
            return 'unknown'

    def _validate_language_script(self, language: str, script: str) -> bool:
        """
        Validate if a predicted language matches the detected script.

        Args:
            language: Predicted ISO 639-1 language code
            script: Detected script name

        Returns:
            True if language is valid for the script, False otherwise
        """
        # Numeric text has no language
        if script == 'numeric':
            return False

        # Allow mixed scripts (can't validate reliably)
        if script == 'mixed':
            return True

        # Unknown script - allow
        if script == 'unknown':
            return True

        # Check if language is in the valid set for this script
        if script in self.SCRIPT_LANGUAGES:
            return language in self.SCRIPT_LANGUAGES[script]

        # Default to allowing if we don't have rules
        return True

    def detect(self, text: str, threshold: float = MIN_CONFIDENCE) -> Dict[str, any]:
        """
        Detect language from text with script validation.

        Args:
            text: Input text to classify
            threshold: Minimum confidence threshold (default: 0.3)

        Returns:
            Dictionary with:
                - language: ISO 639-1 code (lowercase) or 'unknown'
                - confidence: Float between 0 and 1
                - raw_text: Original input text
                - script: Detected writing system
                - is_numeric: Boolean indicating if text is purely numeric
                - is_short: Boolean indicating if text is very short
                - script_validated: Boolean indicating if language matches script
        """
        result = {
            'language': 'unknown',
            'confidence': 0.0,
            'raw_text': text,
            'script': 'unknown',
            'is_numeric': False,
            'is_short': False,
            'script_validated': True
        }

        # Check for empty text
        if not text or not text.strip():
            return result

        # Normalize
        normalized = self._normalize_text(text)

        # Detect script
        script = self._detect_script(normalized)
        result['script'] = script

        # Check for numeric-only content
        if self._is_numeric_only(normalized):
            result['is_numeric'] = True
            return result

        # Check for very short text
        if self._is_too_short(normalized):
            result['is_short'] = True
            return result

        # Predict language - get top 5 predictions
        predictions = self.model.predict(normalized, k=5)

        # Extract language codes and confidences
        lang_codes = [pred.replace('__label__', '') for pred in predictions[0]]
        confidences = [float(conf) for conf in predictions[1]]

        # Find the first prediction that matches the script
        best_lang = 'unknown'
        best_conf = 0.0
        is_valid = False

        for lang_code, confidence in zip(lang_codes, confidences):
            if self._validate_language_script(lang_code, script):
                best_lang = lang_code
                best_conf = confidence
                is_valid = True
                break

        result['script_validated'] = is_valid
        result['language'] = best_lang
        result['confidence'] = best_conf

        # Mark as unknown if confidence is too low
        if best_conf < threshold:
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