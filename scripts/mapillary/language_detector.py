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
    # Threshold for reliable predictions. Confidences are normalized over
    # surviving candidates: confident text lands ≥0.7, ambiguity below 0.4.
    # Constrained (diacritic-filtered) predictions face threshold × 0.6.
    MIN_CONFIDENCE = 0.4

    # Mapping of scripts to valid languages
    SCRIPT_LANGUAGES = {
        # 'sr' appears in both: Serbian uses Latin and Cyrillic co-officially,
        # and Latin-script Serbian dominates signage
        'latin': ['en', 'es', 'fr', 'de', 'it', 'pt', 'nl', 'sv', 'da', 'no', 'fi',
                 'pl', 'cs', 'sk', 'hr', 'sh', 'sr', 'sl', 'ro', 'hu', 'et', 'lv',
                 'lt', 'tr', 'id', 'ms', 'vi', 'tl', 'sw', 'so', 'af', 'zu', 'xh',
                 'is', 'ca', 'eu', 'gl', 'cy', 'ga', 'gd', 'mt', 'sq', 'bs', 'mk',
                 'la', 'nn', 'oc', 'rm', 'sc', 'wa', 'br', 'co', 'fo', 'fy', 'lb',
                 'li', 'mg', 'mi', 'nap', 'pms', 'scn', 've', 'vec', 'vls', 'wo'],
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

    # Characters that are strongly diagnostic of specific orthographies.
    # A language is only a valid candidate if its alphabet can produce every
    # special character observed in the text. Only narrow, high-signal
    # characters are listed — broadly shared accents (é, à, ô, …) are omitted
    # because too many languages (and loanwords) use them.
    DIACRITIC_LANGS = {
        # South Slavic / Baltic / Czech / Slovak / Estonian
        'č': {'hr', 'sh', 'bs', 'sr', 'sl', 'cs', 'sk', 'lt', 'lv', 'et'},
        'š': {'hr', 'sh', 'bs', 'sr', 'sl', 'cs', 'sk', 'lt', 'lv', 'et'},
        'ž': {'hr', 'sh', 'bs', 'sr', 'sl', 'cs', 'sk', 'lt', 'lv', 'et'},
        'ć': {'hr', 'sh', 'bs', 'sr', 'pl'},
        'đ': {'hr', 'sh', 'bs', 'sr', 'vi'},
        'ř': {'cs'}, 'ě': {'cs'}, 'ů': {'cs'},
        'ľ': {'sk'}, 'ĺ': {'sk'}, 'ŕ': {'sk'},
        # Polish
        'ł': {'pl'}, 'ń': {'pl'}, 'ś': {'pl'}, 'ź': {'pl'},
        'ą': {'pl', 'lt'}, 'ę': {'pl', 'lt'}, 'ż': {'pl', 'mt'},
        # Lithuanian / Latvian
        'ė': {'lt'}, 'į': {'lt'}, 'ų': {'lt'},
        'ā': {'lv'}, 'ē': {'lv'}, 'ī': {'lv'}, 'ņ': {'lv'}, 'ģ': {'lv'},
        'ķ': {'lv'}, 'ļ': {'lv'},
        'ū': {'lt', 'lv'},
        # Hungarian
        'ő': {'hu'}, 'ű': {'hu'},
        # Turkish / Romanian
        'ı': {'tr'}, 'ğ': {'tr'},
        'ş': {'tr', 'ro'}, 'ș': {'ro'}, 'ț': {'ro'}, 'ţ': {'ro'},
        'ă': {'ro'},
        'â': {'ro', 'fr', 'pt', 'vi'}, 'î': {'ro', 'fr'},
        # Icelandic / Faroese
        'þ': {'is'}, 'ð': {'is', 'fo'},
        # Nordic
        'å': {'sv', 'da', 'no', 'nn', 'fi'},
        'ø': {'da', 'no', 'nn', 'fo'},
        'æ': {'da', 'no', 'nn', 'is', 'fo'},
        # German / umlaut group
        'ß': {'de'},
        'ä': {'de', 'sv', 'fi', 'et', 'sk'},
        'ö': {'de', 'sv', 'fi', 'et', 'is', 'hu', 'tr'},
        'ü': {'de', 'et', 'hu', 'tr'},
        # Iberian
        'ñ': {'es', 'eu', 'gl', 'tl'},
        'ã': {'pt'}, 'õ': {'pt', 'et'},
        'ç': {'fr', 'pt', 'ca', 'tr', 'sq'},
        # Vietnamese
        'ơ': {'vi'}, 'ư': {'vi'},
    }

    def __init__(self, model_path: Optional[str] = None,
                 use_diacritic_filter: bool = True):
        """
        Initialize the language detector.

        Args:
            model_path: Path to fastText model. If None, uses MODEL_PATH in current dir.
            use_diacritic_filter: Constrain predictions to languages whose
                alphabet contains every diagnostic diacritic seen in the text.
        """
        self.model_path = model_path or self.MODEL_PATH
        self.use_diacritic_filter = use_diacritic_filter

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

    def _is_too_short(self, text: str, script: str = 'unknown') -> bool:
        """Check if text is too short for reliable detection.

        Counts only alphabetic characters — digits carry no language signal
        (e.g. '63 Av' has 2 letters of content, not 4). CJK gets a lower bar
        since two CJK characters form a real word.
        """
        alpha_count = sum(1 for c in text if c.isalpha())
        min_len = 2 if script == 'cjk' else 3
        return alpha_count < min_len

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
                - diacritics: Diagnostic special characters found in the text
                - constrained: True if the diacritic filter narrowed candidates
                  (confidence is then renormalized over surviving candidates)
        """
        result = {
            'language': 'unknown',
            'confidence': 0.0,
            'raw_text': text,
            'script': 'unknown',
            'is_numeric': False,
            'is_short': False,
            'script_validated': True,
            'diacritics': [],
            'constrained': False,
            'top_languages': [],
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
        if self._is_too_short(normalized, script):
            result['is_short'] = True
            return result

        lowered = normalized.lower()

        # Diacritic constraint: only languages whose alphabet contains every
        # diagnostic special character in the text remain candidates
        observed = sorted(ch for ch in set(lowered) if ch in self.DIACRITIC_LANGS)
        result['diacritics'] = observed
        allowed = None
        if self.use_diacritic_filter and observed:
            allowed = set.intersection(*(self.DIACRITIC_LANGS[ch] for ch in observed))
            if not allowed:
                # Conflicting constraints (multilingual sign / OCR glitch) —
                # back off rather than force a wrong answer
                allowed = None

        # Predict language. Lowercased: fastText is case-sensitive and trained
        # on natural-case text; ALL-CAPS signage text otherwise degrades badly.
        # Calls the C++ binding directly: fasttext 0.9.3's Python wrapper uses
        # np.array(probs, copy=False), which raises under NumPy 2.
        raw = self.model.f.predict(lowered + "\n", 20, 0.0, "strict")
        candidates = [(label.replace('__label__', ''), float(prob))
                      for prob, label in raw]

        script_valid = [(l, p) for l, p in candidates
                        if self._validate_language_script(l, script)]
        if allowed is not None:
            valid = [(l, p) for l, p in script_valid if l in allowed]
            # Sanity guard: if the surviving candidates hold little of
            # fastText's probability mass, the constraint contradicts the
            # model so strongly that a phantom OCR diacritic (e.g. ă read as
            # ä) is likelier than fastText being that wrong — back off rather
            # than renormalize a fringe candidate up to certainty.
            # 0.15 separates measured cases: phantom-diacritic text ~14%,
            # legit constrained text ≥19% (usually ≥50%). Backing off wrongly
            # just reverts to unconstrained behavior, so the cost is mild.
            script_mass = sum(p for _, p in script_valid)
            allowed_mass = sum(p for _, p in valid)
            if not valid or (script_mass > 0
                             and allowed_mass / script_mass < 0.15):
                allowed = None
                valid = script_valid
        else:
            valid = script_valid

        # Confidences are normalized over the surviving candidates — they
        # answer "how sure, among the plausible languages?" so script- or
        # alphabet-excluded competitors don't depress them
        best_lang = 'unknown'
        best_conf = 0.0
        is_valid = bool(valid)
        top_languages = []
        if valid:
            total = sum(p for _, p in valid)
            if total > 0:
                top_languages = [{'language': l, 'confidence': p / total}
                                 for l, p in valid[:3]]
                best_lang = top_languages[0]['language']
                best_conf = top_languages[0]['confidence']

        result['constrained'] = allowed is not None
        result['script_validated'] = is_valid
        result['language'] = best_lang
        result['confidence'] = best_conf
        result['top_languages'] = top_languages

        # Mark as unknown if confidence is too low. Constrained predictions
        # get a lower bar: the alphabet filter already eliminated implausible
        # languages, so a modest within-family confidence is still meaningful.
        effective = threshold * 0.6 if result['constrained'] else threshold
        if best_conf < effective:
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

    # Names for every label fastText lid.176 can emit (plus 'unknown')
    LANGUAGE_NAMES = {
        'af': 'Afrikaans', 'als': 'Alemannic', 'am': 'Amharic', 'an': 'Aragonese',
        'ar': 'Arabic', 'arz': 'Egyptian Arabic', 'as': 'Assamese',
        'ast': 'Asturian', 'av': 'Avar', 'az': 'Azerbaijani',
        'azb': 'South Azerbaijani', 'ba': 'Bashkir', 'bar': 'Bavarian',
        'bcl': 'Central Bicolano', 'be': 'Belarusian', 'bg': 'Bulgarian',
        'bh': 'Bihari', 'bn': 'Bengali', 'bo': 'Tibetan', 'bpy': 'Bishnupriya',
        'br': 'Breton', 'bs': 'Bosnian', 'bxr': 'Buryat', 'ca': 'Catalan',
        'cbk': 'Chavacano', 'ce': 'Chechen', 'ceb': 'Cebuano',
        'ckb': 'Central Kurdish', 'co': 'Corsican', 'cs': 'Czech',
        'cv': 'Chuvash', 'cy': 'Welsh', 'da': 'Danish', 'de': 'German',
        'diq': 'Zazaki', 'dsb': 'Lower Sorbian', 'dty': 'Doteli',
        'dv': 'Dhivehi', 'el': 'Greek', 'eml': 'Emilian', 'en': 'English',
        'eo': 'Esperanto', 'es': 'Spanish', 'et': 'Estonian', 'eu': 'Basque',
        'fa': 'Persian', 'fi': 'Finnish', 'fo': 'Faroese', 'fr': 'French',
        'frr': 'North Frisian', 'fy': 'West Frisian', 'ga': 'Irish',
        'gd': 'Scottish Gaelic', 'gl': 'Galician', 'gn': 'Guarani',
        'gom': 'Goan Konkani', 'gu': 'Gujarati', 'gv': 'Manx', 'he': 'Hebrew',
        'hi': 'Hindi', 'hif': 'Fiji Hindi', 'hr': 'Croatian',
        'hsb': 'Upper Sorbian', 'ht': 'Haitian Creole', 'hu': 'Hungarian',
        'hy': 'Armenian', 'ia': 'Interlingua', 'id': 'Indonesian',
        'ie': 'Interlingue', 'ilo': 'Ilocano', 'io': 'Ido', 'is': 'Icelandic',
        'it': 'Italian', 'ja': 'Japanese', 'jbo': 'Lojban', 'jv': 'Javanese',
        'ka': 'Georgian', 'kk': 'Kazakh', 'km': 'Khmer', 'kn': 'Kannada',
        'ko': 'Korean', 'krc': 'Karachay-Balkar', 'ku': 'Kurdish',
        'kv': 'Komi', 'kw': 'Cornish', 'ky': 'Kyrgyz', 'la': 'Latin',
        'lb': 'Luxembourgish', 'lez': 'Lezgian', 'li': 'Limburgish',
        'lmo': 'Lombard', 'lo': 'Lao', 'lrc': 'Northern Luri',
        'lt': 'Lithuanian', 'lv': 'Latvian', 'mai': 'Maithili',
        'mg': 'Malagasy', 'mhr': 'Eastern Mari', 'mi': 'Maori',
        'min': 'Minangkabau', 'mk': 'Macedonian', 'ml': 'Malayalam',
        'mn': 'Mongolian', 'mr': 'Marathi', 'mrj': 'Hill Mari', 'ms': 'Malay',
        'mt': 'Maltese', 'mwl': 'Mirandese', 'my': 'Burmese', 'myv': 'Erzya',
        'mzn': 'Mazanderani', 'nah': 'Nahuatl', 'nap': 'Neapolitan',
        'nds': 'Low German', 'ne': 'Nepali', 'new': 'Newari', 'nl': 'Dutch',
        'nn': 'Norwegian Nynorsk', 'no': 'Norwegian', 'oc': 'Occitan',
        'or': 'Odia', 'os': 'Ossetian', 'pa': 'Punjabi', 'pam': 'Kapampangan',
        'pfl': 'Palatine German', 'pl': 'Polish', 'pms': 'Piedmontese',
        'pnb': 'Western Punjabi', 'ps': 'Pashto', 'pt': 'Portuguese',
        'qu': 'Quechua', 'rm': 'Romansh', 'ro': 'Romanian', 'ru': 'Russian',
        'rue': 'Rusyn', 'sa': 'Sanskrit', 'sah': 'Yakut', 'sc': 'Sardinian',
        'scn': 'Sicilian', 'sco': 'Scots', 'sd': 'Sindhi',
        'sh': 'Serbo-Croatian', 'si': 'Sinhala', 'sk': 'Slovak',
        'sl': 'Slovenian', 'so': 'Somali', 'sq': 'Albanian', 'sr': 'Serbian',
        'su': 'Sundanese', 'sv': 'Swedish', 'sw': 'Swahili', 'ta': 'Tamil',
        'te': 'Telugu', 'tg': 'Tajik', 'th': 'Thai', 'tk': 'Turkmen',
        'tl': 'Tagalog', 'tr': 'Turkish', 'tt': 'Tatar', 'tyv': 'Tuvan',
        'ug': 'Uyghur', 'uk': 'Ukrainian', 'ur': 'Urdu', 'uz': 'Uzbek',
        'vec': 'Venetian', 'vep': 'Veps', 'vi': 'Vietnamese', 'vls': 'West Flemish',
        'vo': 'Volapük', 'wa': 'Walloon', 'war': 'Waray', 'wuu': 'Wu Chinese',
        'xal': 'Kalmyk', 'xmf': 'Mingrelian', 'yi': 'Yiddish', 'yo': 'Yoruba',
        'yue': 'Cantonese', 'zh': 'Chinese',
        'unknown': 'Unknown',
    }

    def get_language_name(self, iso_code: str) -> str:
        """Convert a fastText language code to its display name."""
        return self.LANGUAGE_NAMES.get(iso_code.lower(), iso_code.upper())


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