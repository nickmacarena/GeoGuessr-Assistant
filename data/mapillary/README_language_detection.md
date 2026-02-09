# Language Detection for Traffic Sign Text Data

This directory contains a script for adding language detection to Mapillary traffic sign text data using fastText.

## Overview

The script `add_language_fasttext.py` processes the `det_with_text.csv` file and adds language predictions using fastText's pre-trained language identification model (lid.176.bin). The model can detect 176 languages with high accuracy.

## Files

- `add_language_fasttext.py` - Main script for language detection
- `det_with_text.csv` - Input CSV with detected text from traffic signs
- `det_with_text_fasttext.csv` - Output CSV with language columns added
- `lid.176.bin` - fastText pre-trained language identification model (downloaded automatically)

## Requirements

- Python 3.7+
- pandas
- fasttext-wheel (or fasttext)

The script automatically installs the required dependencies and downloads the language model if not present.

## Usage

### Basic Usage

Simply run the script:

```bash
python add_language_fasttext.py
```

The script will:
1. Install fasttext library if needed
2. Download the lid.176.bin model if not present (~131 MB)
3. Read `det_with_text.csv`
4. Process rows with detected text (n_regions > 0)
5. Add language detection columns
6. Save results to `det_with_text_fasttext.csv`
7. Display statistics and sample results

### Output

The script adds two new columns:
- `language_fasttext`: ISO 639 language code (e.g., 'en', 'ja', 'fr') or special values:
  - `no_text`: No text detected (n_regions = 0)
  - `empty`: n_regions > 0 but texts column is empty
  - `unknown`: Text is too short (< 2 characters)
- `confidence_fasttext`: Confidence score (0.0 to 1.0)

## Results Summary

Based on the processing of 302,999 rows:

- **Total rows**: 302,999
- **Rows with detected text regions**: 121,238 (40.0%)
- **Rows with language detected**: 70,359 (58.0% of rows with text)

### Top Languages Detected

| Language | Count | Percentage | Avg Confidence |
|----------|-------|------------|----------------|
| English (en) | 27,358 | 22.6% | 0.465 |
| Japanese (ja) | 8,052 | 6.6% | 0.549 |
| French (fr) | 6,989 | 5.8% | 0.276 |
| German (de) | 6,329 | 5.2% | 0.430 |
| Chinese (zh) | 5,424 | 4.5% | 0.638 |
| Spanish (es) | 2,849 | 2.4% | 0.384 |
| Danish (da) | 2,716 | 2.2% | 0.208 |
| Italian (it) | 1,736 | 1.4% | 0.440 |
| Portuguese (pt) | 1,045 | 0.9% | 0.406 |
| Russian (ru) | 970 | 0.8% | 0.343 |

### Confidence Statistics

- **Mean confidence**: 0.440
- **Median confidence**: 0.383
- **High confidence (>0.9)**: 4,925 detections (6.6%)

### Confidence Distribution

| Range | Count | Percentage |
|-------|-------|------------|
| 0.0 - 0.5 | 43,094 | 61.3% |
| 0.5 - 0.7 | 15,902 | 22.6% |
| 0.7 - 0.9 | 6,438 | 9.2% |
| 0.9 - 1.0 | 4,624 | 6.6% |

## Important Notes

### Numeric Text Handling

The script detects 25,925 numeric-only texts (e.g., "35", "520", "7"). These are often assigned low confidence scores or marked as "unknown" since numbers don't have inherent language. This is expected behavior.

For numeric-only texts:
- 40.5% marked as "unknown"
- 20.4% assigned to French (fr)
- 13.0% assigned to English (en)
- Other assignments with low confidence

### Short Text Limitations

Text shorter than 2 characters is automatically marked as "unknown" because language detection models require sufficient context. Single character or very short text is unreliable for language identification.

### Confidence Interpretation

- **>0.9**: Very high confidence - likely correct
- **0.7-0.9**: High confidence - usually reliable
- **0.5-0.7**: Moderate confidence - may need verification
- **<0.5**: Low confidence - uncertain, especially for short text

## Model Information

### fastText Language Identification Model (lid.176.bin)

- **Model**: Pre-trained fastText language identification model
- **Languages**: 176 languages
- **Size**: ~131 MB
- **URL**: https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
- **Paper**: ["Bag of Tricks for Efficient Text Classification"](https://arxiv.org/abs/1607.01759)

### Supported Languages

The model supports 176 languages including:
- Major world languages (English, Chinese, Spanish, French, German, etc.)
- Regional languages and dialects
- Less common languages
- Script-specific variants (e.g., wuu for Wu Chinese)

## Troubleshooting

### fasttext Installation Issues

If you encounter compatibility issues with fasttext:

1. Try `fasttext-wheel` first (usually more compatible):
   ```bash
   pip install fasttext-wheel
   ```

2. If that fails, try building from source:
   ```bash
   pip install fasttext
   ```

3. On macOS with M1/M2 chips, ensure you're using compatible Python version (3.9+)

### Model Download Issues

If the model download fails:

1. Manual download:
   ```bash
   wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
   ```

2. Or using curl:
   ```bash
   curl -O https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
   ```

3. Place the file in the same directory as the script

## Script Features

- **Automatic dependency installation**: Installs fasttext if needed
- **Automatic model download**: Downloads lid.176.bin if not present
- **Robust error handling**: Gracefully handles empty/invalid text
- **Progress tracking**: Shows processing steps and statistics
- **Comprehensive reporting**: Displays language distribution and confidence metrics
- **Sample results**: Shows examples of detected languages
- **Well-commented code**: Easy to understand and modify

## Example Output

```
Row 5:
  Text: FARMOUNT
  Language: en
  Confidence: 0.8914
  Image: --A4b2SOWVi4KL_ryAAtTg.jpg

Row 18:
  Text: PELIGRO EICAVACIOR PROFURDA
  Language: de
  Confidence: 0.2072
  Image: --USokD3k9HzZbwzNZntiQ.jpg
```

## Future Improvements

Potential enhancements:
- Add language filtering options (only detect specific languages)
- Implement multi-label classification for mixed-language text
- Add custom rules for numeric text handling
- Integrate with geographic data to improve predictions based on location
- Add visualization of language distribution on maps

## References

- [fastText official repository](https://github.com/facebookresearch/fastText)
- [fastText language identification](https://fasttext.cc/docs/en/language-identification.html)
- [Mapillary Traffic Sign Dataset](https://www.mapillary.com/dataset/trafficsign)

## License

This script is part of the GeoguessrAI project. See the main project README for license information.

## Author

Created for GeoguessrAI project - 2026-02-09