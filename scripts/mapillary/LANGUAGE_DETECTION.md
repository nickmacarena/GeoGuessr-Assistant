# Language Detection with fastText

**Decision:** Using fastText for production language detection based on accuracy and performance testing.

## Performance Summary

- **Model loading:** ~102ms (one-time cost at startup)
- **Inference latency:** ~0.01ms per prediction
- **Throughput:** >100,000 predictions/second
- **Model size:** 125MB

## Why fastText?

Compared fastText vs lingua-py on 121,238 traffic sign texts:

| Metric | fastText | lingua-py |
|--------|----------|-----------|
| **Accuracy on English** | ✅ Better | ❌ Poor (often misclassifies as Afrikaans) |
| **Avg confidence** | 0.44 | 0.20 |
| **Latency** | ✅ 0.01ms | ~0.1ms |
| **Latin scripts** | ✅ Reliable | ❌ Unreliable |
| **CJK (Chinese/Japanese)** | ✅ Excellent | ✅ Excellent |
| **Model agreement** | 19.7% (both models disagree on 80% of cases) |

Key findings:
- Both models struggle with very short text (1-3 characters)
- Numbers and abbreviations often misclassified
- fastText is more conservative and accurate on Western languages
- lingua-py over-predicts Afrikaans (16,973 false positives)

## Production Usage

### Quick Start

```python
from language_detector import LanguageDetector

# Initialize once at startup (102ms)
detector = LanguageDetector()

# Fast inference (0.01ms per call)
result = detector.detect("STOP")
print(result)
# {'language': 'en', 'confidence': 0.543, 'raw_text': 'STOP',
#  'is_numeric': False, 'is_short': False}
```

### Batch Processing

```python
texts = ["STOP", "ONE WAY", "禁止停车"]
results = detector.detect_batch(texts)
```

### Filtering Strategy

```python
result = detector.detect(text)

if result['is_numeric']:
    language = 'neutral'  # Numbers don't have language
elif result['is_short']:
    language = 'unknown'  # Too short to detect reliably
elif result['confidence'] < 0.3:
    language = 'uncertain'
else:
    language = result['language']
```

### Recommendations

1. **Load model once** at application startup (not per request)
2. **Reuse detector instance** across all requests
3. **Set confidence threshold** to 0.3+ for filtering low-quality predictions
4. **Handle special cases:** filter `is_numeric=True` and `is_short=True` results

## Files

- **[language_detector.py](language_detector.py)** — Production module (optimized, clean API)
- **[add_language_fasttext.py](add_language_fasttext.py)** — Batch processing script for CSVs
- **[benchmark_language_detection.py](benchmark_language_detection.py)** — Latency benchmarks
- **[compare_language_models.py](compare_language_models.py)** — fastText vs lingua-py comparison
- **lid.176.bin** — fastText model (125MB, gitignored, auto-downloaded on first run)

## Model Download

The model is auto-downloaded on first run. To download manually:

```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
# or
curl -O https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

Place the file in `scripts/mapillary/` (same directory as the scripts).

## Supported Languages

176 languages including:
- **Western:** English, Spanish, French, German, Italian, Portuguese
- **Eastern:** Chinese, Japanese, Korean, Thai, Vietnamese
- **European:** Russian, Polish, Czech, Swedish, Norwegian, Danish
- **Middle Eastern:** Arabic, Hebrew, Turkish

## Dataset Results

Processed 121,238 traffic signs with detected text from 302,999 total rows:

- **70,359 detections** (58% had confident predictions)

### Top Languages

| Language | Count | % of detections | Avg Confidence |
|----------|-------|-----------------|----------------|
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

### Confidence Distribution

| Range | Count | % |
|-------|-------|---|
| 0.0 – 0.5 | 43,094 | 61.3% |
| 0.5 – 0.7 | 15,902 | 22.6% |
| 0.7 – 0.9 | 6,438 | 9.2% |
| 0.9 – 1.0 | 4,624 | 6.6% |

Mean: 0.440 · Median: 0.383 · High confidence (>0.9): 4,925 (6.6%)

### Numeric Text Note

25,925 numeric-only texts (e.g., "35", "520") were detected. Numbers don't carry
language information so results are unreliable — 40.5% marked unknown, the rest
assigned low-confidence predictions. Always filter with `is_numeric=True`.

## Limitations

1. **Very short text (<2 chars):** Marked as `unknown`, `is_short=True`
2. **Numeric-only:** Marked as `unknown`, `is_numeric=True`
3. **Mixed scripts:** May favor dominant script (e.g., "STOP 停" → Chinese)
4. **Confidence != Accuracy:** Use 0.3+ threshold for reasonable predictions

## Troubleshooting

### fasttext Installation Issues

```bash
pip install fasttext-wheel   # usually more compatible
# if that fails:
pip install fasttext          # builds from source
```

On macOS Apple Silicon, ensure you're using Python 3.9+.

### Benchmarks

```bash
python benchmark_language_detection.py
```

Expected: ~0.01ms per item, >100k predictions/sec.
