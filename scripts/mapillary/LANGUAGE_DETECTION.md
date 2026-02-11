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

### Key Findings:
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

### Get Language Name

```python
lang_name = detector.get_language_name('en')  # Returns: "English"
```

## Files

- **[language_detector.py](language_detector.py)** - Production module (optimized, clean API)
- **[add_language_fasttext.py](add_language_fasttext.py)** - Batch processing script for CSVs
- **[benchmark_language_detection.py](benchmark_language_detection.py)** - Latency benchmarks
- **[compare_language_models.py](compare_language_models.py)** - fastText vs lingua-py comparison
- **lid.176.bin** - fastText model (125MB, auto-downloaded)

## Model Download

The model is auto-downloaded on first run. To manually download:

```bash
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

## Supported Languages

176 languages including:
- **Western:** English, Spanish, French, German, Italian, Portuguese
- **Eastern:** Chinese, Japanese, Korean, Thai, Vietnamese
- **European:** Russian, Polish, Czech, Swedish, Norwegian, Danish
- **Middle Eastern:** Arabic, Hebrew, Turkish
- **And 150+ more**

## Limitations

1. **Very short text (<2 chars):** Marked as `unknown`, `is_short=True`
2. **Numeric-only:** Marked as `unknown`, `is_numeric=True`
3. **Mixed scripts:** May favor dominant script (e.g., "STOP 停" → Chinese)
4. **Confidence != Accuracy:** Use 0.3+ threshold for reasonable predictions

## Recommendations for Production

1. ✅ **Load model once** at application startup (not per request)
2. ✅ **Reuse detector instance** across all requests
3. ✅ **Set confidence threshold** to 0.3+ for filtering low-quality predictions
4. ✅ **Handle special cases:**
   - Filter numeric-only text (`is_numeric=True`)
   - Handle very short text (`is_short=True`)
5. ✅ **Monitor predictions** - log low-confidence detections for review

## Example: Filtering Strategy

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

## Benchmarks

Run benchmarks anytime:

```bash
python benchmark_language_detection.py
```

Expected output:
- Single inference: ~0.01ms (p50)
- Batch of 100: ~0.01ms per item
- Throughput: >100k predictions/sec

## Dataset Results

Processed 121,238 traffic signs with detected text:
- **70,359 detections** (58% had confident predictions)
- **Top languages:** English (22.6%), Japanese (6.6%), French (5.8%), German (5.2%)
- **High confidence (>0.9):** 4,925 detections (perfect Chinese/Japanese, clear text)

See [README_language_detection.md](README_language_detection.md) for detailed dataset statistics.