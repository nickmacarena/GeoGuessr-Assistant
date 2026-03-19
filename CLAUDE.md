# GeoGuessr AI

Computer vision pipeline: detect traffic signs → classify → infer geographic location.

## Project structure

```
GGAI/models/          — trained model weights (sign_detector, sign_classifier)
scripts/mapillary/    — data collection & processing scripts
scripts/training/     — model training & inference scripts
data/                 — datasets and derived data (gitignored)
assets/               — images for README
JOURNAL.md            — running work log (most recent entry first)
```

## Key commands

```bash
# Run full pipeline on an image
python scripts/training/detect_and_classify.py --image <path>

# Install deps
pip install -r requirements.txt
```

## Conventions

- **JOURNAL.md**: Most recent entries at the top. Each entry has a date header, "What we built" section with bullet points, and "Key findings/decisions" where relevant. Keep entries concise.
- **Commit messages**: `Action: summary` format (e.g., `Add: Mapillary sign collection script`, `Fix: detector path resolution`, `Update: README with latest benchmarks`)
- **Branch strategy**: Work on `main` for now (solo project)
- Models are stored as `.pt` files with a `label_map.json` alongside classifiers
