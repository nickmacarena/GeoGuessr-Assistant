# Project Journal

Running log of work done, updated with each meaningful push.

---

## 2026-02-20 — YOLOv8 Sign Detector Training (Kaggle)

### What we built
- **Kaggle training notebook** (`scripts/training/kaggle_train_detector.ipynb`)
  - Trains YOLOv8n binary sign detector on MTSD via Kaggle T4 GPU
  - Converts MTSD annotations → YOLO format labels (Cell 3)
  - Monkey-patches ultralytics label resolution to handle Kaggle's read-only input paths (Cell 5)
  - **Best mAP50: 61.2%** after 25 epochs on ~17k train images
  - Outputs `yolov8n_mtsd_best.pt` → saved to `GGAI/models/sign_detector/yolov8n_mtsd/best_model.pt`

- **Kaggle CLI workflow** established for pulling outputs locally:
  - `kaggle kernels output burtsbeezer/notebook4195d7ce95 -p <dest>`
  - Updated Cell 7 to copy results files to `/kaggle/working/` so they're API-accessible

### Key findings
- Newer ultralytics auto-downloaded `yolo26n.pt` (YOLO v2.6n) as base weights instead of `yolov8n.pt`; 319/355 layers transferred
- Kaggle API only exposes top-level `/kaggle/working/` files — subdirectories like `runs/` are not accessible; fixed in notebook

### Results saved
- `GGAI/models/sign_detector/yolov8n_mtsd/results/` — training curves, confusion matrix, val predictions, args, metrics CSV

---

## 2026-02-11 — Sign Classifier + GPS Investigation

### What we built
- **MTSD Sign Classifier** (`scripts/training/train_sign_classifier.py`)
  - EfficientNet-B0 fine-tuned on 245k labeled MTSD sign crops
  - 400 classes (excluding `other-sign`), weighted sampling for class imbalance
  - **96.5% val accuracy** after 20 epochs on Apple M1 Pro (~5 min/epoch via MPS)
  - Outputs `GGAI/models/sign_classifier/best_model.pt` + `label_map.json`

- **Prediction Visualizer** (`scripts/training/visualize_predictions.py`)
  - Samples random crops, runs model, shows top-3 predictions with confidence %
  - Fetches reference SVG icons from `mapillary/mapillary_sprite_source` on GitHub
  - Renders a comparison grid: crop | pred1 | pred2 | pred3 (green border = correct)

- **Mapillary GPS fetch** (`scripts/mapillary/mtsd_fetch_gps.py`)
  - Built to fetch GPS coordinates for MTSD images via Mapillary Graph API
  - **Dead end:** MTSD uses old alphanumeric image keys (pre-2021); Graph API only supports new numeric IDs. Old v3 API (`a.mapillary.com`) is shut down. GPS for MTSD images is not recoverable via API.
  - Script kept for reference / future use with newer Mapillary images

### Key decisions
- Chose B0 over B2: sign crops are 96×96px, B0 is less overparameterized for small inputs
- `other-sign` excluded from training (102k samples, catch-all label, not useful to classify)
- Sign → region mapping deferred to `mapping.json` from `mapillary_sprite_source` (5 regions: us/eu/br/au/ca)

### Sign label taxonomy
`{category}--{type}--{variant}` e.g. `regulatory--stop--g1`
- Variant numbers (g1, g2, g25) are arbitrary IDs for visually distinct regional designs
- Reference SVGs: `https://github.com/mapillary/mapillary_sprite_source/blob/master/package_signs/{label}.svg`

---

## 2026-02-11 — Project Reorganization + Road Line Exploration

### Reorganization
- Moved all processing scripts from `data/mapillary/` → `scripts/mapillary/` (separation of code from data)
- Added `.env` / `.env.example` for Mapillary API token storage

### Sign country distribution
- Built `scripts/mapillary/build_sign_country_dist.py` + `country_bboxes.json`
- Computes per-country sign type frequency from MTSD GPS metadata
- Outputs `GGAI/data/sign_country_dist.json` — used for region scoring at inference time

### Road line color classification (explored, not kept)
- Explored detecting road markings (white/yellow lines) as a geolocation signal on a feature branch
- Built detection + color classification pipeline (`detect_road_markings.py`, `classify_road_lines.py`)
- **Abandoned:** too noisy, inconsistent across lighting/weather, weak geo-signal compared to signs
- Branch merged but scripts removed; approach documented here for reference

---

## 2026-02-09 — Language Detection for Sign Text

### What we built
- **fastText language detector** (`scripts/mapillary/language_detector.py`)
  - Production module wrapping `lid.176.bin` (176-language fastText model)
  - 0.01ms/prediction, >100k predictions/sec
  - Handles edge cases: numeric-only text, very short text (<2 chars)

- **Batch processing script** (`scripts/mapillary/add_language_fasttext.py`)
  - Processes `det_with_text.csv` → adds `language_fasttext` + `confidence_fasttext` columns
  - Processed 121,238 sign texts; 58% had confident predictions
  - Top detected languages: English (22.6%), Japanese (6.6%), French (5.8%), German (5.2%)

### Key decisions
- fastText chosen over lingua-py: 2× higher avg confidence, reliable on Latin scripts, lingua-py had 16k false Afrikaans predictions
- Confidence threshold 0.3+ recommended for production use

### Reference
See [scripts/mapillary/LANGUAGE_DETECTION.md](scripts/mapillary/LANGUAGE_DETECTION.md) for full benchmark results.

---

## 2026-02-09 — Data Pipeline Setup

### What we built
- MTSD data download + extraction pipeline (`data/README.md`)
- `scripts/mapillary/mtsd_helper/mtsd_build_index.py` — builds `crops_index.csv` from MTSD annotations
- `scripts/mapillary/mtsd_helper/mtsd_extract_crops.py` — extracts sign bounding box crops from full images
- Text detection on sign crops via `detect_text_regions_with_rec.py`

### Data produced (local only, gitignored)
- `data/mapillary/crops/` — 245k sign crop images
- `data/mapillary/crops/crops_index.csv` — 245k rows: crop_path, image_path, category, bbox_abs

---

## 2025-11-16 — Project Start

Initial data preprocessing scripts committed.
