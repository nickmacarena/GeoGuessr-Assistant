# Project Journal

Running log of work done, updated with each meaningful push.

---

## 2026-06-10 — Sign→Country Classifier, Live GeoGuessr Companion, Text/Language Pipeline

### What we built
- **Sign→country Naive Bayes classifier** (`scripts/training/train_geo_classifier.py`)
  - Groups 154k collected Mapillary signs into 59,617 ~100m grid-cell "scenes", fits multinomial NB over sign-class counts
  - Held-out accuracy (44 countries, no prior): top-1 33% / top-5 67% overall; **69% top-1 / 91% top-5 for scenes with 10+ signs** (chance = 2.3%)
  - Model is pure JSON (`GGAI/models/geo_classifier/sign_country_model.json`) — per-class log-likelihoods, no torch at inference
- **Live GeoGuessr companion** (`scripts/training/geo_live.py`)
  - Daemon: loads all models once, watches `~/Desktop` for screenshots, serves auto-updating inspector at `localhost:8077`
  - Play flow: Cmd-Shift-3/4 during a round → annotated result in ~3-5s, hover boxes for evidence
- **Text/language pipeline** (`scripts/training/text_detector.py` + upgraded `language_detector.py`)
  - OCR via macOS Vision (subprocess — Vision SIGBUSes in-process with torch/MPS; DYLD_LIBRARY_PATH must be stripped)
  - Vertically-adjacent OCR lines merge into blocks before language ID (billboards arrive line-by-line otherwise)
  - **Diacritic constraint filter**: ~50 diagnostic chars (č, ő, ı, þ, ă…) restrict candidates to languages whose alphabet matches; confidences renormalized over survivors; probability-mass guard (<15%) backs off on phantom OCR diacritics
  - Sign tooltips show learned per-sign country likelihoods from the NB model; text tooltips show top-3 languages
- **Inspector UX** (`geo_inspector.py`): boxes z-ordered by area + single page-level JS tooltip so nested detections stay hoverable; `--text-conf` flag (default 0.5 — Vision confidence is tiered ~1.0/0.5/0.3, not calibrated)

### Key findings
- mIoU-style intuition repeated for languages: fastText is case-sensitive — ALL-CAPS signage text degrades badly, lowercase before predicting
- fastText often prefers `sh`/`sr` (Latin Serbian) for Croatian text; they're near-inseparable on sign-length text — diacritic note signals "former Yugoslavia" which is the realistic ceiling
- Renormalizing over a constraint's lone survivor can inflate a 0.5% candidate to 100% — phantom OCR diacritics (ă→ä) make this a real failure mode; the mass guard bounds it
- `python3` on this machine resolves to an unrelated venv; use `.venv/bin/python3` (no DYLD prefix needed)

---

## 2026-06-10 — Lane Line Segmentation (BDD100K) + Interactive Geo Inspector

### What we built
- **Lane mask rasterizer** (`scripts/training/bdd_rasterize_lanes.py`)
  - Converts BDD100K poly2d lane annotations (with cubic beziers) → pixel masks
  - Final taxonomy: 9 classes = color (white/yellow) × multiplicity (single/double) × style (solid/dashed) + bg
  - Lane lines only — drops curb/crosswalk and transverse (`direction != parallel`) markings
- **Kaggle training notebook** (`scripts/training/kaggle_train_lane_seg.ipynb`)
  - DeepLabV3 + MobileNetV3-Large, 512×256, 25 epochs on T4 (~12h)
  - Checkpoint selection by **image-level weighted F1** (class present = ≥50 px), not mIoU
  - Masks uploaded as Kaggle dataset `burtsbeezer/bdd100k-lane-masks-v2`
- **Trained model** (`GGAI/models/lane_segmentation_v4/best_model.pt`)
  - Per-class F1: s_white_dashed 0.87, d_yellow_solid 0.86, s_white_solid 0.82, s_yellow_solid 0.75, s_yellow_dashed 0.58, d_yellow_dashed 0.56, d_white_solid 0.43, d_white_dashed 0.34
  - Checkpoint embeds class names, per-class F1/IoU, presence thresholds for inference
- **Interactive geo inspector** (`scripts/training/geo_inspector.py`)
  - Runs sign pipeline (YOLO → EfficientNet) + lane pipeline (segmentation → connected components) independently — no fusion
  - Outputs standalone HTML: hover any box → classification + countries where that sign design / lane marking is used
  - Sign regions from `GGAI/models/sign_classifier/region_mapping.json` (cached from mapillary_sprite_source, 1550 entries)
  - Demo: `assets/demo_geo_inspector.html` (BDD frame with 6 lane classes + US-only sign)

### Key findings
- **mIoU is misleading for presence tasks**: 7-class model had mIoU 0.29 (blobby masks) but image-level wF1 0.81 — presence detection is strong even when localization is sloppy
- Fine-tuning a converged model 25 more epochs (v3) regressed val loss +30% — overfit, no metric gains
- Dashed-vs-solid is the geo payload: s_yellow_dashed (US/CA/MX passing-allowed) successfully separated from solid (F1 0.58) despite being only 0.007% of training pixels
- Yellow centerlines ≠ US-only: also JP/KR/TW, Latin America, Norway, Iceland; EU yellow = roadworks
- Cross-region sanity check passed: yellow classes fire on US/BDD frames, stay quiet on a Dutch roundabout (one small false positive); EU-only sign designs surface correctly

---

## 2026-03-19 — Mapillary Sign Collection Pipeline

### What we built
- **City/bbox configuration** (`scripts/mapillary/cities.py`)
  - 58 cities across 35 countries, organized by region (Europe, Americas, Asia, Oceania, Africa, Middle East)
  - Each city defines center coords + grid radius (~10km×10km area → ~100 tiles per city)

- **Map features collection script** (`scripts/mapillary/collect_signs.py`)
  - Queries Mapillary `map_features` API for traffic signs in tiled bboxes (<0.01° each)
  - Filters results to only our classifier's 400 known sign classes (via `label_map.json`)
  - Deduplicates by feature ID across overlapping tiles
  - Offline reverse geocoding via `reverse_geocoder` (lat/lon → country code, no API calls)
  - Outputs single CSV: `feature_id, lat, lon, sign_class, country, city`
  - Supports `--test` (one tile), `--city <name>`, `--all`, and `--resume` modes
  - Rate limiting + retry on 429

- **Mapillary API test script** (`scripts/mapillary/test_api.py`)
  - Validates search API (bbox image queries) and entity API (thumbnail URLs)
  - Confirmed API working with SF bbox; Paris bbox too small for coverage

- **Pipeline demo visualizations** (`GGAI/models/pipeline_demo*.png`)
  - 4-panel images: original street view → detected signs → classified crops → reference SVG icons

### Key findings
- Mapillary's API taxonomy includes ~55 classes per tile, but only ~19 match our 400-class label map (construction barriers, road markings, traffic lights filtered out)
- Berlin test tile returned 344 features, reverse geocoded correctly to DE
- Coverage varies widely by region — need large bboxes for sparser areas

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

- **End-to-end inference script** (`scripts/training/detect_and_classify.py`)
  - Chains YOLOv8 detector → EfficientNet-B0 classifier on a single street-view image
  - Outputs annotated JPEG with bounding boxes, top-3 sign predictions, and confidence %
  - Auto-selects device (MPS → CUDA → CPU); configurable `--conf`, `--iou`, `--top-k`
  - Usage: `python detect_and_classify.py --image <path>`

- **`requirements.txt`** added with core deps: `torch`, `torchvision`, `ultralytics`, `pillow`

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
