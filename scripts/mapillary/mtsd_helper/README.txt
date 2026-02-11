MTSD → OCR + Language-ID Prep (Helper Scripts)
=============================================

These scripts help you turn the Mapillary Traffic Sign Dataset (MTSD) into
exactly what you need for your GeoGuessr text cue project:

1) Build a unified index from the MTSD annotations → `mtsd_index.csv`
2) (Optional) Crop sign regions to speed up OCR → `crops/` + `crops_index.csv`
3) Run OCR + Language-ID on crops (or full images) → `mtsd_text_lang.csv`

Directory assumptions (after you merged images):
  full_ds/
    images/
    mtsd_v2_fully_annotated/
    mtsd_v2_partially_annotated/

Quick start:
------------
# 1) Build index
python mtsd_build_index.py   --images_dir /path/to/full_ds/images   --ann_dirs /path/to/full_ds/mtsd_v2_fully_annotated /path/to/full_ds/mtsd_v2_partially_annotated   --out_csv mtsd_index.csv

# 2) (optional) Extract sign crops
python mtsd_extract_crops.py   --images_dir /path/to/full_ds/images   --index_csv mtsd_index.csv   --out_dir crops   --min_size 24

# 3) OCR + LangID
python mtsd_run_ocr_langid.py   --crops_dir crops   --out_csv mtsd_text_lang.csv   --langid cld3

Install deps:
-------------
pip install pillow pandas numpy opencv-python tqdm
pip install paddlepaddle paddleocr
pip install pycld3   # or use fastText if you prefer

Outputs:
--------
- mtsd_index.csv           : per-annotation row with image path, bbox (if available), category
- crops/ + crops_index.csv : cropped sign patches
- mtsd_text_lang.csv       : OCR text and language predictions (script hint + prob)

Why this helps your project:
----------------------------
This pipeline gives you a **clean, language-labeled text corpus from real street signs**.
You can then:
- measure baseline LangID accuracy on actual signage text,
- find hard cases (missing diacritics, ALL-CAPS, blur) to target improvements,
- build your **"explain-by-letters"** logic (highlight diagnostic characters per language),
- and later train compact models/heuristics for in-game coaching overlays.
