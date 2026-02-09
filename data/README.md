# Data Generation Pipeline for GeoGuessr Assistant

This document describes how to reproduce the training data used for the GeoGuessr Assistant, which identifies road signs and text (including language detection) from street imagery.

## Overview

The pipeline transforms the [Mapillary Traffic Sign Dataset (MTSD)](https://www.mapillary.com/dataset/trafficsign) into training-ready data with:
- **Traffic sign bounding boxes** and class labels (313 sign types)
- **OCR-extracted text** from signs with confidence scores
- **Language detection** capabilities for geolocation hints

**Note**: The raw MTSD dataset is NOT included in this repository due to size and licensing. You must download it separately (see below).

---

## Prerequisites

### 1. Download the MTSD Dataset

Visit the [Mapillary Traffic Sign Dataset](https://www.mapillary.com/dataset/trafficsign) page and download:

- **MTSD v2 Fully Annotated** (`mtsd_v2_fully_annotated.zip`)
  - ~10,000 images with complete annotations
  - All traffic signs labeled with bounding boxes

- **MTSD v2 Partially Annotated** (optional)
  - Additional images with subset annotations
  - Useful for expanding training data

- **Images** (`images.zip`)
  - High-resolution street view images from around the world

After downloading, extract the files into a directory structure like:
```
data/
├── mapillary/
│   └── full_ds/
│       ├── images/                          # Extracted from images.zip
│       │   ├── --48MAqc82-bZdgGpaiexA.jpg
│       │   └── ...
│       ├── mtsd_v2_fully_annotated/        # Extracted annotation package
│       │   ├── annotations/
│       │   │   ├── --48MAqc82-bZdgGpaiexA.json
│       │   │   └── ...
│       │   ├── splits/
│       │   │   ├── train.txt
│       │   │   ├── val.txt
│       │   │   └── test.txt
│       │   └── README.md
│       └── mtsd_v2_partially_annotated/    # Optional
│           └── annotations/
```

### 2. Install Python Dependencies

Create a virtual environment and install required packages:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy opencv-python tqdm pillow
pip install paddlepaddle paddleocr  # For OCR text recognition
pip install pycld3  # For language identification (optional)
```

**Key dependencies:**
- `pandas` - Data manipulation and CSV handling
- `opencv-python` - Image processing and cropping
- `paddleocr` - Text detection and recognition on signs
- `pycld3` - Compact Language Detector (for language ID)

---

## Data Generation Steps

### Step 1: Build Unified Index

Parse all MTSD JSON annotations into a single CSV file for easy processing.

**Script:** `mtsd_helper/mtsd_build_index.py`

```bash
python mtsd_helper/mtsd_build_index.py \
  --images_dir /path/to/full_ds/images \
  --ann_dirs /path/to/full_ds/mtsd_v2_fully_annotated/annotations \
  --out_csv mtsd_index.csv \
  --debug
```

**Arguments:**
- `--images_dir`: Directory containing the image files
- `--ann_dirs`: One or more directories containing JSON annotations
- `--out_csv`: Output CSV path (default: `mtsd_index.csv`)
- `--img_exts`: Comma-separated image extensions to try (default: `.jpg,.jpeg,.png`)
- `--debug`: Print verbose processing information

**Output:** `mtsd_index.csv`

Example format:
```csv
image_file,category,bbox,segmentation,source,image_path
--48MAqc82-bZdgGpaiexA,other-sign,"[2657.890625, 1013.0859375, 37.578125, 38.0859375]",,per_image_json,/path/to/images/--48MAqc82-bZdgGpaiexA.jpg
--7fWq6WjZM8L1eUSuvOEA,regulatory--maximum-speed-limit-40--g1,"[2511.640625, 1391.66015625, 111.71875, 117.3046875]",,per_image_json,/path/to/images/--7fWq6WjZM8L1eUSuvOEA.jpg
```

**What it does:**
- Parses MTSD JSON format (`{"width": ..., "height": ..., "objects": [{"label": ..., "bbox": {...}}]}`)
- Supports both COCO-style and per-image JSON formats
- Resolves image paths by trying different extensions
- Validates that image files exist
- Normalizes bounding boxes to `[x, y, w, h]` format

---

### Step 2: Extract Sign Crops (Optional but Recommended)

Extract individual traffic sign regions from full images to speed up OCR processing.

**Script:** `mtsd_helper/mtsd_extract_crops.py`

```bash
python mtsd_helper/mtsd_extract_crops.py \
  --images_dir /path/to/full_ds/images \
  --index_csv mtsd_index.csv \
  --out_dir crops \
  --min_size 24
```

**Arguments:**
- `--images_dir`: Directory containing the source images
- `--index_csv`: CSV from Step 1 with bounding boxes
- `--out_dir`: Output directory for cropped images
- `--min_size`: Minimum short-side pixel size to keep (default: 24)

**Output:**
- `crops/` directory with individual sign images
  - Named as `{image_id}_{row_index}.jpg`
- `crops/crops_index.csv` mapping crops to original images

Example crops_index.csv:
```csv
crop_path,image_path,category,bbox_abs
crops/--48MAqc82-bZdgGpaiexA_0.jpg,/path/to/images/--48MAqc82-bZdgGpaiexA.jpg,other-sign,"[2655, 1011, 39, 40]"
```

**What it does:**
- Crops each sign from the full image using bounding boxes
- Applies 2-pixel padding around each crop
- Filters out crops smaller than minimum size
- Saves high-quality JPEGs (quality=95)
- Stores absolute bounding box coordinates

---

### Step 3: Run OCR Text Detection & Recognition

Detect and recognize text on traffic signs using PaddleOCR.

**Script:** `detect_text_regions_with_rec.py`

```bash
python detect_text_regions_with_rec.py \
  --paths_csv mtsd_index.csv \
  --out_csv det_with_text.csv \
  --max_side 960 \
  --every 500
```

**Arguments:**
- `--paths_csv`: CSV with `image_path` and `bbox` columns (from Step 1)
- `--out_csv`: Output CSV with OCR results
- `--max_side`: Resize longer side to this for OCR (0 = no limit, 960 recommended for speed)
- `--every`: Flush results to CSV every N rows (default: 500)

**Output:** `det_with_text.csv`

Example format:
```csv
image_path,bbox,n_regions,boxes,texts,scores,error
/path/to/image.jpg,"[1219.7109375, 1661.783203125, 42.796875, 51.7939453125]",1,"[[[1223.0, 1686.0], [1259.0, 1686.0], [1259.0, 1711.0], [1223.0, 1711.0]]]","[""35""]","[0.9961197376251221]",
/path/to/image2.jpg,"[2421.9140625, 1571.326171875, 160.48828125, 38.6630859375]",2,"[[[2435.0, 1574.0], [2533.0, 1580.0], [2532.0, 1605.0], [2434.0, 1600.0]], [[2546.0, 1585.0], [2571.0, 1589.0], [2569.0, 1607.0], [2544.0, 1603.0]]]","[""FAIRDALE"", ""AVE""]","[0.9938940405845642, 0.98996502161026]",
```

**What it does:**
- Crops sign regions from full images using bounding boxes
- Runs PaddleOCR detection + recognition pipeline
- Extracts text boxes (polygons), recognized text, and confidence scores
- Converts coordinates to full-image space
- Handles errors gracefully (missing files, invalid crops, OCR failures)
- Supports periodic flushing for large datasets

---

## Output Files Summary

After running the complete pipeline, you'll have:

| File | Description | Size (approx) |
|------|-------------|---------------|
| `mtsd_index.csv` | All traffic signs with bounding boxes and categories | ~45 MB |
| `crops/` | Directory of cropped traffic sign images | ~1.8 GB |
| `crops/crops_index.csv` | Index mapping crops to source images | ~2 MB |
| `det_with_text.csv` | OCR results with detected text and confidence | ~50 MB |

---

## Expected Dataset Statistics

Based on MTSD v2 fully annotated:
- **~10,000 images** from 6 continents
- **~52,000 traffic sign annotations**
- **313 sign classes** (speed limits, warnings, regulatory, etc.)
- **~15-20% of signs contain readable text**

Sign categories include:
- `regulatory--maximum-speed-limit-{N}--g{1-8}` (speed limits by region)
- `warning--*` (warning signs: curves, bumps, pedestrians, etc.)
- `information--*` (informational signs: parking, directions, etc.)
- `complementary--*` (supplementary text panels)
- `other-sign` (uncategorized signs)

---

## Data Formats

### MTSD Annotation JSON Format

Each image has a corresponding JSON file:

```json
{
  "width": 1920,
  "height": 1080,
  "ispano": false,
  "objects": [
    {
      "label": "regulatory--maximum-speed-limit-40--g1",
      "bbox": {
        "xmin": 934.21875,
        "ymin": 799.98046875,
        "xmax": 961.40625,
        "ymax": 827.9296875
      },
      "key": "t79kx4b2vuyv2mvi88x1s6",
      "properties": {
        "occluded": false,
        "ambiguous": false
      }
    }
  ]
}
```

### Bounding Box Convention

All scripts use `[x, y, width, height]` format in absolute pixel coordinates:
- `x, y` = top-left corner
- `width, height` = box dimensions

---

## Troubleshooting

### Issue: "No JSON files found"
- Check that `--ann_dirs` points to the `annotations/` folder(s)
- Verify you've extracted the MTSD annotation packages

### Issue: "Dropped N rows whose images were not found"
- Ensure `--images_dir` points to the extracted `images/` folder
- Check that image files have extensions: `.jpg`, `.jpeg`, or `.png`

### Issue: PaddleOCR errors or slow performance
- Reduce `--max_side` parameter (try 640 or 480)
- Ensure you have sufficient RAM (8GB+ recommended)
- Consider processing in smaller batches

### Issue: Missing text on obvious signs
- Some signs may not have clear text or use symbols only
- OCR confidence threshold can be adjusted in post-processing
- Check the `error` column in `det_with_text.csv` for failures

---

## Next Steps: Language Detection

For language identification on extracted text, you can:

1. **Use pycld3** (Compact Language Detector):
```python
import pandas as pd
import pycld3

df = pd.read_csv("det_with_text.csv")
df = df[df['n_regions'] > 0]

def detect_language(texts_json):
    texts = json.loads(texts_json)
    combined = " ".join(texts)
    result = pycld3.get_language(combined)
    return result.language if result else "unknown"

df['language'] = df['texts'].apply(detect_language)
```

2. **Use fastText** (requires more setup but better for short text):
```bash
pip install fasttext
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

---

## Citation

If you use this dataset or pipeline, please cite the original MTSD paper:

```bibtex
@article{MTSD2019,
  title={The Mapillary Traffic Sign Dataset for Detection and Classification on a Global Scale},
  author={Ertler, Christian and Mislej, Jerneja and Ollmann, Tobias and Porzi, Lorenzo and Neuhold, Gerhard and Kuang, Yubin},
  journal={arXiv preprint arXiv:1909.04422},
  year={2019}
}
```

---

## License

- **MTSD Dataset**: See [Mapillary's license terms](https://www.mapillary.com/dataset/trafficsign)
- **Preprocessing scripts**: MIT License (include your license here)

---

## Support

For issues with the data generation pipeline, please open an issue on this repository.

For questions about the original MTSD dataset, refer to the [official documentation](https://www.mapillary.com/dataset/trafficsign).
