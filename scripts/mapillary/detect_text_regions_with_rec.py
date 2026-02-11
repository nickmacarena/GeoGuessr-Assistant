#!/usr/bin/env python3
# detect_text_regions_with_rec.py

import argparse
import os
import os.path as op
import json

import cv2
import pandas as pd
from paddleocr import PaddleOCR


def parse_bbox(bbox_str):
    """
    Parse MTSD bbox string: "[x, y, w, h]" -> (x, y, w, h) floats.
    """
    vals = json.loads(bbox_str)
    if not isinstance(vals, (list, tuple)) or len(vals) != 4:
        raise ValueError(f"Unexpected bbox format: {bbox_str}")
    x, y, w, h = map(float, vals)
    return x, y, w, h


def main():
    ap = argparse.ArgumentParser(
        description="Detect text regions + words in MTSD crops using PaddleOCR."
    )
    ap.add_argument(
        "--paths_csv",
        required=True,
        help="CSV with at least columns: image_path, bbox (MTSD format [x,y,w,h])",
    )
    ap.add_argument("--out_csv", required=True, help="Output CSV with OCR results")
    ap.add_argument(
        "--max_side",
        type=int,
        default=0,
        help="Resize longer side of crop to this inside PaddleOCR (0 = no limit)",
    )
    ap.add_argument(
        "--every",
        type=int,
        default=500,
        help="Flush results to CSV every N rows",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.paths_csv)

    if "image_path" not in df.columns or "bbox" not in df.columns:
        raise SystemExit("Expected columns 'image_path' and 'bbox' in CSV")

    # General OCR pipeline with detection + recognition
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    rows = []

    def flush(mode="a"):
        if not rows:
            return
        header = not op.exists(args.out_csv) or mode == "w"
        pd.DataFrame(rows).to_csv(
            args.out_csv,
            index=False,
            mode=mode,
            header=header,
        )
        rows.clear()

    total = len(df)
    #yeet = 0
    for idx, r in df.iterrows():
        #yeet += 1
        #if yeet == 20:
        #    break
        img_path = str(r["image_path"])
        bbox_str = str(r["bbox"])

        if not op.exists(img_path):
            rows.append(
                {
                    "image_path": img_path,
                    "bbox": bbox_str,
                    "n_regions": 0,
                    "boxes": "[]",
                    "texts": "[]",
                    "scores": "[]",
                    "error": "missing_file",
                }
            )
            continue

        try:
            x, y, w, h = parse_bbox(bbox_str)
        except Exception as e:
            rows.append(
                {
                    "image_path": img_path,
                    "bbox": bbox_str,
                    "n_regions": 0,
                    "boxes": "[]",
                    "texts": "[]",
                    "scores": "[]",
                    "error": f"bad_bbox: {e}",
                }
            )
            continue

        img = cv2.imread(img_path)
        if img is None:
            rows.append(
                {
                    "image_path": img_path,
                    "bbox": bbox_str,
                    "n_regions": 0,
                    "boxes": "[]",
                    "texts": "[]",
                    "scores": "[]",
                    "error": "imread_failed",
                }
            )
            continue

        H, W = img.shape[:2]
        x0 = max(0, int(round(x)))
        y0 = max(0, int(round(y)))
        x1 = min(W, int(round(x + w)))
        y1 = min(H, int(round(y + h)))

        if x1 <= x0 or y1 <= y0:
            rows.append(
                {
                    "image_path": img_path,
                    "bbox": bbox_str,
                    "n_regions": 0,
                    "boxes": "[]",
                    "texts": "[]",
                    "scores": "[]",
                    "error": "empty_crop",
                }
            )
            continue

        crop = img[y0:y1, x0:x1]

        try:
            predict_kwargs = {}
            if args.max_side and args.max_side > 0:
                predict_kwargs["text_det_limit_side_len"] = args.max_side
                predict_kwargs["text_det_limit_type"] = "max"

            result = ocr.predict(crop, **predict_kwargs)

            boxes_global = []
            texts = []
            scores = []

            if result:
                try:
                    res0 = result[0].json  # pipeline result object -> dict
                    inner = res0.get("res", res0)

                    # Some pipelines wrap everything under "overall_ocr_res"
                    overall = inner.get("overall_ocr_res", inner)

                    dt_polys = overall.get("dt_polys", []) or []
                    rec_texts = overall.get("rec_texts", []) or []
                    rec_scores = overall.get("rec_scores", []) or []

                    # Handle numpy arrays
                    if hasattr(rec_scores, "tolist"):
                        rec_scores = rec_scores.tolist()

                    # Convert polys to full-image coordinates and Python floats
                    dt_polys = [
                        [[float(px) + x0, float(py) + y0] for (px, py) in poly]
                        for poly in dt_polys
                    ]

                    # Zip ensures we only keep aligned entries
                    for poly, txt, sc in zip(dt_polys, rec_texts, rec_scores):
                        boxes_global.append(poly)
                        texts.append(str(txt))
                        scores.append(float(sc))

                except Exception as e:
                    rows.append(
                        {
                            "image_path": img_path,
                            "bbox": bbox_str,
                            "n_regions": 0,
                            "boxes": "[]",
                            "texts": "[]",
                            "scores": "[]",
                            "error": f"parse_error: {e}",
                        }
                    )
                    continue

            rows.append(
                {
                    "image_path": img_path,
                    "bbox": bbox_str,
                    "n_regions": len(boxes_global),
                    "boxes": json.dumps(boxes_global),
                    "texts": json.dumps(texts, ensure_ascii=False),
                    "scores": json.dumps(scores),
                    "error": "",
                }
            )

        except Exception as e:
            rows.append(
                {
                    "image_path": img_path,
                    "bbox": bbox_str,
                    "n_regions": 0,
                    "boxes": "[]",
                    "texts": "[]",
                    "scores": "[]",
                    "error": str(e),
                }
            )

        # periodic flush
        i1 = idx + 1
        if (i1 % args.every) == 0:
            flush("a" if i1 > args.every else "w")
            print(f"[ocr] processed {i1}/{total}")

    # final flush
    flush("a" if total > args.every else "w")
    print(f"Done. Wrote results for {total} crops → {args.out_csv}")


if __name__ == "__main__":
    main()
