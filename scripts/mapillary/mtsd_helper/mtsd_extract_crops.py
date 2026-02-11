
import argparse, os, os.path as op, pandas as pd, numpy as np, cv2, json
from tqdm import tqdm

def safe_crop(img, x, y, w, h, pad=2):
    H, W = img.shape[:2]
    x0 = max(int(x - pad), 0); y0 = max(int(y - pad), 0)
    x1 = min(int(x + w + pad), W); y1 = min(int(y + h + pad), H)
    return img[y0:y1, x0:x1], (x0, y0, x1, y1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--min_size", type=int, default=24, help="minimum short-side size to keep")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.index_csv)

    rows = []
    for i, r in tqdm(df.iterrows(), total=len(df)):
        img = cv2.imread(r["image_path"])
        if img is None:
            continue
        bbox = r.get("bbox")
        if isinstance(bbox, str):
            try:
                bbox = json.loads(bbox)
            except Exception:
                bbox = None
        if not bbox:
            continue
        x,y,w,h = bbox
        crop, (x0,y0,x1,y1) = safe_crop(img, x,y,w,h)
        h_, w_ = crop.shape[:2]
        if min(h_, w_) < args.min_size:
            continue
        # Save crop
        base_name = op.splitext(op.basename(r["image_file"]))[0]
        crop_name = f"{base_name}_{i}.jpg"
        crop_path = op.join(args.out_dir, crop_name)
        cv2.imwrite(crop_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        rows.append({
            "crop_path": crop_path,
            "image_path": r["image_path"],
            "category": r.get("category","unknown"),
            "bbox_abs": [x0,y0,x1-x0,y1-y0]
        })
    out_csv = op.join(args.out_dir, "crops_index.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {len(rows)} crops → {out_csv}")

if __name__ == "__main__":
    main()
