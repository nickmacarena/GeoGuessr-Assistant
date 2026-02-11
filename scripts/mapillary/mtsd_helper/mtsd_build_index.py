import argparse, json, os, glob, sys
import pandas as pd

DEFAULT_EXTS = [".jpg", ".jpeg", ".png"]

def load_jsons(ann_dirs, recursive=True):
    files = []
    for d in ann_dirs:
        pattern = os.path.join(d, "**", "*.json") if recursive else os.path.join(d, "*.json")
        files.extend(glob.glob(pattern, recursive=recursive))
    files = sorted(set(files))
    if not files:
        print("No JSON files found in:", ann_dirs, file=sys.stderr)
    return files

def norm_bbox_any(b):
    """Accept dict {x,y,w,h} or {xmin,ymin,xmax,ymax} or list [x,y,w,h]/[xmin,ymin,xmax,ymax]."""
    if b is None:
        return None
    if isinstance(b, dict):
        if all(k in b for k in ("x","y","w","h")):
            return [float(b["x"]), float(b["y"]), float(b["w"]), float(b["h"])]
        if all(k in b for k in ("xmin","ymin","xmax","ymax")):
            x, y, x2, y2 = float(b["xmin"]), float(b["ymin"]), float(b["xmax"]), float(b["ymax"])
            return [x, y, max(0.0, x2 - x), max(0.0, y2 - y)]
        return None
    if isinstance(b, (list, tuple)) and len(b) >= 4:
        x, y, a, b2 = b[:4]
        try:
            x, y, a, b2 = float(x), float(y), float(a), float(b2)
        except Exception:
            return None
        # Heuristic: if a>x and b2>y significantly, treat as [xmin,ymin,xmax,ymax], else [x,y,w,h]
        if a > x + 1 and b2 > y + 1:
            return [x, y, a - x, b2 - y]
        return [x, y, a, b2]
    return None

def parse_coco(js):
    images = {im["id"]: im.get("file_name") for im in js.get("images", []) if "id" in im and "file_name" in im}
    cats = {c["id"]: c.get("name", str(c["id"])) for c in js.get("categories", []) if "id" in c}
    rows = []
    for a in js.get("annotations", []):
        img_id = a.get("image_id")
        if img_id not in images:
            continue
        file_name = images[img_id]
        cat_name = cats.get(a.get("category_id"), "unknown")
        bbox = norm_bbox_any(a.get("bbox"))
        seg = a.get("segmentation")
        rows.append({
            "image_file": file_name,
            "category": cat_name,
            "bbox": bbox,
            "segmentation": seg,
            "source": "coco_like"
        })
    return rows

def parse_per_image(js, json_path):
    """
    Handle MTSD-style per-image JSON like your example:
    {
      "width": ..., "height": ..., "objects": [
         {"label": "...", "bbox": {"xmin":..., "ymin":..., "xmax":..., "ymax":...}}, ...
      ]
    }
    No file_name in JSON → derive from json filename.
    """
    rows = []
    if not isinstance(js, dict) or "objects" not in js:
        return rows
    base = os.path.splitext(os.path.basename(json_path))[0]  # e.g. <image_id>
    # We don't know the exact extension here; resolution happens later against --images_dir
    file_stub = base
    for obj in js.get("objects", []):
        label = obj.get("label") or obj.get("category") or "unknown"
        bb = norm_bbox_any(obj.get("bbox") or obj.get("bounding_box"))
        seg = obj.get("polygon") or obj.get("segmentation")
        rows.append({
            "image_file": file_stub,   # stub; will resolve to a real file by trying extensions
            "category": label,
            "bbox": bb,
            "segmentation": seg,
            "source": "per_image_json"
        })
    return rows

def resolve_image_path(images_dir, image_file, exts):
    """
    If image_file already has an extension, try that directly.
    Otherwise, try appending each ext from exts inside images_dir.
    Returns absolute path if exists, else None.
    """
    # If image_file has path components, strip to basename
    name = os.path.basename(str(image_file))
    root, ext = os.path.splitext(name)
    candidates = []
    if ext:
        candidates.append(os.path.join(images_dir, name))
    else:
        for e in exts:
            candidates.append(os.path.join(images_dir, root + e))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images_dir", required=True)
    ap.add_argument("--ann_dirs", nargs="+", required=True)
    ap.add_argument("--out_csv", default="mtsd_index.csv")
    ap.add_argument("--img_exts", default=".jpg,.jpeg,.png", help="comma-separated image extensions to try")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    img_exts = [e.strip().lower() if e.strip().startswith(".") else "." + e.strip().lower()
                for e in args.img_exts.split(",") if e.strip()]
    if not img_exts:
        img_exts = DEFAULT_EXTS

    json_files = load_jsons(args.ann_dirs, recursive=True)
    if args.debug:
        print(f"Found {len(json_files)} json files")

    all_rows = []
    for jf in json_files:
        try:
            with open(jf, "r") as f:
                js = json.load(f)
        except Exception as e:
            if args.debug:
                print(f"[WARN] Failed to read {jf}: {e}", file=sys.stderr)
            continue

        rows = []
        if isinstance(js, dict) and {"images","annotations"}.issubset(js.keys()):
            rows = parse_coco(js)
        if not rows:
            # Your case: per-image with top-level objects[]
            rows = parse_per_image(js, jf)

        if args.debug:
            print(f"[{os.path.basename(jf)}] parsed rows: {len(rows)}")

        all_rows.extend(rows)

    if not all_rows:
        print("No annotations parsed; exiting.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(all_rows).dropna(subset=["image_file"]).copy()

    # Resolve to absolute image paths using images_dir + inferred ext
    def resolver(x):
        p = resolve_image_path(args.images_dir, x, img_exts)
        return p

    df["image_path"] = df["image_file"].apply(resolver)
    # Drop rows whose image we couldn't find
    before = len(df)
    df = df[df["image_path"].notna()].reset_index(drop=True)
    missing = before - len(df)
    if args.debug and missing:
        print(f"[INFO] Dropped {missing} rows whose images were not found in {args.images_dir}")

    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df)} rows to {args.out_csv}")

if __name__ == "__main__":
    main()
