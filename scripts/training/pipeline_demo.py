#!/usr/bin/env python3
"""
Full pipeline demo on a validation image.

Shows four panels:
  1. Original street view
  2. Detected signs (YOLOv8 bounding boxes)
  3. Classified sign crops (EfficientNet-B0 top-1)
  4. Reference SVG icons from mapillary_sprite_source

Usage:
    DYLD_LIBRARY_PATH=/opt/homebrew/lib python pipeline_demo.py
    DYLD_LIBRARY_PATH=/opt/homebrew/lib python pipeline_demo.py --image <path>
    DYLD_LIBRARY_PATH=/opt/homebrew/lib python pipeline_demo.py --seed 7 --conf 0.25
"""

import argparse
import io
import json
import random
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

try:
    import cairosvg
    HAS_CAIRO = True
except ImportError:
    HAS_CAIRO = False
    print("WARNING: cairosvg not found — SVG cells will show label text")

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).parent.parent.parent
VAL_DIR         = REPO_ROOT / "data" / "mapillary" / "yolo_detection" / "images" / "val"
DETECTOR_PATH   = REPO_ROOT / "GGAI" / "models" / "sign_detector" / "yolov8n_mtsd" / "best_model.pt"
CLASSIFIER_PATH = REPO_ROOT / "GGAI" / "models" / "sign_classifier" / "best_model.pt"
LABEL_MAP_PATH  = REPO_ROOT / "GGAI" / "models" / "sign_classifier" / "label_map.json"
OUT_PATH        = REPO_ROOT / "GGAI" / "models" / "pipeline_demo.png"

SVG_BASE = (
    "https://raw.githubusercontent.com/mapillary/mapillary_sprite_source"
    "/master/package_signs/{label}.svg"
)

COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#e67e22", "#9b59b6",
          "#1abc9c", "#e91e63", "#f39c12"]
CELL_SIZE = 180


# ── Device ────────────────────────────────────────────────────────────────────
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Classifier ────────────────────────────────────────────────────────────────
CLASSIFY_TF = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_classifier(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    backbone    = ckpt.get("backbone", "efficientnet_b0")
    num_classes = ckpt["num_classes"]
    if backbone == "efficientnet_b2":
        m = models.efficientnet_b2(weights=None)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, num_classes)
    elif backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, num_classes)
    elif backbone == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=None)
        m.classifier[3] = torch.nn.Linear(m.classifier[3].in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    m.load_state_dict(ckpt["model"])
    return m.to(device).eval()


@torch.no_grad()
def classify(model, crop_img, idx2label, device, k=3):
    x = CLASSIFY_TF(crop_img).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, k)
    return [(idx2label[str(i.item())], p.item()) for i, p in zip(top_idxs, top_probs)]


# ── SVG icons ─────────────────────────────────────────────────────────────────
_svg_cache: dict = {}


def fetch_svg(label: str, size: int = CELL_SIZE):
    if label not in _svg_cache:
        url = SVG_BASE.format(label=label)
        try:
            r = requests.get(url, timeout=10)
            _svg_cache[label] = r.content if r.status_code == 200 else None
        except Exception:
            _svg_cache[label] = None

    svg_bytes = _svg_cache[label]
    if svg_bytes is None or not HAS_CAIRO:
        return None
    try:
        png = cairosvg.svg2png(bytestring=svg_bytes, output_width=size, output_height=size)
        img = Image.open(io.BytesIO(png)).convert("RGBA")
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        return bg.convert("RGB")
    except Exception:
        return None


# ── Drawing helpers ───────────────────────────────────────────────────────────
def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def draw_boxes(image, detections):
    img  = image.copy()
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
    except Exception:
        font = ImageFont.load_default()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        rgb = hex_rgb(COLORS[i % len(COLORS)])
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=4)
        badge = f" #{i+1} "
        bbox = draw.textbbox((x1, y1 - 24), badge, font=font)
        draw.rectangle(bbox, fill=rgb)
        draw.text((bbox[0], bbox[1]), badge, fill=(255, 255, 255), font=font)
    return img


def label_short(label: str) -> str:
    parts = label.split("--")
    return "--".join(parts[-2:]) if len(parts) >= 2 else label


def label_display(label: str) -> str:
    """Two-line display: category on first line, type--variant on second."""
    parts = label.split("--")
    if len(parts) >= 3:
        return f"{parts[0]}\n{'--'.join(parts[1:])}"
    return label


# ── Val image selection ───────────────────────────────────────────────────────
def pick_image(detector, conf, seed, min_signs=2, max_signs=5, max_tries=30):
    images = sorted(VAL_DIR.glob("*.jpg"))
    rng = random.Random(seed)
    rng.shuffle(images)

    best = None  # fallback: best single-detection image
    for img_path in images[:max_tries]:
        results  = detector(str(img_path), conf=conf, iou=0.45, verbose=False)
        boxes    = results[0].boxes.xyxy.cpu().numpy()
        confs    = results[0].boxes.conf.cpu().numpy()
        n = len(boxes)
        if min_signs <= n <= max_signs:
            print(f"  Selected: {img_path.name}  ({n} signs)")
            return img_path, boxes, confs
        if best is None and n >= 1:
            best = (img_path, boxes, confs)

    if best:
        img_path, boxes, confs = best
        print(f"  Fallback: {img_path.name}  ({len(boxes)} signs)")
        return best
    return None, None, None


# ── Render ────────────────────────────────────────────────────────────────────
def render(original, annotated, detections, img_name, out_path):
    n = len(detections)

    # Scale for display (cap width at 900px)
    max_w = 900
    iw, ih = original.size
    scale  = min(1.0, max_w / iw)
    dw, dh = int(iw * scale), int(ih * scale)
    orig_disp = original.resize((dw, dh), Image.LANCZOS)
    ann_disp  = annotated.resize((dw, dh), Image.LANCZOS)

    fig_w = max(14, n * (CELL_SIZE / 72) + 2)
    fig   = plt.figure(figsize=(fig_w, 13), facecolor="white")

    # Two independent gridspecs so top images and bottom cells size independently
    gs_top = gridspec.GridSpec(
        1, 2, figure=fig,
        left=0.03, right=0.97, top=0.93, bottom=0.52,
        wspace=0.06,
    )
    gs_bot = gridspec.GridSpec(
        2, n, figure=fig,
        left=0.03, right=0.97, top=0.47, bottom=0.03,
        wspace=0.08, hspace=0.38,
    )

    # ── Row 1: original + annotated ──────────────────────────────────────────
    ax_orig = fig.add_subplot(gs_top[0, 0])
    ax_ann  = fig.add_subplot(gs_top[0, 1])

    ax_orig.imshow(np.array(orig_disp))
    ax_orig.set_title("1. Original Street View", fontsize=13, fontweight="bold", pad=8)
    ax_orig.axis("off")

    ax_ann.imshow(np.array(ann_disp))
    ax_ann.set_title("2. Detected Signs (YOLOv8)", fontsize=13, fontweight="bold", pad=8)
    ax_ann.axis("off")

    # ── Rows 2-3: per-sign crops and reference icons ──────────────────────────
    for i, det in enumerate(detections):
        color = COLORS[i % len(COLORS)]
        label, conf_score = det["preds"][0]
        short = label_short(label)
        disp  = label_display(label)

        # Row 2: detected crop
        ax_c = fig.add_subplot(gs_bot[0, i])
        crop_resized = det["crop"].resize((CELL_SIZE, CELL_SIZE), Image.LANCZOS)
        ax_c.imshow(np.array(crop_resized))
        ax_c.set_title(
            f"#{i+1}  {conf_score:.0%}\n{short}",
            fontsize=8.5, pad=5, fontweight="bold",
        )
        ax_c.axis("off")
        for sp in ax_c.spines.values():
            sp.set_visible(True); sp.set_edgecolor(color); sp.set_linewidth(3)

        # Row 3: reference SVG icon
        ax_s = fig.add_subplot(gs_bot[1, i])
        if det["svg"] is not None:
            ax_s.imshow(np.array(det["svg"]))
        else:
            ax_s.set_facecolor("#f0f0f0")
            ax_s.text(0.5, 0.5, disp, ha="center", va="center",
                      fontsize=7, transform=ax_s.transAxes)
        ax_s.set_title(f"#{i+1} Reference\n{short}", fontsize=8.5, pad=5)
        ax_s.axis("off")
        for sp in ax_s.spines.values():
            sp.set_visible(True); sp.set_edgecolor(color); sp.set_linewidth(3)

    # Row labels
    fig.text(0.005, 0.375, "3. Classified\n    Crops", va="center",
             fontsize=10, fontweight="bold", rotation=90, color="#333")
    fig.text(0.005, 0.155, "4. Reference\n    Icons", va="center",
             fontsize=10, fontweight="bold", rotation=90, color="#333")

    fig.suptitle(f"Full Pipeline Demo — {img_name}", fontsize=14, y=0.97)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  default=None, help="Path to a specific image")
    parser.add_argument("--conf",   type=float, default=0.3)
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--out",    default=str(OUT_PATH))
    args = parser.parse_args()

    from ultralytics import YOLO

    device = get_device()
    print(f"Device: {device}")

    print("Loading detector...")
    detector = YOLO(str(DETECTOR_PATH))

    print("Loading classifier...")
    classifier = load_classifier(CLASSIFIER_PATH, device)

    with open(LABEL_MAP_PATH) as f:
        idx2label = json.load(f)["idx2label"]

    # Get image + raw detections
    if args.image:
        img_path = Path(args.image)
        results     = detector(str(img_path), conf=args.conf, iou=0.45, verbose=False)
        boxes_xyxy  = results[0].boxes.xyxy.cpu().numpy()
        det_confs   = results[0].boxes.conf.cpu().numpy()
        print(f"  {img_path.name}: {len(boxes_xyxy)} sign(s) detected")
    else:
        print(f"Searching val set (seed={args.seed}, conf≥{args.conf})...")
        img_path, boxes_xyxy, det_confs = pick_image(
            detector, args.conf, args.seed
        )
        if img_path is None:
            print("ERROR: no suitable image found")
            return

    if len(boxes_xyxy) == 0:
        print("No signs detected — try lowering --conf")
        return

    original = Image.open(img_path).convert("RGB")
    iw, ih   = original.size

    # Classify each detection
    detections = []
    for i, (box, det_conf) in enumerate(zip(boxes_xyxy, det_confs)):
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(iw, x2); y2 = min(ih, y2)
        crop  = original.crop((x1, y1, x2, y2))
        preds = classify(classifier, crop, idx2label, device, k=3)
        detections.append({
            "box":      (x1, y1, x2, y2),
            "det_conf": float(det_conf),
            "crop":     crop,
            "preds":    preds,
        })
        print(f"  Sign #{i+1}: det={det_conf:.2f}  → {label_short(preds[0][0])} ({preds[0][1]:.0%})")

    # Fetch reference SVGs
    print("Fetching reference SVG icons...")
    for det in detections:
        det["svg"] = fetch_svg(det["preds"][0][0])
        status = "OK" if det["svg"] else "no icon"
        print(f"  {label_short(det['preds'][0][0])}: {status}")

    # Annotated image
    annotated = draw_boxes(original, detections)

    # Render
    render(original, annotated, detections, img_path.name, Path(args.out))


if __name__ == "__main__":
    main()