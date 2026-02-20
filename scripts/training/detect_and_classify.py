#!/usr/bin/env python3
"""
End-to-end sign detection + classification on a street-view image.

Pipeline:
  1. YOLOv8 detector  → bounding boxes (where are the signs?)
  2. EfficientNet-B0  → top-3 class predictions per box (what sign is it?)
  3. Render annotated image with boxes and predicted labels

Output: annotated JPEG saved alongside input image (or --out path).

Usage:
    DYLD_LIBRARY_PATH=/opt/homebrew/lib python detect_and_classify.py --image <path>
    DYLD_LIBRARY_PATH=/opt/homebrew/lib python detect_and_classify.py \
        --image <path> \
        --detector GGAI/models/sign_detector/yolov8n_mtsd/weights/best.pt \
        --classifier GGAI/models/sign_classifier/best_model.pt \
        --conf 0.3 \
        --top-k 3
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np


# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT        = Path(__file__).parent.parent.parent
DETECTOR_PATH    = REPO_ROOT / "GGAI" / "models" / "sign_detector" / "yolov8n_mtsd" / "weights" / "best.pt"
CLASSIFIER_PATH  = REPO_ROOT / "GGAI" / "models" / "sign_classifier" / "best_model.pt"
LABEL_MAP_PATH   = REPO_ROOT / "GGAI" / "models" / "sign_classifier" / "label_map.json"

CLASSIFIER_IMG_SIZE = 96
COLORS = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db",
          "#9b59b6", "#1abc9c", "#e91e63", "#ff5722", "#607d8b"]


# ── Classifier loading ────────────────────────────────────────────────────────

CLASSIFY_TF = transforms.Compose([
    transforms.Resize((CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE)),
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
def classify_crop(model, crop_img, idx2label, device, k=3):
    """Return list of (label, confidence) for top-k predictions."""
    x = CLASSIFY_TF(crop_img).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, k)
    return [(idx2label[str(i.item())], p.item()) for i, p in zip(top_idxs, top_probs)]


# ── Rendering ─────────────────────────────────────────────────────────────────

def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def label_short(label):
    """Last two parts of MTSD label for compact display."""
    parts = label.split("--")
    return "--".join(parts[-2:]) if len(parts) >= 2 else label


def draw_results(image, detections, font_size=14):
    """
    Draw detection boxes and top-3 classification results on image.

    detections: list of {box: (x1,y1,x2,y2), det_conf: float, preds: [(label, conf), ...]}
    """
    draw = ImageDraw.Draw(image)
    try:
        font      = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        font_bold = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size + 2)
    except Exception:
        font = font_bold = ImageFont.load_default()

    for i, det in enumerate(detections):
        x1, y1, x2, y2 = det["box"]
        color = COLORS[i % len(COLORS)]
        rgb   = hex_to_rgb(color)

        # Bounding box
        draw.rectangle([x1, y1, x2, y2], outline=rgb, width=3)

        # Build label text: top-3 predictions
        lines = []
        for rank, (label, conf) in enumerate(det["preds"]):
            prefix = "▶ " if rank == 0 else "  "
            lines.append(f"{prefix}{label_short(label)} {conf:.0%}")

        text = "\n".join(lines)

        # Background rect for text
        bbox_txt = draw.textbbox((x1, y1), text, font=font)
        pad = 3
        bg_box = [bbox_txt[0] - pad, bbox_txt[1] - pad,
                  bbox_txt[2] + pad, bbox_txt[3] + pad]
        # Keep label inside image
        if bg_box[1] < 0:
            shift = -bg_box[1]
            bg_box[1] += shift; bg_box[3] += shift
            bbox_txt = (bbox_txt[0], bbox_txt[1] + shift)

        draw.rectangle(bg_box, fill=(0, 0, 0, 180))
        draw.text((bbox_txt[0], bbox_txt[1]), text, fill=(255, 255, 255), font=font)

        # Small index badge
        badge = f"#{i+1}"
        draw.text((x1 + 4, y2 - font_size - 4), badge, fill=rgb, font=font_bold)

    return image


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Detect and classify traffic signs")
    parser.add_argument("--image",      required=True, help="Path to input street-view image")
    parser.add_argument("--detector",   default=str(DETECTOR_PATH))
    parser.add_argument("--classifier", default=str(CLASSIFIER_PATH))
    parser.add_argument("--label-map",  default=str(LABEL_MAP_PATH))
    parser.add_argument("--conf",       type=float, default=0.3,
                        help="Detector confidence threshold")
    parser.add_argument("--iou",        type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--top-k",      type=int, default=3)
    parser.add_argument("--out",        default=None,
                        help="Output image path (default: <input>_annotated.jpg)")
    args = parser.parse_args()

    image_path = Path(args.image)
    out_path   = Path(args.out) if args.out else image_path.with_stem(image_path.stem + "_annotated")

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Load models ───────────────────────────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: pip install ultralytics")
        return

    print(f"Loading detector:   {args.detector}")
    detector = YOLO(args.detector)

    print(f"Loading classifier: {args.classifier}")
    classifier = load_classifier(Path(args.classifier), device)

    with open(args.label_map) as f:
        idx2label = json.load(f)["idx2label"]

    # ── Detect ────────────────────────────────────────────────────────────────
    print(f"\nRunning detector on {image_path.name}...")
    results = detector(str(image_path), conf=args.conf, iou=args.iou, verbose=False)
    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()   # (N, 4)
    det_confs  = results[0].boxes.conf.cpu().numpy()   # (N,)
    print(f"  Found {len(boxes_xyxy)} sign(s)")

    if len(boxes_xyxy) == 0:
        print("  No signs detected. Try lowering --conf.")
        return

    # ── Classify each box ─────────────────────────────────────────────────────
    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    detections = []
    for i, (box, det_conf) in enumerate(zip(boxes_xyxy, det_confs)):
        x1, y1, x2, y2 = map(int, box)
        # Clamp to image
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img_w, x2); y2 = min(img_h, y2)

        crop = image.crop((x1, y1, x2, y2))
        preds = classify_crop(classifier, crop, idx2label, device, k=args.top_k)

        detections.append({
            "box":      (x1, y1, x2, y2),
            "det_conf": float(det_conf),
            "preds":    preds,
        })

        top_label, top_conf = preds[0]
        print(f"  Sign #{i+1}: det={det_conf:.2f}  "
              f"→ {label_short(top_label)} ({top_conf:.1%})")

    # ── Render + save ─────────────────────────────────────────────────────────
    annotated = image.copy()
    annotated = draw_results(annotated, detections)
    annotated.save(out_path, quality=95)
    print(f"\nSaved annotated image → {out_path}")


if __name__ == "__main__":
    main()
