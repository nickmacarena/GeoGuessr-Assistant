#!/usr/bin/env python3
"""
Interactive geo inspector: signs + lane lines on one street-view image.

Runs both pipelines independently (no fusion):
  1. YOLOv8 detector → EfficientNet-B0 classifier  → sign boxes
  2. DeepLabV3 lane segmentation (9-class)         → lane line boxes
     (connected components per class → one box per line)

Output: standalone HTML file. Hover any box to see its classification
and the countries where that sign design / lane marking is used.

Usage:
    DYLD_LIBRARY_PATH=/opt/homebrew/lib python geo_inspector.py --image <path>
    python geo_inspector.py --image <path> --out report.html --conf 0.3
"""

import argparse
import base64
import html
import io
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
from PIL import Image


# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).parent.parent.parent
DETECTOR_PATH   = REPO_ROOT / "GGAI" / "models" / "sign_detector" / "yolov8n_mtsd" / "best_model.pt"
CLASSIFIER_PATH = REPO_ROOT / "GGAI" / "models" / "sign_classifier" / "best_model.pt"
LABEL_MAP_PATH  = REPO_ROOT / "GGAI" / "models" / "sign_classifier" / "label_map.json"
REGION_MAP_PATH = REPO_ROOT / "GGAI" / "models" / "sign_classifier" / "region_mapping.json"
LANE_MODEL_PATH = REPO_ROOT / "GGAI" / "models" / "lane_segmentation_v4" / "best_model.pt"

CLASSIFIER_IMG_SIZE = 96

# ── Geo knowledge (static lookups, not learned) ───────────────────────────────

SIGN_REGION_COUNTRIES = {
    "us": "United States + MUTCD-style (MX, parts of Latin America)",
    "ca": "Canada",
    "eu": "Europe (Vienna Convention)",
    "au": "Australia / New Zealand",
    "br": "Brazil",
}

LANE_INFO = {
    "s_white_solid": {
        "display": "Single white solid",
        "countries": [],
        "note": "Universal — edge lines / lane boundaries worldwide. Low geo signal.",
    },
    "s_white_dashed": {
        "display": "Single white dashed",
        "countries": [],
        "note": "Universal — same-direction lane separator worldwide. As a CENTERLINE "
                "it suggests EU/UK/AU rather than the Americas.",
    },
    "d_white_solid": {
        "display": "Double white solid",
        "countries": ["UK", "EU (no-crossing)", "JP", "CN"],
        "note": "No-crossing barrier line in countries with white centerlines.",
    },
    "d_white_dashed": {
        "display": "Double white dashed",
        "countries": ["EU (lane-change advisory)"],
        "note": "Rare everywhere — often reversible or advisory lanes.",
    },
    "s_yellow_solid": {
        "display": "Single yellow solid",
        "countries": ["US", "CA", "MX", "JP", "KR", "TW", "Latin America", "NO", "IS"],
        "note": "Centerline separating opposing traffic. Strong Americas/East-Asia "
                "signal; in most of EU yellow means roadworks (NO/IS excepted).",
    },
    "s_yellow_dashed": {
        "display": "Single yellow dashed",
        "countries": ["US", "CA", "MX"],
        "note": "Passing-allowed centerline — very strong North America signal.",
    },
    "d_yellow_solid": {
        "display": "Double yellow solid",
        "countries": ["US", "CA", "MX", "JP", "KR", "Latin America"],
        "note": "No-passing centerline. Strong Americas signal; sparse in JP/KR.",
    },
    "d_yellow_dashed": {
        "display": "Double yellow dashed",
        "countries": ["US", "CA"],
        "note": "Mixed solid/dashed passing zones — strong North America signal.",
    },
}

LANE_BOX_COLORS = {
    "s_white_solid":   "#ffffff",
    "s_white_dashed":  "#b0b0b0",
    "d_white_solid":   "#9bd0ff",
    "d_white_dashed":  "#6f9fc8",
    "s_yellow_solid":  "#ffe04d",
    "s_yellow_dashed": "#c8b400",
    "d_yellow_solid":  "#ffa500",
    "d_yellow_dashed": "#c87800",
}

SIGN_BOX_COLOR = "#e74c3c"

SEG_IMG_SIZE = (512, 256)            # (W, H) — must match training
MIN_COMPONENT_AREA = 50              # px in seg-mask space (matches presence threshold)

SEG_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

CLASSIFY_TF = transforms.Compose([
    transforms.Resize((CLASSIFIER_IMG_SIZE, CLASSIFIER_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Sign pipeline ─────────────────────────────────────────────────────────────

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
    x = CLASSIFY_TF(crop_img).unsqueeze(0).to(device)
    probs = F.softmax(model(x), dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, k)
    return [(idx2label[str(i.item())], p.item()) for i, p in zip(top_idxs, top_probs)]


def detect_signs(image_path, image, detector, classifier, idx2label, region_map,
                 device, conf, iou, top_k):
    """Run detector + classifier; return list of sign detection dicts."""
    img_w, img_h = image.size
    results = detector(str(image_path), conf=conf, iou=iou, verbose=False)
    boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
    det_confs  = results[0].boxes.conf.cpu().numpy()

    detections = []
    for box, det_conf in zip(boxes_xyxy, det_confs):
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(img_w, x2); y2 = min(img_h, y2)

        crop  = image.crop((x1, y1, x2, y2))
        preds = classify_crop(classifier, crop, idx2label, device, k=top_k)

        top_label  = preds[0][0]
        regions    = region_map.get(top_label, {}).get("regions", [])

        detections.append({
            "box":      (x1, y1, x2, y2),
            "det_conf": float(det_conf),
            "preds":    preds,
            "regions":  regions,
        })
    return detections


# ── Lane pipeline ─────────────────────────────────────────────────────────────

def load_lane_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]
    model = deeplabv3_mobilenet_v3_large(weights=None, aux_loss=True)
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(10, num_classes, kernel_size=1)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), ckpt


@torch.no_grad()
def detect_lanes(image, lane_model, ckpt, device):
    """Run segmentation; return per-line boxes via connected components.

    Boxes are returned in original-image pixel coordinates.
    """
    class_names = ckpt["class_names"]
    img_w, img_h = image.size

    img_small = image.resize(SEG_IMG_SIZE, Image.BILINEAR)
    x = SEG_TF(img_small).unsqueeze(0).to(device)
    out = lane_model(x)["out"]
    pred = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)  # (H, W)

    sx = img_w / SEG_IMG_SIZE[0]
    sy = img_h / SEG_IMG_SIZE[1]

    detections = []
    for class_id in range(1, len(class_names)):
        cname = class_names[class_id]
        binary = (pred == class_id).astype(np.uint8)
        if binary.sum() < MIN_COMPONENT_AREA:
            continue

        n_comp, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for c in range(1, n_comp):
            area = stats[c, cv2.CC_STAT_AREA]
            if area < MIN_COMPONENT_AREA:
                continue
            bx = stats[c, cv2.CC_STAT_LEFT]
            by = stats[c, cv2.CC_STAT_TOP]
            bw = stats[c, cv2.CC_STAT_WIDTH]
            bh = stats[c, cv2.CC_STAT_HEIGHT]

            detections.append({
                "class":  cname,
                "box":    (int(bx * sx), int(by * sy),
                           int((bx + bw) * sx), int((by + bh) * sy)),
                "area":   int(area),
            })
    return detections


# ── HTML rendering ────────────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Geo Inspector — {title}</title>
<style>
  body {{
    background: #1a1a2e; color: #eee; font-family: Helvetica, Arial, sans-serif;
    margin: 0; padding: 24px;
  }}
  h1 {{ font-size: 18px; font-weight: 600; }}
  .legend {{ font-size: 13px; color: #aaa; margin-bottom: 12px; }}
  .stage {{ position: relative; display: inline-block; max-width: 100%; }}
  .stage img {{ display: block; max-width: 100%; height: auto; }}
  .det {{
    position: absolute; border: 2px solid; border-radius: 2px;
    box-sizing: border-box; cursor: crosshair;
  }}
  .det:hover {{ background: rgba(255, 255, 255, 0.12); z-index: 50; }}
  .det .tip {{
    display: none; position: absolute; left: 0; top: 100%; margin-top: 4px;
    background: rgba(12, 12, 24, 0.96); border: 1px solid #555; border-radius: 6px;
    padding: 10px 12px; min-width: 260px; max-width: 380px; z-index: 100;
    font-size: 12.5px; line-height: 1.5; box-shadow: 0 4px 16px rgba(0,0,0,0.6);
  }}
  .det:hover .tip {{ display: block; }}
  .det.flip .tip {{ top: auto; bottom: 100%; margin-top: 0; margin-bottom: 4px; }}
  .tip h3 {{ margin: 0 0 6px; font-size: 13px; }}
  .tip .conf {{ color: #8fd; }}
  .tip .countries {{ color: #ffd700; }}
  .tip .note {{ color: #aaa; font-style: italic; margin-top: 6px; }}
  .tip ul {{ margin: 4px 0; padding-left: 16px; }}
</style>
</head>
<body>
<h1>Geo Inspector — {title}</h1>
<div class="legend">
  {n_signs} sign(s) · {n_lanes} lane line(s) — hover any box for classification + country likelihood
</div>
<div class="stage">
  <img src="data:image/jpeg;base64,{img_b64}" width="{img_w}" height="{img_h}">
{boxes}
</div>
</body>
</html>
"""

BOX_TEMPLATE = """  <div class="det{flip}" style="left:{l:.2f}%;top:{t:.2f}%;width:{w:.2f}%;height:{h:.2f}%;border-color:{color};">
    <div class="tip">{tip}</div>
  </div>
"""


def pct_box(box, img_w, img_h):
    x1, y1, x2, y2 = box
    return (100 * x1 / img_w, 100 * y1 / img_h,
            100 * (x2 - x1) / img_w, 100 * (y2 - y1) / img_h)


def sign_tooltip(det):
    rows = []
    for rank, (label, conf) in enumerate(det["preds"]):
        marker = "▶ " if rank == 0 else "&nbsp;&nbsp;"
        rows.append(f"<li>{marker}{html.escape(label)} "
                    f"<span class='conf'>{conf:.0%}</span></li>")

    regions = det["regions"]
    if regions:
        region_lines = [f"<li>{html.escape(SIGN_REGION_COUNTRIES.get(r, r))}</li>"
                        for r in regions]
        region_html = ("<div class='countries'>Used in:</div><ul>"
                       + "".join(region_lines) + "</ul>")
        if set(regions) == {"us", "eu"}:
            region_html += ("<div class='note'>Design used in both US and EU "
                            "sign systems — weak geo signal.</div>")
    else:
        region_html = "<div class='note'>No region data for this sign design.</div>"

    return (f"<h3>🛑 Traffic sign <span class='conf'>(det {det['det_conf']:.0%})</span></h3>"
            f"<ul>{''.join(rows)}</ul>{region_html}")


def lane_tooltip(det):
    info = LANE_INFO[det["class"]]
    if info["countries"]:
        countries = ", ".join(info["countries"])
        country_html = f"<div class='countries'>Likely: {html.escape(countries)}</div>"
    else:
        country_html = "<div class='countries'>Worldwide</div>"
    return (f"<h3>🛣 {html.escape(info['display'])}</h3>"
            f"{country_html}"
            f"<div class='note'>{html.escape(info['note'])}</div>")


def render_html(image, sign_dets, lane_dets, title):
    img_w, img_h = image.size

    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=88)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    boxes = []
    for det in sign_dets:
        l, t, w, h = pct_box(det["box"], img_w, img_h)
        flip = " flip" if t + h > 75 else ""
        boxes.append(BOX_TEMPLATE.format(
            flip=flip, l=l, t=t, w=w, h=h,
            color=SIGN_BOX_COLOR, tip=sign_tooltip(det)))

    for det in lane_dets:
        l, t, w, h = pct_box(det["box"], img_w, img_h)
        flip = " flip" if t + h > 75 else ""
        boxes.append(BOX_TEMPLATE.format(
            flip=flip, l=l, t=t, w=w, h=h,
            color=LANE_BOX_COLORS[det["class"]], tip=lane_tooltip(det)))

    return HTML_TEMPLATE.format(
        title=html.escape(title), img_b64=img_b64,
        img_w=img_w, img_h=img_h,
        n_signs=len(sign_dets), n_lanes=len(lane_dets),
        boxes="".join(boxes))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactive sign + lane geo inspector")
    parser.add_argument("--image",      required=True)
    parser.add_argument("--detector",   default=str(DETECTOR_PATH))
    parser.add_argument("--classifier", default=str(CLASSIFIER_PATH))
    parser.add_argument("--label-map",  default=str(LABEL_MAP_PATH))
    parser.add_argument("--region-map", default=str(REGION_MAP_PATH))
    parser.add_argument("--lane-model", default=str(LANE_MODEL_PATH))
    parser.add_argument("--conf",       type=float, default=0.3)
    parser.add_argument("--iou",        type=float, default=0.45)
    parser.add_argument("--top-k",      type=int, default=3)
    parser.add_argument("--out",        default=None,
                        help="Output HTML path (default: <input>_inspector.html)")
    args = parser.parse_args()

    image_path = Path(args.image)
    out_path = Path(args.out) if args.out else image_path.with_suffix("").with_name(
        image_path.stem + "_inspector.html")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    from ultralytics import YOLO

    print(f"Loading detector:   {args.detector}")
    detector = YOLO(args.detector)
    print(f"Loading classifier: {args.classifier}")
    classifier = load_classifier(Path(args.classifier), device)
    print(f"Loading lane model: {args.lane_model}")
    lane_model, lane_ckpt = load_lane_model(Path(args.lane_model), device)

    with open(args.label_map) as f:
        idx2label = json.load(f)["idx2label"]
    with open(args.region_map) as f:
        region_map = json.load(f)

    image = Image.open(image_path).convert("RGB")

    print(f"\nRunning sign pipeline on {image_path.name}...")
    sign_dets = detect_signs(image_path, image, detector, classifier, idx2label,
                             region_map, device, args.conf, args.iou, args.top_k)
    print(f"  {len(sign_dets)} sign(s)")
    for i, d in enumerate(sign_dets):
        label, conf = d["preds"][0]
        print(f"  Sign #{i+1}: {label} ({conf:.0%}) regions={d['regions']}")

    print(f"\nRunning lane pipeline...")
    lane_dets = detect_lanes(image, lane_model, lane_ckpt, device)
    print(f"  {len(lane_dets)} lane line(s)")
    for d in lane_dets:
        print(f"  {d['class']:18s} area={d['area']} box={d['box']}")

    html_doc = render_html(image, sign_dets, lane_dets, image_path.name)
    out_path.write_text(html_doc)
    print(f"\nSaved inspector → {out_path}")
    print(f"Open with: open '{out_path}'")


if __name__ == "__main__":
    main()
