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
import sys
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
GEO_MODEL_PATH  = REPO_ROOT / "GGAI" / "models" / "geo_classifier" / "sign_country_model.json"

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
TEXT_BOX_COLOR = "#2ecc71"

# Language → countries where it's the (or a major) road-sign language
LANG_COUNTRIES = {
    "en": ["US", "GB", "CA", "AU", "NZ", "ZA", "IE", "IN", "MY", "KE", "NG"],
    "es": ["ES", "MX", "CO", "CL", "AR", "PE"],
    "pt": ["PT", "BR"],
    "de": ["DE", "AT", "CH"],
    "fr": ["FR", "BE", "CH", "CA (Québec)"],
    "it": ["IT", "CH"],
    "nl": ["NL", "BE"],
    "sv": ["SE", "FI"],
    "da": ["DK"],
    "no": ["NO"], "nn": ["NO"],
    "fi": ["FI"],
    "is": ["IS"],
    "pl": ["PL"], "cs": ["CZ"], "sk": ["SK"], "hu": ["HU"], "ro": ["RO"],
    "hr": ["HR"], "sh": ["HR", "RS", "BA", "ME"], "bs": ["BA"],
    "sl": ["SI"], "sr": ["RS", "BA", "ME"], "bg": ["BG"], "el": ["GR"],
    "tr": ["TR"], "et": ["EE"], "lv": ["LV"], "lt": ["LT"],
    "ru": ["RU", "BY", "KZ"], "uk": ["UA"],
    "ja": ["JP"], "ko": ["KR"], "zh": ["TW", "CN", "SG", "MY"],
    "th": ["TH"], "vi": ["VN"], "id": ["ID"], "ms": ["MY"],
    "he": ["IL"], "ar": ["AE", "SA", "MA", "EG", "JO"],
    "af": ["ZA"], "tl": ["PH"], "ca": ["ES (Catalonia)"],
}

# Text/language pipeline is optional: Vision needs macOS, fastText needs lid.176.bin.
# OCR runs in a subprocess — Vision SIGBUSes if invoked in a process with torch/MPS.
TEXT_DETECTOR_SCRIPT = Path(__file__).parent / "text_detector.py"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "mapillary"))
try:
    from language_detector import LanguageDetector
    TEXT_PIPELINE_AVAILABLE = (sys.platform == "darwin"
                               and TEXT_DETECTOR_SCRIPT.exists())
except Exception:
    TEXT_PIPELINE_AVAILABLE = False

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


def sign_country_probs(country_model, sign_class, k=5):
    """P(country | single sign class) under uniform prior. Top-k (country, prob)."""
    ll = country_model["log_lik"].get(sign_class)
    if ll is None:
        return None
    mx = max(ll.values())
    exp = {c: np.exp(v - mx) for c, v in ll.items()}
    total = sum(exp.values())
    ranked = sorted(exp, key=exp.get, reverse=True)[:k]
    return [(c, exp[c] / total) for c in ranked]


def detect_signs(image_path, image, detector, classifier, idx2label, region_map,
                 device, conf, iou, top_k, country_model=None):
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
        country_probs = (sign_country_probs(country_model, top_label)
                         if country_model else None)

        detections.append({
            "box":           (x1, y1, x2, y2),
            "det_conf":      float(det_conf),
            "preds":         preds,
            "regions":       regions,
            "country_probs": country_probs,
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


# ── Text pipeline ─────────────────────────────────────────────────────────────

# Vision OCR confidence is tiered, not calibrated: ~1.0 = perfect head-on
# text, ~0.5 = legible but angled/stylized (most real signage), ~0.3 =
# marginal. 0.5 therefore means "legible"; 0.9 would hide most real text.
DEFAULT_TEXT_CONF = 0.5   # OCR confidence threshold for showing a text block
OCR_FLOOR = 0.3           # lines above this are collected for block merging


def merge_text_lines(lines):
    """Group vertically-adjacent, horizontally-aligned OCR lines into blocks.

    Vision OCR returns one observation per visual line, so a sentence laid
    out vertically (billboards, multi-line signs) arrives as fragments that
    are individually too short for reliable language ID. Merged blocks give
    fastText full-sentence context.

    Block membership: next line starts within 1.2× current line height below,
    and the two lines overlap horizontally by ≥30% of the narrower line.
    Returns blocks: {box, conf (max of members), text (joined), n_lines}.
    """
    lines = sorted(lines, key=lambda r: (r["box"][1], r["box"][0]))
    blocks = []

    for line in lines:
        x1, y1, x2, y2 = line["box"]
        placed = False
        for blk in blocks:
            bx1, by1, bx2, by2 = blk["box"]
            line_h = max(y2 - y1, 1)
            blk_line_h = max(blk["last_line_h"], 1)
            v_gap = y1 - by2
            overlap = min(x2, bx2) - max(x1, bx1)
            min_w = max(min(x2 - x1, bx2 - bx1), 1)
            if (v_gap <= 1.2 * max(line_h, blk_line_h)
                    and overlap >= 0.3 * min_w):
                blk["box"] = (min(bx1, x1), min(by1, y1),
                              max(bx2, x2), max(by2, y2))
                blk["texts"].append(line["text"])
                blk["conf"] = max(blk["conf"], line["conf"])
                blk["last_line_h"] = line_h
                placed = True
                break
        if not placed:
            blocks.append({
                "box": tuple(line["box"]),
                "texts": [line["text"]],
                "conf": line["conf"],
                "last_line_h": y2 - y1,
            })

    return [{"box": b["box"], "conf": b["conf"],
             "text": " ".join(b["texts"]), "n_lines": len(b["texts"])}
            for b in blocks]


def _ocr_env():
    """Subprocess env for the OCR worker. Strips DYLD_LIBRARY_PATH:
    homebrew dylibs shadow the system image libraries Vision needs,
    silently breaking OCR."""
    import os
    return {k: v for k, v in os.environ.items() if k != "DYLD_LIBRARY_PATH"}


class OCRWorker:
    """Persistent Vision OCR subprocess (one JSON request/response per line).

    Spawning per image costs ~0.8s of Python/pyobjc startup; this pays it
    once. Vision can't run in-process with torch/MPS (SIGBUS), hence a
    subprocess at all.
    """

    def __init__(self):
        import subprocess
        self.proc = subprocess.Popen(
            [sys.executable, str(TEXT_DETECTOR_SCRIPT), "--serve"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, env=_ocr_env())
        ready = json.loads(self.proc.stdout.readline())
        if not ready.get("ready"):
            raise RuntimeError("OCR worker failed to start")

    def ocr(self, image_path, min_conf):
        self.proc.stdin.write(json.dumps(
            {"image": str(image_path), "min_conf": min_conf}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("OCR worker died")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(resp.get("error", "OCR failed"))
        return resp["results"]

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.terminate()


def detect_texts(image_path, lang_detector, min_conf=DEFAULT_TEXT_CONF,
                 worker=None):
    """OCR via subprocess (Vision can't share a process with torch/MPS),
    merge lines into blocks, then run language ID per block in-process.

    Lines are collected down to OCR_FLOOR so multi-line sentences merge
    whole; only blocks whose best line reaches min_conf are shown.
    Pass a persistent OCRWorker to skip per-image subprocess startup.
    """
    import subprocess

    floor = min(OCR_FLOOR, min_conf)
    if worker is not None:
        try:
            ocr_results = worker.ocr(image_path, floor)
        except Exception as e:
            print(f"  OCR worker failed: {e}")
            return []
    else:
        proc = subprocess.run(
            [sys.executable, str(TEXT_DETECTOR_SCRIPT), str(image_path),
             "--json", "--min-conf", str(floor)],
            capture_output=True, text=True, timeout=60, env=_ocr_env())
        if proc.returncode != 0:
            print(f"  OCR subprocess failed: {proc.stderr.strip()[:200]}")
            return []
        ocr_results = json.loads(proc.stdout)

    blocks = merge_text_lines(ocr_results)
    blocks = [b for b in blocks if b["conf"] >= min_conf]

    detections = []
    for b in blocks:
        lang = lang_detector.detect(b["text"])
        top_langs = [(lang_detector.get_language_name(t["language"]),
                      t["confidence"]) for t in lang["top_languages"]]
        detections.append({
            "box":       b["box"],
            "ocr_conf":  b["conf"],
            "text":      b["text"],
            "n_lines":   b["n_lines"],
            "lang":      lang,
            "lang_name": lang_detector.get_language_name(lang["language"]),
            "top_langs": top_langs,
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
  .det:hover {{ background: rgba(255, 255, 255, 0.12); }}
  .det .tip-src {{ display: none; }}
  #gtip {{
    display: none; position: absolute;
    background: rgba(12, 12, 24, 0.96); border: 1px solid #555; border-radius: 6px;
    padding: 10px 12px; min-width: 260px; max-width: 380px; z-index: 99999;
    font-size: 12.5px; line-height: 1.5; box-shadow: 0 4px 16px rgba(0,0,0,0.6);
    pointer-events: none;
  }}
  #gtip h3 {{ margin: 0 0 6px; font-size: 13px; }}
  #gtip .conf {{ color: #8fd; }}
  #gtip .countries {{ color: #ffd700; }}
  #gtip .note {{ color: #aaa; font-style: italic; margin-top: 6px; }}
  #gtip ul {{ margin: 4px 0; padding-left: 16px; }}
  #gtip .cbar {{ display: flex; align-items: center; gap: 6px; margin: 2px 0; }}
  #gtip .cname {{ width: 28px; font-weight: 600; }}
  #gtip .bar {{ height: 8px; background: #ffd700; border-radius: 2px; display: inline-block; }}
  #gtip .cpct {{ color: #8fd; }}
</style>
</head>
<body>
<h1>Geo Inspector — {title}</h1>
<div class="legend">
  {legend} — hover any box for classification + country likelihood
</div>
<div class="stage">
  <img src="data:image/jpeg;base64,{img_b64}" width="{img_w}" height="{img_h}">
{boxes}
</div>
<div id="gtip"></div>
{script}
</body>
</html>
"""

# Single page-level tooltip: boxes keep area-sorted z-index for hit-testing
# (small inside big stays hoverable) and never get raised, while the tooltip
# always paints on top. Kept out of HTML_TEMPLATE so str.format doesn't
# require brace-escaping the JS.
TOOLTIP_SCRIPT = """<script>
const gtip = document.getElementById('gtip');
document.querySelectorAll('.det').forEach(d => {
  d.addEventListener('mouseenter', () => {
    gtip.innerHTML = d.querySelector('.tip-src').innerHTML;
    const r = d.getBoundingClientRect();
    gtip.style.display = 'block';
    gtip.style.left = Math.max(8, Math.min(r.left + window.scrollX,
      window.scrollX + window.innerWidth - gtip.offsetWidth - 16)) + 'px';
    let top = r.bottom + window.scrollY + 4;
    if (r.bottom + gtip.offsetHeight + 8 > window.innerHeight) {
      top = r.top + window.scrollY - gtip.offsetHeight - 4;
    }
    gtip.style.top = Math.max(window.scrollY + 4, top) + 'px';
  });
  d.addEventListener('mouseleave', () => { gtip.style.display = 'none'; });
});
</script>"""

BOX_TEMPLATE = """  <div class="det" style="left:{l:.2f}%;top:{t:.2f}%;width:{w:.2f}%;height:{h:.2f}%;border-color:{color};z-index:{z};">
    <div class="tip-src">{tip}</div>
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

    if det.get("country_probs"):
        bars = []
        for country, p in det["country_probs"]:
            barw = max(2, int(p * 100))
            bars.append(
                f"<div class='cbar'><span class='cname'>{html.escape(country)}</span>"
                f"<span class='bar' style='width:{barw}px'></span>"
                f"<span class='cpct'>{p:.0%}</span></div>")
        geo_html = ("<div class='countries'>Country likelihood (this sign alone):</div>"
                    + "".join(bars))
    else:
        regions = det["regions"]
        if regions:
            region_lines = [f"<li>{html.escape(SIGN_REGION_COUNTRIES.get(r, r))}</li>"
                            for r in regions]
            geo_html = ("<div class='countries'>Used in:</div><ul>"
                        + "".join(region_lines) + "</ul>"
                        "<div class='note'>No training data for this sign — "
                        "showing design regions instead.</div>")
        else:
            geo_html = "<div class='note'>No geo data for this sign design.</div>"

    return (f"<h3>🛑 Traffic sign <span class='conf'>(det {det['det_conf']:.0%})</span></h3>"
            f"<ul>{''.join(rows)}</ul>{geo_html}")


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


def text_tooltip(det):
    lang = det["lang"]
    lines_note = (f" · {det['n_lines']} lines merged"
                  if det.get("n_lines", 1) > 1 else "")
    header = (f"<h3>📝 Text <span class='conf'>(OCR {det['ocr_conf']:.0%}"
              f"{lines_note})</span></h3>"
              f"<div>&ldquo;{html.escape(det['text'])}&rdquo;</div>")

    if lang["is_numeric"]:
        return header + "<div class='note'>Numbers only — no language signal.</div>"
    if lang["is_short"]:
        return header + "<div class='note'>Too short for reliable language ID.</div>"

    top_langs = det.get("top_langs", [])
    tops_html = " · ".join(
        f"<b>{html.escape(name)}</b> <span class='conf'>{conf:.0%}</span>"
        for name, conf in top_langs)

    extras = []
    if lang.get("constrained"):
        chars = " ".join(lang.get("diacritics", []))
        extras.append(f"⚡ alphabet filter: {html.escape(chars)} — "
                      f"non-matching languages excluded")
    if lang["script"] not in ("latin", "unknown"):
        extras.append(f"script: {html.escape(lang['script'])}")
    extras_html = (f"<div class='note'>{' · '.join(extras)}</div>"
                   if extras else "")

    if lang["language"] == "unknown":
        weak = (f"<div class='note'>Weak guesses: {tops_html}</div>"
                if top_langs else "")
        return (header + "<div class='note'>Language unclear "
                "(below confidence threshold).</div>" + weak + extras_html)

    countries = LANG_COUNTRIES.get(lang["language"])
    if countries:
        country_html = (f"<div class='countries'>Likely: "
                        f"{html.escape(', '.join(countries))}</div>")
    else:
        country_html = ""
    return (header
            + f"<div>Language: {tops_html}</div>"
            + country_html + extras_html)


def render_html(image, sign_dets, lane_dets, title, text_dets=None):
    img_w, img_h = image.size
    text_dets = text_dets or []

    # Boxes are percentage-positioned against the original dimensions, so the
    # embedded image can be downscaled freely — faster encode, lighter page
    display = image
    if max(image.size) > 1600:
        display = image.copy()
        display.thumbnail((1600, 1600))

    buf = io.BytesIO()
    display.save(buf, format="JPEG", quality=88)
    img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    entries = []
    for det in sign_dets:
        entries.append((det["box"], SIGN_BOX_COLOR, sign_tooltip(det)))
    for det in lane_dets:
        entries.append((det["box"], LANE_BOX_COLORS[det["class"]], lane_tooltip(det)))
    for det in text_dets:
        entries.append((det["box"], TEXT_BOX_COLOR, text_tooltip(det)))

    # Smaller boxes stack above larger ones so nested detections (e.g. text
    # inside a sign) stay hoverable
    def area(box):
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)

    entries.sort(key=lambda e: area(e[0]), reverse=True)

    boxes = []
    for z, (box, color, tip) in enumerate(entries, start=10):
        l, t, w, h = pct_box(box, img_w, img_h)
        boxes.append(BOX_TEMPLATE.format(
            l=l, t=t, w=w, h=h, color=color, tip=tip, z=z))

    legend = (f"{len(sign_dets)} sign(s) · {len(lane_dets)} lane line(s)"
              + (f" · {len(text_dets)} text region(s)" if text_dets else ""))

    return HTML_TEMPLATE.format(
        title=html.escape(title), img_b64=img_b64,
        img_w=display.size[0], img_h=display.size[1],
        legend=legend,
        boxes="".join(boxes),
        script=TOOLTIP_SCRIPT)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Interactive sign + lane geo inspector")
    parser.add_argument("--image",      required=True)
    parser.add_argument("--detector",   default=str(DETECTOR_PATH))
    parser.add_argument("--classifier", default=str(CLASSIFIER_PATH))
    parser.add_argument("--label-map",  default=str(LABEL_MAP_PATH))
    parser.add_argument("--region-map", default=str(REGION_MAP_PATH))
    parser.add_argument("--lane-model", default=str(LANE_MODEL_PATH))
    parser.add_argument("--geo-model",  default=str(GEO_MODEL_PATH))
    parser.add_argument("--conf",       type=float, default=0.3)
    parser.add_argument("--iou",        type=float, default=0.45)
    parser.add_argument("--top-k",      type=int, default=3)
    parser.add_argument("--text-conf",  type=float, default=DEFAULT_TEXT_CONF,
                        help="OCR confidence threshold for text boxes")
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

    country_model = None
    if Path(args.geo_model).exists():
        with open(args.geo_model) as f:
            country_model = json.load(f)
        print(f"Loaded geo model:   {args.geo_model}")

    image = Image.open(image_path).convert("RGB")

    print(f"\nRunning sign pipeline on {image_path.name}...")
    sign_dets = detect_signs(image_path, image, detector, classifier, idx2label,
                             region_map, device, args.conf, args.iou, args.top_k,
                             country_model=country_model)
    print(f"  {len(sign_dets)} sign(s)")
    for i, d in enumerate(sign_dets):
        label, conf = d["preds"][0]
        print(f"  Sign #{i+1}: {label} ({conf:.0%}) regions={d['regions']}")

    print(f"\nRunning lane pipeline...")
    lane_dets = detect_lanes(image, lane_model, lane_ckpt, device)
    print(f"  {len(lane_dets)} lane line(s)")
    for d in lane_dets:
        print(f"  {d['class']:18s} area={d['area']} box={d['box']}")

    text_dets = []
    if TEXT_PIPELINE_AVAILABLE:
        print(f"\nRunning text pipeline...")
        lang_detector = LanguageDetector()
        text_dets = detect_texts(image_path, lang_detector, min_conf=args.text_conf)
        print(f"  {len(text_dets)} text region(s)")
        for d in text_dets:
            print(f"  {d['text']!r} → {d['lang']['language']} "
                  f"({d['lang']['confidence']:.0%})")
    else:
        print("\nText pipeline unavailable (needs macOS Vision + fasttext) — skipping.")

    html_doc = render_html(image, sign_dets, lane_dets, image_path.name,
                           text_dets=text_dets)
    out_path.write_text(html_doc)
    print(f"\nSaved inspector → {out_path}")
    print(f"Open with: open '{out_path}'")


if __name__ == "__main__":
    main()
