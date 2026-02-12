#!/usr/bin/env python3
"""
Visualize sign classifier predictions on random crop samples.

For each sampled crop:
  - Runs best_model.pt and gets top-3 predictions with confidence scores
  - Fetches the reference SVG for each predicted class from mapillary_sprite_source
  - Renders a grid row: [crop] [pred1 SVG + conf%] [pred2 SVG + conf%] [pred3 SVG + conf%]

Output: GGAI/models/sign_classifier/prediction_samples.png

Usage:
    DYLD_LIBRARY_PATH=/opt/homebrew/lib python visualize_predictions.py
    DYLD_LIBRARY_PATH=/opt/homebrew/lib python visualize_predictions.py --n 16 --seed 7
    DYLD_LIBRARY_PATH=/opt/homebrew/lib python visualize_predictions.py --label regulatory--stop--g1
"""

import argparse
import csv
import io
import json
import random
from pathlib import Path

import requests
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    import cairosvg
    HAS_CAIRO = True
except ImportError:
    HAS_CAIRO = False
    print("WARNING: cairosvg not available — SVG cells will show label text only")


# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent.parent
DATA_ROOT   = REPO_ROOT / "data" / "mapillary"
CROPS_DIR   = DATA_ROOT / "crops"
CROPS_INDEX = CROPS_DIR / "crops_index.csv"
MODEL_DIR   = REPO_ROOT / "GGAI" / "models" / "sign_classifier"
MODEL_PATH  = MODEL_DIR / "best_model.pt"
LABEL_MAP   = MODEL_DIR / "label_map.json"

SVG_BASE = (
    "https://raw.githubusercontent.com/mapillary/mapillary_sprite_source"
    "/master/package_signs/{label}.svg"
)

IMG_SIZE = 96
TOP_K    = 3


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(device):
    ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=False)
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
    m.to(device).eval()
    return m


# ── Inference ─────────────────────────────────────────────────────────────────

VAL_TF = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@torch.no_grad()
def predict_top_k(model, img_path, idx2label, device, k=TOP_K):
    """Returns list of (label, confidence) sorted by confidence desc."""
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return []
    x = VAL_TF(img).unsqueeze(0).to(device)
    logits = model(x)
    probs  = F.softmax(logits, dim=1)[0]
    top_probs, top_idxs = torch.topk(probs, k)
    return [(idx2label[str(i.item())], p.item()) for i, p in zip(top_idxs, top_probs)]


# ── SVG fetching & rendering ──────────────────────────────────────────────────

_svg_cache: dict = {}

def fetch_svg_as_pil(label: str, size=96):
    """Download SVG from GitHub and render to a PIL Image."""
    if label in _svg_cache:
        svg_bytes = _svg_cache[label]
    else:
        url = SVG_BASE.format(label=label)
        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code != 200:
                _svg_cache[label] = None
                return None
            svg_bytes = resp.content
            _svg_cache[label] = svg_bytes
        except Exception:
            _svg_cache[label] = None
            return None

    if svg_bytes is None:
        return None

    if not HAS_CAIRO:
        return None

    try:
        png_bytes = cairosvg.svg2png(
            bytestring=svg_bytes,
            output_width=size,
            output_height=size,
        )
        return Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    except Exception:
        return None


def label_to_display(label: str) -> str:
    """Shorten label for display (last two components)."""
    parts = label.split("--")
    return "\n".join(parts[-2:]) if len(parts) >= 2 else label


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_crops(n, seed, label_filter=None):
    """Sample n random rows from crops_index.csv (optionally filtered by label)."""
    rows = []
    with open(CROPS_INDEX) as f:
        for row in csv.DictReader(f):
            if row["category"] == "other-sign":
                continue
            if label_filter and row["category"] != label_filter:
                continue
            rows.append(row)
    rng = random.Random(seed)
    return rng.sample(rows, min(n, len(rows)))


# ── Plotting ──────────────────────────────────────────────────────────────────

CELL_PX   = 128   # pixels per cell in the output figure
CONF_COLORS = ["#2ecc71", "#f39c12", "#e74c3c"]  # green / orange / red for rank 1/2/3


def make_figure(samples, model, idx2label, device, out_path):
    n_rows  = len(samples)
    n_cols  = 1 + TOP_K          # crop + top-3 preds
    fig_w   = n_cols * 1.6
    fig_h   = n_rows * 1.8

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.35})
    if n_rows == 1:
        axes = [axes]

    for row_i, sample_row in enumerate(samples):
        axrow = axes[row_i]
        crop_path = CROPS_DIR / sample_row["crop_path"].replace("crops/", "")
        true_label = sample_row["category"]

        # ── Crop image ────────────────────────────────────────────────────────
        ax = axrow[0]
        try:
            img = Image.open(crop_path).convert("RGB")
            ax.imshow(img)
        except Exception:
            ax.set_facecolor("#cccccc")
        ax.set_xticks([]); ax.set_yticks([])
        short = label_to_display(true_label)
        ax.set_title(f"True:\n{short}", fontsize=6, pad=2)

        # ── Top-K predictions ─────────────────────────────────────────────────
        preds = predict_top_k(model, crop_path, idx2label, device)

        for k_i in range(TOP_K):
            ax = axrow[1 + k_i]
            ax.set_xticks([]); ax.set_yticks([])

            if k_i >= len(preds):
                ax.set_facecolor("#eeeeee")
                continue

            pred_label, conf = preds[k_i]
            svg_img = fetch_svg_as_pil(pred_label, size=CELL_PX)

            if svg_img is not None:
                # Composite onto white background
                bg = Image.new("RGBA", svg_img.size, (255, 255, 255, 255))
                bg.paste(svg_img, mask=svg_img.split()[3])
                ax.imshow(np.array(bg.convert("RGB")))
            else:
                ax.set_facecolor("#f8f8f8")
                short = label_to_display(pred_label)
                ax.text(0.5, 0.5, short, ha="center", va="center",
                        fontsize=5, transform=ax.transAxes, wrap=True)

            correct = (pred_label == true_label)
            border_color = "#2ecc71" if correct else CONF_COLORS[k_i]
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(2)

            short_pred = label_to_display(pred_label)
            ax.set_title(f"#{k_i+1} {conf:.1%}\n{short_pred}",
                         fontsize=5.5, pad=2,
                         color="#1a7a3a" if correct else "#333333")

    # Column headers on first row
    axes[0][0].set_xlabel("Crop", fontsize=7, labelpad=1)
    for k_i in range(TOP_K):
        axes[0][1 + k_i].set_xlabel(f"Pred #{k_i+1}", fontsize=7, labelpad=1)

    plt.suptitle("Sign Classifier — Top-3 Predictions (green border = correct)",
                 fontsize=9, y=1.01)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize sign classifier predictions")
    parser.add_argument("--n",      type=int,   default=12,   help="Number of crops to sample")
    parser.add_argument("--seed",   type=int,   default=42)
    parser.add_argument("--label",  default=None, help="Filter to a specific true label")
    parser.add_argument("--out",    default=str(MODEL_DIR / "prediction_samples.png"))
    args = parser.parse_args()

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Load model + label map
    print("Loading model...")
    model = load_model(device)

    with open(LABEL_MAP) as f:
        lm = json.load(f)
    idx2label = lm["idx2label"]
    print(f"  {len(idx2label)} classes")

    # Sample crops
    print(f"Sampling {args.n} crops (seed={args.seed})...")
    samples = sample_crops(args.n, args.seed, label_filter=args.label)
    print(f"  Got {len(samples)} samples")

    # Render
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print("Running inference + fetching SVGs...")
    make_figure(samples, model, idx2label, device, out_path)


if __name__ == "__main__":
    main()
