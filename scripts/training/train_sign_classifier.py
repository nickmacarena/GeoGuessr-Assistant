#!/usr/bin/env python3
"""
Train a traffic sign classifier on MTSD crop data.

Fine-tunes EfficientNet-B2 on the 245k labeled sign crops from the MTSD dataset
to output MTSD taxonomy labels (e.g. "regulatory--stop--g1").

Output: trained model weights + label map saved to GGAI/models/sign_classifier/

Usage:
    python train_sign_classifier.py
    python train_sign_classifier.py --epochs 30 --batch-size 128 --backbone efficientnet_b0
    python train_sign_classifier.py --resume GGAI/models/sign_classifier/checkpoint_epoch5.pt

Hardware: runs on MPS (Apple Silicon), CUDA, or CPU.
Expected time on M1 Pro: ~5 min/epoch with default settings.
"""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


# ── Paths (relative to repo root) ────────────────────────────────────────────
REPO_ROOT   = Path(__file__).parent.parent.parent
DATA_ROOT   = REPO_ROOT / "data" / "mapillary"
CROPS_DIR   = DATA_ROOT / "crops"
CROPS_INDEX = CROPS_DIR / "crops_index.csv"
OUT_DIR     = REPO_ROOT / "GGAI" / "models" / "sign_classifier"

# Signs with this label have no MTSD class — excluded from training
SKIP_LABEL  = "other-sign"

# Input resolution for the classifier
IMG_SIZE = 96


# ── Dataset ───────────────────────────────────────────────────────────────────

class SignCropDataset(Dataset):
    """MTSD sign crop dataset loaded from crops_index.csv."""

    def __init__(self, rows, label2idx, transform):
        self.rows      = rows           # list of {crop_path, category} dicts
        self.label2idx = label2idx
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row  = self.rows[i]
        path = CROPS_DIR / row["crop_path"].replace("crops/", "")
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (IMG_SIZE, IMG_SIZE))
        img   = self.transform(img)
        label = self.label2idx[row["category"]]
        return img, label


def load_index(min_samples=5):
    """
    Load crops_index.csv, filter out other-sign and rare classes.

    Returns:
        rows: list of valid rows
        label2idx: {label_str: int}
        idx2label: {int: label_str}
        class_counts: Counter
    """
    rows = []
    with open(CROPS_INDEX) as f:
        for row in csv.DictReader(f):
            if row["category"] != SKIP_LABEL:
                rows.append(row)

    counts = Counter(r["category"] for r in rows)
    # Drop classes with fewer than min_samples (too few to learn from)
    valid_labels = {lab for lab, n in counts.items() if n >= min_samples}
    rows = [r for r in rows if r["category"] in valid_labels]

    labels    = sorted(valid_labels)
    label2idx = {lab: i for i, lab in enumerate(labels)}
    idx2label = {i: lab for lab, i in label2idx.items()}
    return rows, label2idx, idx2label, counts


def train_val_split(rows, val_fraction=0.1, seed=42):
    import random
    rng = random.Random(seed)
    rows = list(rows)
    rng.shuffle(rows)
    n_val = int(len(rows) * val_fraction)
    return rows[n_val:], rows[:n_val]


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model(backbone, num_classes):
    """
    Build EfficientNet or MobileNet backbone with custom head.

    Args:
        backbone: "efficientnet_b0", "efficientnet_b2", "mobilenet_v3_small"
        num_classes: number of output classes
    """
    if backbone == "efficientnet_b2":
        m = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
    elif backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        in_features = m.classifier[1].in_features
        m.classifier[1] = nn.Linear(in_features, num_classes)
    elif backbone == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    return m


# ── Training loop ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True, desc=""):
    model.train(train)
    total_loss = 0.0
    correct    = 0
    total      = 0

    pbar = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += len(labels)

            # Live update: show running loss and accuracy in the progress bar
            pbar.set_postfix({
                "loss": f"{total_loss / total:.4f}",
                "acc":  f"{correct / total:.3f}",
            })

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description="Train MTSD sign classifier")
    parser.add_argument("--backbone",   default="efficientnet_b2",
                        choices=["efficientnet_b0", "efficientnet_b2", "mobilenet_v3_small"])
    parser.add_argument("--epochs",     type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val-frac",   type=float, default=0.1)
    parser.add_argument("--min-samples",type=int, default=5,
                        help="Drop classes with fewer than this many samples")
    parser.add_argument("--workers",    type=int, default=4)
    parser.add_argument("--resume",     default=None,
                        help="Path to checkpoint .pt to resume from")
    parser.add_argument("--out-dir",    default=str(OUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("Loading crop index...")
    rows, label2idx, idx2label, counts = load_index(args.min_samples)
    num_classes = len(label2idx)
    print(f"Classes: {num_classes}  |  Crops: {len(rows):,}")

    train_rows, val_rows = train_val_split(rows, args.val_frac)
    print(f"Train: {len(train_rows):,}  |  Val: {len(val_rows):,}")

    # Save label map alongside the model
    with open(out_dir / "label_map.json", "w") as f:
        json.dump({"label2idx": label2idx, "idx2label": idx2label}, f, indent=2)

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_ds = SignCropDataset(train_rows, label2idx, train_tf)
    val_ds   = SignCropDataset(val_rows,   label2idx, val_tf)

    # Weighted sampler to balance class frequencies
    train_labels = [label2idx[r["category"]] for r in train_rows]
    class_counts = Counter(train_labels)
    weights = [1.0 / class_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Building {args.backbone}...")
    model = build_model(args.backbone, num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 1
    best_val_acc = 0.0

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch  = ckpt["epoch"] + 1
        best_val_acc = ckpt.get("best_val_acc", 0.0)
        print(f"  Resuming from epoch {start_epoch}, best_val_acc={best_val_acc:.3f}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining for {args.epochs} epochs...\n")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True,
            desc=f"Epoch {epoch:3d}/{args.epochs} [train]")
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, None, device, train=False,
            desc=f"Epoch {epoch:3d}/{args.epochs} [val]  ")
        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train loss={train_loss:.4f}  acc={train_acc:.3f}  │  "
              f"val loss={val_loss:.4f}  acc={val_acc:.3f}  "
              f"({elapsed:.0f}s)")

        # Save checkpoint every epoch
        ckpt = {
            "epoch":        epoch,
            "model":        model.state_dict(),
            "optimizer":    optimizer.state_dict(),
            "scheduler":    scheduler.state_dict(),
            "best_val_acc": best_val_acc,
            "num_classes":  num_classes,
            "backbone":     args.backbone,
        }
        torch.save(ckpt, out_dir / f"checkpoint_epoch{epoch:03d}.pt")

        # Keep best model separately
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, out_dir / "best_model.pt")
            print(f"  *** New best val_acc={best_val_acc:.3f} — saved best_model.pt ***")

        # Clean up old checkpoints (keep last 2 + best)
        checkpoints = sorted(out_dir.glob("checkpoint_epoch*.pt"))
        for old in checkpoints[:-2]:
            old.unlink()

    print(f"\nDone. Best val_acc={best_val_acc:.3f}")
    print(f"Model saved to {out_dir}/best_model.pt")
    print(f"Label map saved to {out_dir}/label_map.json")


if __name__ == "__main__":
    main()
