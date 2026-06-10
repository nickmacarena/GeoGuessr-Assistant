#!/usr/bin/env python3
"""
Train a lane marking segmentation model on BDD100K.

Architecture: DeepLabV3 with MobileNetV3-Large backbone (lightweight, fast).
Input: 1280x720 RGB image → Output: 1280x720 mask with 7 classes.

Classes:
    0 = background
    1 = single white
    2 = double white
    3 = single yellow
    4 = double yellow
    5 = road curb
    6 = crosswalk

Usage:
    # Train locally (MPS/CPU)
    python train_lane_segmentation.py

    # Train on Kaggle/Colab (CUDA)
    python train_lane_segmentation.py --device cuda

    # Resume from checkpoint
    python train_lane_segmentation.py --resume checkpoints/lane_seg_epoch_5.pt
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
import cv2
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────────
NUM_CLASSES = 7  # background + 6 lane types
IMG_SIZE = (512, 256)  # (W, H) — downscale from 1280x720 for training speed
BATCH_SIZE = 8
NUM_EPOCHS = 25
LR = 1e-3
NUM_WORKERS = 4

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "bdd100k"
IMG_DIR = DATA_DIR / "images" / "100k"
MASK_DIR = DATA_DIR / "lane_masks"
OUT_DIR = Path(__file__).parent.parent.parent / "GGAI" / "models" / "lane_segmentation"

# Class weights: upweight yellow classes ~4x, downweight background
# Based on pixel frequency analysis (background >> white >> yellow)
CLASS_WEIGHTS = torch.tensor([
    0.1,   # 0: background (dominant)
    1.0,   # 1: single white
    2.0,   # 2: double white (rarer)
    4.0,   # 3: single yellow (key geo signal)
    4.0,   # 4: double yellow (key geo signal)
    0.5,   # 5: road curb (less useful for geo)
    0.5,   # 6: crosswalk (less useful for geo)
], dtype=torch.float32)


class BDDLaneDataset(Dataset):
    """BDD100K lane segmentation dataset."""

    def __init__(self, split="train", img_size=IMG_SIZE):
        self.img_dir = IMG_DIR / split
        self.mask_dir = MASK_DIR / split
        self.img_size = img_size

        # Find all masks and match to images
        mask_files = sorted(self.mask_dir.glob("*.png"))
        self.samples = []
        for mf in mask_files:
            img_file = self.img_dir / f"{mf.stem}.jpg"
            if img_file.exists():
                self.samples.append((img_file, mf))

        print(f"  {split}: {len(self.samples)} samples "
              f"(of {len(mask_files)} masks with matching images)")

        # Image normalization (ImageNet stats for pretrained backbone)
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        # Load and resize image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)

        # Load and resize mask (nearest neighbor to preserve class IDs)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)

        # Apply transforms
        img = self.img_transform(img)
        mask = torch.from_numpy(mask).long()

        return img, mask


def build_model(num_classes=NUM_CLASSES, pretrained=True):
    """Build DeepLabV3 with MobileNetV3-Large backbone."""
    weights = "DEFAULT" if pretrained else None
    model = deeplabv3_mobilenet_v3_large(weights=weights)

    # Replace classifier head for our number of classes
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    if model.aux_classifier is not None:
        model.aux_classifier[-1] = nn.Conv2d(10, num_classes, kernel_size=1)

    return model


def compute_iou(pred, target, num_classes):
    """Compute per-class IoU."""
    ious = []
    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    return ious


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    n_batches = 0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        output = model(imgs)["out"]
        # Output may be different size than mask due to model stride
        if output.shape[2:] != masks.shape[1:]:
            output = nn.functional.interpolate(
                output, size=masks.shape[1:], mode="bilinear", align_corners=False
            )

        loss = criterion(output, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    all_ious = [[] for _ in range(NUM_CLASSES)]

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        output = model(imgs)["out"]
        if output.shape[2:] != masks.shape[1:]:
            output = nn.functional.interpolate(
                output, size=masks.shape[1:], mode="bilinear", align_corners=False
            )

        loss = criterion(output, masks)
        total_loss += loss.item()
        n_batches += 1

        preds = output.argmax(dim=1)
        for b in range(preds.shape[0]):
            ious = compute_iou(preds[b].cpu(), masks[b].cpu(), NUM_CLASSES)
            for c, iou in enumerate(ious):
                if not np.isnan(iou):
                    all_ious[c].append(iou)

    avg_loss = total_loss / n_batches
    class_names = ["bg", "s_white", "d_white", "s_yellow", "d_yellow", "curb", "xwalk"]
    class_ious = {}
    for c in range(NUM_CLASSES):
        if all_ious[c]:
            class_ious[class_names[c]] = np.mean(all_ious[c])
        else:
            class_ious[class_names[c]] = float('nan')

    mean_iou = np.nanmean([v for v in class_ious.values() if not np.isnan(v)])
    return avg_loss, mean_iou, class_ious


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--img-size", type=str, default=f"{IMG_SIZE[0]}x{IMG_SIZE[1]}")
    args = parser.parse_args()

    # Parse image size
    w, h = map(int, args.img_size.split("x"))
    img_size = (w, h)

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Data
    print("Loading datasets...")
    train_ds = BDDLaneDataset("train", img_size=img_size)
    val_ds = BDDLaneDataset("val", img_size=img_size)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # Model
    print("Building model...")
    model = build_model().to(device)

    # Loss with class weights
    weights = CLASS_WEIGHTS.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_iou = 0

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt.get("best_iou", 0)
        print(f"Resumed from epoch {start_epoch}, best IoU: {best_iou:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs, {len(train_ds)} train / {len(val_ds)} val")
    print(f"Image size: {img_size[0]}x{img_size[1]}, Batch: {args.batch_size}, LR: {args.lr}")
    print("=" * 80)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, mean_iou, class_ious = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Format class IoUs
        iou_str = " | ".join(
            f"{k}:{v:.3f}" if not np.isnan(v) else f"{k}:---"
            for k, v in class_ious.items()
            if k != "bg"
        )

        print(f"Epoch {epoch+1:2d}/{args.epochs} | "
              f"train_loss: {train_loss:.4f} | val_loss: {val_loss:.4f} | "
              f"mIoU: {mean_iou:.4f} | lr: {lr:.6f} | {elapsed:.0f}s")
        print(f"  IoU: {iou_str}")

        # Save best model
        if mean_iou > best_iou:
            best_iou = mean_iou
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_iou": best_iou,
                "class_ious": class_ious,
                "img_size": img_size,
                "num_classes": NUM_CLASSES,
            }, OUT_DIR / "best_model.pt")
            print(f"  ★ New best model saved (mIoU: {best_iou:.4f})")

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_iou": best_iou,
            }, OUT_DIR / f"checkpoint_epoch_{epoch+1}.pt")

    print(f"\nDone. Best mIoU: {best_iou:.4f}")
    print(f"Model saved to: {OUT_DIR / 'best_model.pt'}")


if __name__ == "__main__":
    main()
