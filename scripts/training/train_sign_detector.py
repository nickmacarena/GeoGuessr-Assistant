#!/usr/bin/env python3
"""
Train a YOLOv8 binary sign detector on MTSD data.

Detects bounding boxes for all traffic signs in a street-view image.
Binary: class 0 = "sign" (any sign, any type).
Classification of sign type is handled separately by the EfficientNet classifier.

Prerequisites:
    pip install ultralytics
    python scripts/training/prepare_detection_data.py

Output:
    GGAI/models/sign_detector/yolov8n_mtsd/weights/best.pt

Usage:
    python train_sign_detector.py
    python train_sign_detector.py --model yolov8s --epochs 100 --batch 8
    python train_sign_detector.py --resume GGAI/models/sign_detector/yolov8n_mtsd/weights/last.pt
"""

import argparse
from pathlib import Path


REPO_ROOT    = Path(__file__).parent.parent.parent
DATASET_YAML = REPO_ROOT / "data" / "mapillary" / "yolo_detection" / "dataset.yaml"
OUT_DIR      = REPO_ROOT / "GGAI" / "models" / "sign_detector"


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8 sign detector on MTSD")
    parser.add_argument("--model",   default="yolov8n",
                        choices=["yolov8n", "yolov8s", "yolov8m"],
                        help="YOLOv8 variant (n=nano fastest, m=medium most accurate)")
    parser.add_argument("--epochs",  type=int, default=50)
    parser.add_argument("--batch",   type=int, default=16,
                        help="Batch size (reduce to 8 if you hit memory limits)")
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--resume",  default=None,
                        help="Path to last.pt to resume training from")
    parser.add_argument("--name",    default=None,
                        help="Run name under GGAI/models/sign_detector/ (default: {model}_mtsd)")
    args = parser.parse_args()

    # ── Verify data prep has been run ─────────────────────────────────────────
    if not DATASET_YAML.exists():
        print(f"ERROR: {DATASET_YAML} not found.")
        print("Run data prep first:")
        print("  python scripts/training/prepare_detection_data.py")
        return

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics not installed.")
        print("  pip install ultralytics")
        return

    import torch
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "0"   # ultralytics wants "0" not "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    run_name = args.name or f"{args.model}_mtsd"

    # ── Load model ────────────────────────────────────────────────────────────
    if args.resume:
        print(f"Resuming from {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"Starting fresh: {args.model}.pt")
        model = YOLO(f"{args.model}.pt")  # downloads pretrained weights if needed

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\nTraining {args.model} for {args.epochs} epochs")
    print(f"  data:    {DATASET_YAML}")
    print(f"  output:  {OUT_DIR / run_name}")
    print()

    results = model.train(
        data=str(DATASET_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(OUT_DIR),
        name=run_name,
        workers=4,
        # Augmentation: aggressive, signs appear at many scales/angles
        degrees=10,        # slight rotation
        scale=0.5,         # zoom in/out ±50%
        fliplr=0.5,        # horizontal flip (most signs are symmetric)
        mosaic=1.0,        # mosaic augmentation (helps with small signs)
        # Early stopping
        patience=15,       # stop if no improvement for 15 epochs
        # Logging
        verbose=True,
        plots=True,        # save train/val loss curves + mAP plot
        save_period=10,    # save checkpoint every 10 epochs
    )

    best = OUT_DIR / run_name / "weights" / "best.pt"
    print(f"\nDone.")
    print(f"  Best model: {best}")
    print(f"  Best mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
    print(f"\nNext step:")
    print(f"  python scripts/training/detect_and_classify.py --image <path_to_image>")


if __name__ == "__main__":
    main()
