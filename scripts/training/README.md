# Sign Classifier Training

EfficientNet-B0 classifier fine-tuned on MTSD sign crops to output MTSD taxonomy labels
(e.g. `regulatory--stop--g1`). Trained on 245k labeled crops across 400 sign classes.

## Results

- **Val accuracy: 96.5%** after 20 epochs on 400 classes
- **Top-3 accuracy:** ground truth appears in top-3 predictions in ~99% of cases
- **Hardware:** ~5 min/epoch on Apple M1 Pro (MPS)

## Files

| File | Purpose |
|------|---------|
| `train_sign_classifier.py` | Training script |
| `visualize_predictions.py` | Sample crops → top-3 predictions + reference SVGs |
| `GGAI/models/sign_classifier/best_model.pt` | Trained weights (gitignored, ~52MB) |
| `GGAI/models/sign_classifier/label_map.json` | `label2idx` / `idx2label` for 400 classes |

## Training

```bash
# Default: EfficientNet-B0, 20 epochs, batch 64
python scripts/training/train_sign_classifier.py

# Resume from checkpoint
python scripts/training/train_sign_classifier.py \
  --resume GGAI/models/sign_classifier/checkpoint_epoch010.pt

# Options
python scripts/training/train_sign_classifier.py \
  --backbone efficientnet_b0 \   # or efficientnet_b2, mobilenet_v3_small
  --epochs 30 \
  --batch-size 128 \
  --lr 1e-3
```

Requires `data/mapillary/crops/` and `data/mapillary/crops/crops_index.csv` (see `data/README.md`).

## Inference / Visualization

```bash
# Requires cairosvg + libcairo for SVG rendering
DYLD_LIBRARY_PATH=/opt/homebrew/lib python scripts/training/visualize_predictions.py

# Options
--n 12          # number of crops to sample (default 12)
--seed 42       # random seed
--label <str>   # restrict to a specific MTSD label
--out <path>    # output PNG path
```

Fetches reference SVG icons from `mapillary/mapillary_sprite_source` on GitHub and renders
a grid of: crop image | top-1 prediction | top-2 prediction | top-3 prediction.
Green border = correct prediction.

## Architecture Choices

- **EfficientNet-B0** over B2: crops are 96×96px; B0 is less overparameterized for small inputs
- **WeightedRandomSampler:** balances rare classes (5 samples) against common ones (4,441 samples)
- **`other-sign` excluded:** catch-all label with 102k samples would dominate; not useful for classification
- **Classes with <5 samples dropped:** leaves 400 clean classes

## Label Taxonomy

Labels follow the MTSD format: `{category}--{type}--{variant}`

- `regulatory--stop--g1` — regulatory stop sign, variant 1
- `warning--roundabout--g25` — warning roundabout sign, variant 25 (distinct regional design)
- `warning--texts--g1` — warning sign whose meaning is conveyed by text (no pictogram)

Reference SVGs for all labels:
`https://github.com/mapillary/mapillary_sprite_source/blob/master/package_signs/{label}.svg`
