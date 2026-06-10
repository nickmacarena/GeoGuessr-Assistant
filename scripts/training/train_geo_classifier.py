#!/usr/bin/env python3
"""
Train a sign → country geo classifier from collected Mapillary sign data.

Input:  GGAI/data/mapillary_signs.csv  (feature_id, lat, lon, sign_class, country, city)

Method:
  1. Group signs into ~100m grid-cell "scenes" — approximates the set of signs
     visible from one street-view position.
  2. Multinomial Naive Bayes over sign-class counts:
         score(country | scene) = log P(country) + Σ_signs log P(sign_class | country)
     with Laplace smoothing. Interpretable: every sign class has an explicit
     per-country likelihood, so individual signs can be shown as "votes".
  3. Evaluate on held-out scenes: top-1/3/5 accuracy, broken down by scene size.

Output: GGAI/models/geo_classifier/sign_country_model.json
  {
    countries:   [...],
    classes:     [...],
    log_prior:   {country: float},
    log_lik:     {sign_class: {country: float}},   # log P(class | country)
    fallback:    {country: float},                 # log-lik for unseen classes
    meta:        {...}
  }

Usage:
    python train_geo_classifier.py
    python train_geo_classifier.py --cell-deg 0.002 --test-frac 0.25 --seed 1
"""

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
CSV_PATH = REPO_ROOT / "GGAI" / "data" / "mapillary_signs.csv"
OUT_DIR = REPO_ROOT / "GGAI" / "models" / "geo_classifier"

ALPHA = 0.5  # Laplace smoothing


def load_scenes(csv_path, cell_deg):
    """Group sign rows into grid-cell scenes.

    Returns list of (country, Counter{sign_class: count}).
    """
    cells = defaultdict(Counter)
    cell_country = {}
    for row in csv.DictReader(open(csv_path)):
        lat, lon = float(row["lat"]), float(row["lon"])
        key = (round(lat / cell_deg), round(lon / cell_deg))
        cells[key][row["sign_class"]] += 1
        cell_country[key] = row["country"]  # cells never straddle borders at 100m

    return [(cell_country[k], counts) for k, counts in cells.items()]


def train_nb(scenes, alpha=ALPHA):
    """Fit multinomial NB: log P(country), log P(sign_class | country)."""
    class_counts = defaultdict(Counter)  # country -> Counter{class: n}
    scene_counts = Counter()             # country -> n scenes

    for country, counts in scenes:
        scene_counts[country] += 1
        for cls, n in counts.items():
            class_counts[country][cls] += n

    countries = sorted(scene_counts)
    classes = sorted({c for cc in class_counts.values() for c in cc})
    n_scenes = sum(scene_counts.values())
    vocab = len(classes)

    log_prior = {c: math.log(scene_counts[c] / n_scenes) for c in countries}

    log_lik = {}
    fallback = {}
    for country in countries:
        total = sum(class_counts[country].values())
        denom = total + alpha * (vocab + 1)  # +1 bucket for unseen classes
        for cls in classes:
            log_lik.setdefault(cls, {})[country] = math.log(
                (class_counts[country][cls] + alpha) / denom)
        fallback[country] = math.log(alpha / denom)

    return {
        "countries": countries,
        "classes": classes,
        "log_prior": log_prior,
        "log_lik": log_lik,
        "fallback": fallback,
    }


def predict(model, counts, use_prior=True):
    """Rank countries for a scene (Counter of sign classes). Returns sorted list."""
    scores = {}
    for country in model["countries"]:
        s = model["log_prior"][country] if use_prior else 0.0
        for cls, n in counts.items():
            ll = model["log_lik"].get(cls, {}).get(country)
            if ll is None:
                ll = model["fallback"][country]
            s += n * ll
        scores[country] = s
    return sorted(scores, key=scores.get, reverse=True)


def evaluate(model, scenes, use_prior=True):
    """Top-k accuracy overall and by scene size bucket."""
    buckets = {"1": [], "2-4": [], "5-9": [], "10+": []}
    overall = []

    for country, counts in scenes:
        n_signs = sum(counts.values())
        ranking = predict(model, counts, use_prior=use_prior)
        rank = ranking.index(country) if country in ranking else len(ranking)
        overall.append(rank)
        key = ("1" if n_signs == 1 else
               "2-4" if n_signs <= 4 else
               "5-9" if n_signs <= 9 else "10+")
        buckets[key].append(rank)

    def topk(ranks, k):
        return sum(r < k for r in ranks) / len(ranks) if ranks else float("nan")

    print(f'  {"bucket":8s} {"scenes":>7s} {"top-1":>7s} {"top-3":>7s} {"top-5":>7s}')
    for key in ["1", "2-4", "5-9", "10+"]:
        ranks = buckets[key]
        if ranks:
            print(f'  {key:8s} {len(ranks):>7,} {topk(ranks,1):>7.1%} '
                  f'{topk(ranks,3):>7.1%} {topk(ranks,5):>7.1%}')
    print(f'  {"ALL":8s} {len(overall):>7,} {topk(overall,1):>7.1%} '
          f'{topk(overall,3):>7.1%} {topk(overall,5):>7.1%}')
    return topk(overall, 1)


def main():
    parser = argparse.ArgumentParser(description="Train sign→country NB classifier")
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--cell-deg", type=float, default=0.001,
                        help="Grid cell size in degrees (~111m per 0.001 lat)")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()

    print(f"Loading {args.csv}")
    scenes = load_scenes(args.csv, args.cell_deg)
    sizes = Counter(sum(c.values()) for _, c in scenes)
    n_signs = sum(k * v for k, v in sizes.items())
    print(f"  {len(scenes):,} scenes from {n_signs:,} signs "
          f"(cell ≈ {args.cell_deg * 111000:.0f}m)")
    print(f"  scene size: 1 sign × {sizes[1]:,}, "
          f"2-4 × {sum(v for k, v in sizes.items() if 2 <= k <= 4):,}, "
          f"5+ × {sum(v for k, v in sizes.items() if k >= 5):,}")

    random.seed(args.seed)
    random.shuffle(scenes)
    n_test = int(len(scenes) * args.test_frac)
    test, train = scenes[:n_test], scenes[n_test:]
    print(f"  train: {len(train):,} scenes / test: {len(test):,} scenes")

    print("\nTraining NB model...")
    model = train_nb(train, alpha=args.alpha)
    print(f"  {len(model['countries'])} countries, {len(model['classes'])} sign classes")

    print("\nTest accuracy (with country prior):")
    evaluate(model, test, use_prior=True)
    print("\nTest accuracy (no prior — fair for GeoGuessr where prior shouldn't"
          " reflect collection volume):")
    evaluate(model, test, use_prior=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "sign_country_model.json"
    model["meta"] = {
        "source": str(Path(args.csv).name),
        "cell_deg": args.cell_deg,
        "alpha": args.alpha,
        "n_train_scenes": len(train),
        "n_test_scenes": len(test),
    }
    with open(out_path, "w") as f:
        json.dump(model, f)
    print(f"\nSaved model → {out_path}")


if __name__ == "__main__":
    main()
