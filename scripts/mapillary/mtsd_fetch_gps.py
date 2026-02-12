#!/usr/bin/env python3
"""
Fetch GPS coordinates for all MTSD images from the Mapillary API,
then build a sign_class -> country distribution lookup table.

Output files:
  mtsd_gps.csv              - image_key, lat, lon, country
  sign_country_dist.json    - {sign_class: {country: count, ...}, ...}

Usage:
    python mtsd_fetch_gps.py --token YOUR_TOKEN
    python mtsd_fetch_gps.py --token YOUR_TOKEN --ann-dirs ../../data/mapillary/full_ds/mtsd_v2_fully_annotated/annotations
"""

import argparse
import json
import glob
import time
import csv
import os
from pathlib import Path
from collections import defaultdict

import requests
from tqdm import tqdm


MAPILLARY_API = "https://graph.mapillary.com"
BATCH_SIZE = 50          # images per API request (Mapillary supports comma-joined IDs)
RATE_LIMIT_PAUSE = 0.2   # seconds between batches


def load_all_annotations(ann_dirs):
    """
    Load all per-image JSONs from one or more annotation directories.

    Returns:
        dict: {image_key: [label, ...]}
    """
    image_labels = defaultdict(list)
    for ann_dir in ann_dirs:
        files = glob.glob(os.path.join(ann_dir, "**", "*.json"), recursive=True)
        for fp in files:
            image_key = Path(fp).stem
            try:
                with open(fp) as f:
                    data = json.load(f)
                for obj in data.get("objects", []):
                    label = obj.get("label")
                    if label and label != "other-sign":
                        image_labels[image_key].append(label)
            except Exception:
                pass
    return dict(image_labels)


def fetch_gps_batch(image_keys, token):
    """
    Fetch GPS coordinates for a batch of image keys.

    Returns:
        dict: {image_key: (lat, lon)} for keys that returned data
    """
    ids = ",".join(image_keys)
    url = f"{MAPILLARY_API}/images?ids={ids}&fields=id,computed_geometry"
    headers = {"Authorization": f"OAuth {token}"}
    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  API error: {e}")
        return {}

    results = {}
    for item in data.get("data", []):
        key = item.get("id")
        geom = item.get("computed_geometry") or item.get("geometry")
        if key and geom and geom.get("type") == "Point":
            lon, lat = geom["coordinates"]
            results[key] = (float(lat), float(lon))
    return results


def latlon_to_country(lat, lon):
    """
    Convert lat/lon to ISO-3166 country code using reverse geocoding.
    Uses the free nominatim API (no key required, 1 req/sec limit).

    Returns:
        str: 2-letter country code, or '' if lookup fails
    """
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "json", "zoom": 3}
    headers = {"User-Agent": "GeoGuessrAssistant/1.0"}
    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("address", {}).get("country_code", "").upper()
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(
        description="Fetch GPS coords for MTSD images and build sign→country distribution"
    )
    parser.add_argument("--token", default=None, help="Mapillary API access token (or set MAPILLARY_TOKEN env var)")
    parser.add_argument(
        "--ann-dirs",
        nargs="+",
        default=[
            "../../data/mapillary/full_ds/mtsd_v2_fully_annotated/annotations",
            "../../data/mapillary/full_ds/mtsd_v2_partially_annotated/annotations",
        ],
    )
    parser.add_argument("--out-gps",  default="../../data/mapillary/mtsd_gps.csv")
    parser.add_argument("--out-dist", default="../../data/mapillary/sign_country_dist.json")
    parser.add_argument("--resume",   action="store_true",
                        help="Skip images already in --out-gps")
    args = parser.parse_args()

    # Resolve token from env if not provided
    if args.token is None:
        args.token = os.environ.get("MAPILLARY_TOKEN")
    if not args.token:
        parser.error("Provide --token or set MAPILLARY_TOKEN in your environment / .env file")

    # Resolve paths relative to this script
    script_dir = Path(__file__).parent
    ann_dirs  = [str(script_dir / d) for d in args.ann_dirs]
    out_gps   = script_dir / args.out_gps
    out_dist  = script_dir / args.out_dist

    # ── 1. Load annotations ───────────────────────────────────────────────────
    print("Loading annotations...")
    image_labels = load_all_annotations(ann_dirs)
    image_keys = list(image_labels.keys())
    print(f"  {len(image_keys):,} images with specific sign labels")

    # ── 2. Resume: skip already-fetched keys ──────────────────────────────────
    already_fetched = {}
    if args.resume and out_gps.exists():
        with open(out_gps, newline="") as f:
            for row in csv.DictReader(f):
                already_fetched[row["image_key"]] = (
                    float(row["lat"]), float(row["lon"]), row["country"]
                )
        print(f"  Resuming: {len(already_fetched):,} already fetched")
        image_keys = [k for k in image_keys if k not in already_fetched]
        print(f"  {len(image_keys):,} remaining")

    # ── 3. Fetch GPS from Mapillary API in batches ────────────────────────────
    print(f"\nFetching GPS coordinates ({len(image_keys):,} images, batch={BATCH_SIZE})...")
    gps_results = {}  # image_key -> (lat, lon)

    batches = [image_keys[i:i+BATCH_SIZE] for i in range(0, len(image_keys), BATCH_SIZE)]
    for batch in tqdm(batches, desc="Mapillary API"):
        results = fetch_gps_batch(batch, args.token)
        gps_results.update(results)
        time.sleep(RATE_LIMIT_PAUSE)

    print(f"  Got GPS for {len(gps_results):,} / {len(image_keys):,} images")

    # ── 4. Reverse-geocode lat/lon → country ──────────────────────────────────
    print(f"\nReverse geocoding {len(gps_results):,} coordinates (1 req/sec)...")
    gps_with_country = dict(already_fetched)  # carry forward resumed results

    for key, (lat, lon) in tqdm(gps_results.items(), desc="Nominatim"):
        country = latlon_to_country(lat, lon)
        gps_with_country[key] = (lat, lon, country)
        time.sleep(1.0)  # Nominatim rate limit: 1 req/sec

    # ── 5. Save GPS CSV ───────────────────────────────────────────────────────
    out_gps.parent.mkdir(parents=True, exist_ok=True)
    with open(out_gps, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_key", "lat", "lon", "country"])
        writer.writeheader()
        for key, (lat, lon, country) in gps_with_country.items():
            writer.writerow({"image_key": key, "lat": lat, "lon": lon, "country": country})
    print(f"\nGPS data saved to {out_gps}")

    # ── 6. Build sign_class → country distribution ────────────────────────────
    print("\nBuilding sign → country distribution...")
    dist = defaultdict(lambda: defaultdict(int))

    for image_key, (lat, lon, country) in gps_with_country.items():
        if not country:
            continue
        for label in image_labels.get(image_key, []):
            dist[label][country] += 1

    # Convert to sorted dicts (most common country first)
    dist_sorted = {
        label: dict(sorted(counts.items(), key=lambda x: -x[1]))
        for label, counts in sorted(dist.items())
    }

    with open(out_dist, "w") as f:
        json.dump(dist_sorted, f, indent=2)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    total_labels = sum(len(v) for v in dist_sorted.values())
    print(f"\n{'='*60}")
    print(f"Sign classes mapped: {len(dist_sorted):,}")
    print(f"Total (sign, country) pairs: {total_labels:,}")
    print(f"\nTop 10 most-seen sign classes:")
    top = sorted(dist_sorted.items(), key=lambda x: -sum(x[1].values()))[:10]
    for label, counts in top:
        total = sum(counts.values())
        top_country = next(iter(counts))
        print(f"  {total:5d}  {label}  (top: {top_country})")
    print(f"\nDistribution saved to {out_dist}")


if __name__ == "__main__":
    main()
