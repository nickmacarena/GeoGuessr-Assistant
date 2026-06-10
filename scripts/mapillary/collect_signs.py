#!/usr/bin/env python3
"""
Collect traffic sign map features from Mapillary across ~60 cities.

For each city, generates a grid of small bboxes (<0.01° each) and queries
the Mapillary map_features endpoint for traffic signs. Stores results in
a single CSV: feature_id, lat, lon, sign_class, country.

Usage:
    # Unit test mode: one tile from one city
    python collect_signs.py --test

    # Collect from a single city
    python collect_signs.py --city berlin

    # Collect from all cities
    python collect_signs.py --all

    # Resume an interrupted run (skips cities already in output)
    python collect_signs.py --all --resume
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import requests
import reverse_geocoder as rg
from dotenv import load_dotenv

from cities import CITIES

load_dotenv(Path(__file__).parent.parent.parent / ".env")
TOKEN = os.environ.get("MAPILLARY_TOKEN")
if not TOKEN:
    raise SystemExit("MAPILLARY_TOKEN not set in .env")

BASE = "https://graph.mapillary.com"
HEADERS = {"Authorization": f"OAuth {TOKEN}"}
OUTPUT_DIR = Path(__file__).parent.parent.parent / "GGAI" / "data"
OUTPUT_CSV = OUTPUT_DIR / "mapillary_signs.csv"

TILE_SIZE = 0.009  # slightly under 0.01° to stay within API limit
FIELDS = "id,object_value,object_type,geometry"
RATE_LIMIT_DELAY = 0.05  # 50ms between requests (conservative)

# Load our classifier's known classes to filter API results
# (Mapillary's taxonomy includes construction, markings, traffic lights etc.
#  that aren't traditional signs and have no geographic signal)
_label_map_path = Path(__file__).parent.parent.parent / "GGAI" / "models" / "sign_classifier" / "label_map.json"
KNOWN_CLASSES = set()
if _label_map_path.exists():
    with open(_label_map_path) as _f:
        KNOWN_CLASSES = set(json.load(_f).get("label2idx", {}).keys())


def _print_init():
    if KNOWN_CLASSES:
        print(f"Loaded {len(KNOWN_CLASSES)} known sign classes from label_map.json")


def generate_tiles(center_lat, center_lon, radius):
    """Generate a grid of bbox tiles around a center point."""
    tiles = []
    lat = center_lat - radius
    while lat < center_lat + radius:
        lon = center_lon - radius
        while lon < center_lon + radius:
            bbox = (
                round(lon, 6),
                round(lat, 6),
                round(lon + TILE_SIZE, 6),
                round(lat + TILE_SIZE, 6),
            )
            tiles.append(bbox)
            lon += TILE_SIZE
        lat += TILE_SIZE
    return tiles


def query_tile(bbox, limit=500):
    """Query Mapillary map_features for traffic signs in a bbox."""
    params = {
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
        "fields": FIELDS,
        "object_type": "traffic_sign",
        "limit": limit,
    }
    try:
        r = requests.get(
            f"{BASE}/map_features",
            headers=HEADERS,
            params=params,
            timeout=15,
        )
        if r.status_code == 429:
            print("  Rate limited, waiting 60s...")
            time.sleep(60)
            return query_tile(bbox, limit)
        r.raise_for_status()
        return r.json().get("data", [])
    except requests.RequestException as e:
        print(f"  Error querying {bbox}: {e}")
        return []


def reverse_geocode_batch(coords):
    """Reverse geocode a list of (lat, lon) tuples to country codes."""
    if not coords:
        return []
    results = rg.search(coords)
    return [r["cc"] for r in results]


def collect_city(city_name, country_hint, center_lat, center_lon, radius):
    """Collect all traffic sign features for one city."""
    tiles = generate_tiles(center_lat, center_lon, radius)
    print(f"\n{'='*60}")
    print(f"  {city_name} ({country_hint}) — {len(tiles)} tiles")
    print(f"{'='*60}")

    features = {}  # id -> feature dict (dedup by feature id)
    t_start = time.time()
    for i, bbox in enumerate(tiles):
        data = query_tile(bbox)
        new = 0
        for feat in data:
            fid = feat["id"]
            sign_class = feat.get("object_value", "")
            if KNOWN_CLASSES and sign_class not in KNOWN_CLASSES:
                continue
            if fid not in features:
                coords = feat.get("geometry", {}).get("coordinates", [None, None])
                features[fid] = {
                    "feature_id": fid,
                    "lon": coords[0],
                    "lat": coords[1],
                    "sign_class": sign_class,
                }
                new += 1
        done = i + 1
        elapsed = time.time() - t_start
        per_tile = elapsed / done
        remaining = per_tile * (len(tiles) - done)
        mins, secs = divmod(int(remaining), 60)
        pct = done / len(tiles) * 100
        if done % 5 == 0 or done == len(tiles):
            print(f"  [{pct:5.1f}%] Tile {done}/{len(tiles)} | {len(features)} signs | ETA {mins}m{secs:02d}s")
        time.sleep(RATE_LIMIT_DELAY)

    # Reverse geocode all at once
    if features:
        coords_list = [(f["lat"], f["lon"]) for f in features.values()]
        countries = reverse_geocode_batch(coords_list)
        for feat, cc in zip(features.values(), countries):
            feat["country"] = cc

    print(f"  Done: {len(features)} unique signs")
    return list(features.values())


def load_completed_cities(csv_path):
    """Load set of cities already in the CSV (for resume mode)."""
    completed = set()
    if csv_path.exists():
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed.add(row.get("city", ""))
    return completed


def write_rows(rows, city_name, csv_path, write_header=False):
    """Append rows to the CSV."""
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["feature_id", "lat", "lon", "sign_class", "country", "city"]
        )
        if write_header:
            writer.writeheader()
        for row in rows:
            row["city"] = city_name
            writer.writerow(row)


def run_test():
    """Test mode: query one tile from the first city."""
    city_name, country, lat, lon, radius = CITIES[0]
    tiles = generate_tiles(lat, lon, radius)
    print(f"Test: querying 1 tile from {city_name}")
    print(f"  Tile bbox: {tiles[0]}")

    data = query_tile(tiles[0])
    print(f"  Features returned: {len(data)}")

    if data:
        # Show first few
        for feat in data[:3]:
            coords = feat.get("geometry", {}).get("coordinates", [None, None])
            print(f"  id={feat['id']}  class={feat.get('object_value', '?')}"
                  f"  lon={coords[0]}  lat={coords[1]}")

        # Test reverse geocoding
        sample_coords = []
        for feat in data[:5]:
            c = feat.get("geometry", {}).get("coordinates", [None, None])
            if c[0] and c[1]:
                sample_coords.append((c[1], c[0]))
        if sample_coords:
            countries = reverse_geocode_batch(sample_coords)
            print(f"  Reverse geocoded countries: {countries}")

        # Test that sign classes match our label_map
        label_map_path = Path(__file__).parent.parent.parent / "GGAI" / "models" / "sign_classifier" / "label_map.json"
        if label_map_path.exists():
            with open(label_map_path) as f:
                label_map = json.load(f)
            known_classes = set(label_map.get("label2idx", {}).keys())
            api_classes = {feat.get("object_value", "") for feat in data}
            matched = api_classes & known_classes
            unmatched = api_classes - known_classes
            print(f"\n  Label map check:")
            print(f"    API classes in this tile: {len(api_classes)}")
            print(f"    Matched to our 400 classes: {len(matched)}")
            print(f"    Not in our label map: {len(unmatched)}")
            if unmatched:
                for cls in sorted(unmatched)[:5]:
                    print(f"      - {cls}")
    else:
        print("  No features found — try a different city or larger tile")

    print("\nTest passed." if data else "\nTest returned no data.")


def main():
    parser = argparse.ArgumentParser(description="Collect Mapillary traffic signs")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true", help="Run unit test on one tile")
    group.add_argument("--city", type=str, help="Collect from a single city")
    group.add_argument("--all", action="store_true", help="Collect from all cities")
    parser.add_argument("--resume", action="store_true", help="Skip already-collected cities")
    args = parser.parse_args()

    _print_init()

    if args.test:
        run_test()
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.city:
        city_match = [c for c in CITIES if c[0] == args.city]
        if not city_match:
            available = [c[0] for c in CITIES]
            raise SystemExit(f"City '{args.city}' not found. Available: {available}")
        cities_to_run = city_match
    else:
        cities_to_run = CITIES

    completed = load_completed_cities(OUTPUT_CSV) if args.resume else set()
    write_header = not OUTPUT_CSV.exists() or not args.resume

    total_signs = 0
    for city_name, country, lat, lon, radius in cities_to_run:
        if city_name in completed:
            print(f"Skipping {city_name} (already collected)")
            continue
        rows = collect_city(city_name, country, lat, lon, radius)
        if rows:
            write_rows(rows, city_name, OUTPUT_CSV, write_header=write_header)
            write_header = False  # only write header once
            total_signs += len(rows)

    print(f"\n{'='*60}")
    print(f"  Total signs collected: {total_signs}")
    print(f"  Output: {OUTPUT_CSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
