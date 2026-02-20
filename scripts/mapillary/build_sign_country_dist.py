#!/usr/bin/env python3
"""
Build sign_country_dist.json by sampling Mapillary traffic-sign vector tiles.

For each country in country_bboxes.json:
  1. Enumerate all z=14 tile coordinates within the country bounding box
  2. Randomly sample --tiles-per-country of them
  3. Download each tile from the Mapillary vector tile API (MVT protobuf)
  4. Decode MVT → extract sign 'value' labels (MTSD format, e.g. regulatory--stop--g1)
  5. Aggregate: sign_label → {country_code: count}

Output: GGAI/data/sign_country_dist.json
Partial (resumable): GGAI/data/sign_country_dist_partial.json

Requirements:
    pip install mercantile mapbox-vector-tile requests tqdm python-dotenv

Usage:
    python build_sign_country_dist.py
    python build_sign_country_dist.py --tiles-per-country 200 --countries US FR DE GB
    python build_sign_country_dist.py --resume
    python build_sign_country_dist.py --dry-run   # print tile counts, no requests
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import mercantile
import requests
from dotenv import load_dotenv
from tqdm import tqdm

try:
    import mapbox_vector_tile
except ImportError:
    mapbox_vector_tile = None


# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT          = Path(__file__).parent.parent.parent
COUNTRY_BBOXES     = Path(__file__).parent / "country_bboxes.json"
OUT_PATH           = REPO_ROOT / "GGAI" / "data" / "sign_country_dist.json"
PARTIAL_PATH       = REPO_ROOT / "GGAI" / "data" / "sign_country_dist_partial.json"

# ── Tile API ──────────────────────────────────────────────────────────────────
TILE_URL    = ("https://tiles.mapillary.com/maps/vtp/"
               "mly_map_feature_traffic_sign/2/{z}/{x}/{y}"
               "?access_token={token}")
ZOOM        = 14
LAYER_NAME  = "traffic_sign"
VALUE_FIELD = "value"

# ── Rate limiting ─────────────────────────────────────────────────────────────
SLEEP_BETWEEN_TILES = 0.15   # ~6.7 req/s
MAX_RETRIES         = 3
RETRY_BACKOFF       = 2.0    # seconds; doubles on each retry


# ── Token ─────────────────────────────────────────────────────────────────────

def load_token():
    load_dotenv(REPO_ROOT / ".env")
    token = os.environ.get("MAPILLARY_TOKEN", "")
    if not token:
        raise RuntimeError(
            "MAPILLARY_TOKEN not found. Set it in .env or the environment."
        )
    return token


# ── Tile fetch + decode ────────────────────────────────────────────────────────

def fetch_tile_labels(tile: mercantile.Tile, token: str, session: requests.Session):
    """
    Download one vector tile and return a list of sign label strings.

    Returns [] for empty tiles or on error (never raises).
    """
    url = TILE_URL.format(z=tile.z, x=tile.x, y=tile.y, token=token)
    wait = RETRY_BACKOFF
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(url, timeout=15)

            if resp.status_code == 204 or len(resp.content) == 0:
                return []   # tile exists but has no features

            if resp.status_code == 429:
                time.sleep(wait)
                wait *= 2
                continue

            if resp.status_code != 200:
                return []

            decoded = mapbox_vector_tile.decode(resp.content)
            layer   = decoded.get(LAYER_NAME, {})
            return [
                feat["properties"][VALUE_FIELD]
                for feat in layer.get("features", [])
                if VALUE_FIELD in feat.get("properties", {})
            ]

        except Exception:
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
                wait *= 2

    return []


# ── Helpers ───────────────────────────────────────────────────────────────────

def tiles_for_bbox(west, south, east, north, zoom=ZOOM):
    return list(mercantile.tiles(west, south, east, north, zooms=zoom))


def save_partial(dist, done_countries, path):
    data = {label: dict(counts) for label, counts in dist.items()}
    data["_done_countries"] = sorted(done_countries)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def load_partial(path):
    with open(path) as f:
        saved = json.load(f)
    done = set(saved.pop("_done_countries", []))
    dist = defaultdict(lambda: defaultdict(int))
    for label, counts in saved.items():
        for country, count in counts.items():
            dist[label][country] += count
    return dist, done


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build sign→country distribution from Mapillary vector tiles"
    )
    parser.add_argument("--tiles-per-country", type=int, default=500,
                        help="Tiles to sample per country (default 500)")
    parser.add_argument("--zoom", type=int, default=ZOOM,
                        help=f"Tile zoom level (default {ZOOM})")
    parser.add_argument("--countries", nargs="+", default=None,
                        metavar="CC",
                        help="ISO-3166 codes to process (default: all in country_bboxes.json)")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from partial output file")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(OUT_PATH),
                        help="Output JSON path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print tile counts only; make no HTTP requests")
    args = parser.parse_args()

    if mapbox_vector_tile is None and not args.dry_run:
        print("ERROR: mapbox-vector-tile not installed.")
        print("  pip install mapbox-vector-tile")
        return

    random.seed(args.seed)

    with open(COUNTRY_BBOXES) as f:
        country_bboxes = json.load(f)

    countries = args.countries or list(country_bboxes.keys())
    countries = [c for c in countries if c in country_bboxes]
    if not countries:
        print("No matching countries found.")
        return

    # ── Resume ────────────────────────────────────────────────────────────────
    dist          = defaultdict(lambda: defaultdict(int))
    done_countries = set()
    out_path      = Path(args.out)
    partial_path  = out_path.with_name(out_path.stem + "_partial.json")

    if args.resume and partial_path.exists():
        dist, done_countries = load_partial(partial_path)
        print(f"Resuming: {len(done_countries)} countries already done "
              f"({', '.join(sorted(done_countries))})")

    if not args.dry_run:
        token   = load_token()
        session = requests.Session()
        session.headers["User-Agent"] = "GeoGuessrAI/1.0"

    # ── Per-country sampling ──────────────────────────────────────────────────
    total_requests = 0
    for cc in countries:
        info   = country_bboxes[cc]
        bbox   = info["bbox"]        # [west, south, east, north]
        name   = info["name"]

        all_tiles = tiles_for_bbox(*bbox, zoom=args.zoom)
        n_sample  = min(args.tiles_per_country, len(all_tiles))

        if cc in done_countries:
            print(f"  [{cc}] {name}: skipping (already done)")
            continue

        print(f"\n[{cc}] {name}: {len(all_tiles):,} tiles in bbox → sampling {n_sample}")

        if args.dry_run:
            continue

        sampled = random.sample(all_tiles, n_sample)
        sign_count = 0

        for tile in tqdm(sampled, desc=cc, unit="tile", leave=False):
            labels = fetch_tile_labels(tile, token, session)
            for label in labels:
                dist[label][cc] += 1
            sign_count += len(labels)
            total_requests += 1
            time.sleep(SLEEP_BETWEEN_TILES)

        done_countries.add(cc)
        print(f"  → {sign_count:,} signs found across {n_sample} tiles")

        # Save progress after each country so --resume works
        save_partial(dist, done_countries, partial_path)

    if args.dry_run:
        print("\n(dry-run complete — no requests made)")
        return

    # ── Write final output ────────────────────────────────────────────────────
    final = {
        label: dict(sorted(counts.items(), key=lambda x: -x[1]))
        for label, counts in sorted(dist.items())
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_signs = sum(sum(v.values()) for v in final.values())
    print(f"\n{'='*60}")
    print(f"Countries processed : {len(done_countries)}")
    print(f"Sign classes found  : {len(final):,}")
    print(f"Total sign sightings: {total_signs:,}")
    print(f"HTTP requests made  : {total_requests:,}")
    print(f"\nTop 15 sign classes:")
    top = sorted(final.items(), key=lambda x: -sum(x[1].values()))[:15]
    for label, counts in top:
        total = sum(counts.values())
        top_cc = next(iter(counts))
        print(f"  {total:6,}  {label}  (top country: {top_cc})")

    print(f"\nOutput  → {out_path}")
    print(f"Partial → {partial_path}")

    if partial_path.exists():
        partial_path.unlink()
        print("(partial file removed)")


if __name__ == "__main__":
    main()
