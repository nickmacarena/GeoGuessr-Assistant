#!/usr/bin/env python3
"""
Minimal Mapillary Graph API test.

Tests:
  1. Search API  — find images in a small bounding box
  2. Entity API  — fetch full metadata for one image (incl. thumbnail URL)

Usage:
    python test_api.py
"""

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent.parent / ".env")
TOKEN = os.environ.get("MAPILLARY_TOKEN")
if not TOKEN:
    raise SystemExit("MAPILLARY_TOKEN not set")

BASE = "https://graph.mapillary.com"
HEADERS = {"Authorization": f"OAuth {TOKEN}"}

# ── 1. Search API: small bbox around central Paris ────────────────────────────
print("=== Search API ===")
bbox = "2.3460,48.8530,2.3530,48.8570"   # minx,miny,maxx,maxy
params = {
    "bbox":   bbox,
    "fields": "id,geometry,computed_geometry,captured_at,sequence",
    "limit":  5,
}
r = requests.get(f"{BASE}/images", headers=HEADERS, params=params, timeout=15)
print(f"Status: {r.status_code}")
data = r.json()
images = data.get("data", [])
print(f"Images returned: {len(images)}")
for img in images:
    geom = img.get("computed_geometry") or img.get("geometry") or {}
    coords = geom.get("coordinates", [None, None])
    print(f"  id={img['id']}  lon={coords[0]}  lat={coords[1]}  captured={img.get('captured_at','?')}")

# ── 2. Entity API: full metadata for first result ─────────────────────────────
if images:
    print("\n=== Entity API ===")
    img_id = images[0]["id"]
    fields = "id,geometry,computed_geometry,captured_at,thumb_256_url,thumb_1024_url,thumb_2048_url,sequence,creator"
    r2 = requests.get(f"{BASE}/{img_id}", headers=HEADERS, params={"fields": fields}, timeout=15)
    print(f"Status: {r2.status_code}")
    print(json.dumps(r2.json(), indent=2))
