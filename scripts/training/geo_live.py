#!/usr/bin/env python3
"""
Live GeoGuessr companion: watch for screenshots, annotate, serve in-browser.

Loads all models once, then watches a directory for new screenshots. Each new
image is run through both pipelines (signs + lanes) and served at
http://localhost:<port>/ — the page auto-reloads only when a new result is
ready, so hover tooltips are never interrupted.

Play flow:
  1. python geo_live.py            (leave running; opens browser tab)
  2. Play GeoGuessr fullscreen
  3. Hit the macOS screenshot hotkey (Cmd-Shift-3) when you see signs/lanes
  4. Glance at the browser tab — hover boxes for country likelihoods

Usage:
    DYLD_LIBRARY_PATH=/opt/homebrew/lib python geo_live.py
    python geo_live.py --watch-dir ~/Desktop --port 8077 --conf 0.3
"""

import argparse
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
from PIL import Image

from geo_inspector import (
    DETECTOR_PATH, CLASSIFIER_PATH, LABEL_MAP_PATH, REGION_MAP_PATH,
    LANE_MODEL_PATH, GEO_MODEL_PATH, TEXT_PIPELINE_AVAILABLE, DEFAULT_TEXT_CONF,
    load_classifier, load_lane_model, detect_signs, detect_lanes, detect_texts,
    render_html,
)

if TEXT_PIPELINE_AVAILABLE:
    from geo_inspector import LanguageDetector

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

# Shared state between watcher thread and HTTP server
STATE = {
    "version": 0,
    "html": "<p style='font-family:sans-serif'>Waiting for first screenshot…</p>",
    "status": "waiting",
}
STATE_LOCK = threading.Lock()


VIEWER_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>GeoGuessr Live Inspector</title>
<style>
  body { margin: 0; background: #1a1a2e; }
  #frame { width: 100vw; height: 100vh; border: 0; }
  #status { position: fixed; top: 8px; right: 12px; color: #8fd;
            font: 12px sans-serif; background: rgba(0,0,0,0.6);
            padding: 4px 10px; border-radius: 10px; z-index: 999; }
</style>
</head>
<body>
<div id="status">connecting…</div>
<iframe id="frame" src="/live"></iframe>
<script>
let current = -1;
async function poll() {
  try {
    const r = await fetch('/version');
    const j = await r.json();
    document.getElementById('status').textContent = j.status;
    if (j.version !== current) {
      current = j.version;
      document.getElementById('frame').src = '/live?v=' + j.version;
    }
  } catch (e) {
    document.getElementById('status').textContent = 'daemon stopped';
  }
  setTimeout(poll, 800);
}
poll();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            body, ctype = VIEWER_PAGE.encode(), "text/html"
        elif self.path.startswith("/live"):
            with STATE_LOCK:
                body, ctype = STATE["html"].encode(), "text/html"
        elif self.path.startswith("/version"):
            with STATE_LOCK:
                payload = {"version": STATE["version"], "status": STATE["status"]}
            body, ctype = json.dumps(payload).encode(), "application/json"
        else:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # silence per-request logging


def set_status(text):
    with STATE_LOCK:
        STATE["status"] = text


def publish(html_doc):
    with STATE_LOCK:
        STATE["html"] = html_doc
        STATE["version"] += 1
        STATE["status"] = f"result #{STATE['version']} — hover boxes"


def wait_for_stable(path, checks=3, delay=0.25):
    """Wait until file size stops changing (screenshot may still be writing)."""
    last = -1
    stable = 0
    while stable < checks:
        try:
            size = path.stat().st_size
        except FileNotFoundError:
            return False
        if size == last and size > 0:
            stable += 1
        else:
            stable = 0
        last = size
        time.sleep(delay)
    return True


def main():
    parser = argparse.ArgumentParser(description="Live screenshot geo inspector")
    parser.add_argument("--watch-dir", default=str(Path.home() / "Desktop"))
    parser.add_argument("--port",      type=int, default=8077)
    parser.add_argument("--conf",      type=float, default=0.3)
    parser.add_argument("--iou",       type=float, default=0.45)
    parser.add_argument("--top-k",     type=int, default=3)
    parser.add_argument("--text-conf", type=float, default=DEFAULT_TEXT_CONF,
                        help="OCR confidence threshold for text boxes")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    watch_dir = Path(args.watch_dir).expanduser()
    if not watch_dir.is_dir():
        raise SystemExit(f"Watch dir not found: {watch_dir}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    from ultralytics import YOLO
    print("Loading models (once)...")
    detector = YOLO(str(DETECTOR_PATH))
    classifier = load_classifier(CLASSIFIER_PATH, device)
    lane_model, lane_ckpt = load_lane_model(LANE_MODEL_PATH, device)
    with open(LABEL_MAP_PATH) as f:
        idx2label = json.load(f)["idx2label"]
    with open(REGION_MAP_PATH) as f:
        region_map = json.load(f)
    country_model = None
    if GEO_MODEL_PATH.exists():
        with open(GEO_MODEL_PATH) as f:
            country_model = json.load(f)
    lang_detector = LanguageDetector() if TEXT_PIPELINE_AVAILABLE else None
    if lang_detector is None:
        print("Text pipeline unavailable (needs macOS Vision + fasttext) — skipping.")
    print("Models ready.")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Serving {url}")
    print(f"Watching {watch_dir} for new screenshots — play GeoGuessr and hit "
          f"Cmd-Shift-3")
    if not args.no_browser:
        webbrowser.open(url)

    start_time = time.time()
    seen = set()
    set_status("waiting for screenshot…")

    while True:
        try:
            for p in watch_dir.iterdir():
                if (p.suffix.lower() not in IMAGE_EXTS or p.name.startswith(".")
                        or p in seen or p.stat().st_mtime < start_time):
                    continue
                seen.add(p)
                print(f"\nNew screenshot: {p.name}")
                set_status(f"processing {p.name}…")
                if not wait_for_stable(p):
                    continue
                t0 = time.time()
                try:
                    image = Image.open(p).convert("RGB")
                    sign_dets = detect_signs(
                        p, image, detector, classifier, idx2label, region_map,
                        device, args.conf, args.iou, args.top_k,
                        country_model=country_model)
                    lane_dets = detect_lanes(image, lane_model, lane_ckpt, device)
                    text_dets = (detect_texts(p, lang_detector,
                                              min_conf=args.text_conf)
                                 if lang_detector else [])
                    html_doc = render_html(image, sign_dets, lane_dets, p.name,
                                           text_dets=text_dets)
                    publish(html_doc)
                    print(f"  {len(sign_dets)} signs, {len(lane_dets)} lane lines, "
                          f"{len(text_dets)} text regions "
                          f"({time.time() - t0:.1f}s) → browser updated")
                except Exception as e:
                    print(f"  ERROR processing {p.name}: {e}")
                    set_status(f"error: {e}")
            time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nStopping.")
            server.shutdown()
            break


if __name__ == "__main__":
    main()
