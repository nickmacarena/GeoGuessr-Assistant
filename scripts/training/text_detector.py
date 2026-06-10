#!/usr/bin/env python3
"""
OCR text detection via Apple Vision framework (macOS native).

Returns recognized text strings with pixel bounding boxes, suitable for the
geo inspector's third detection type: text → language → country likelihood.

Requires: pyobjc-framework-Vision (macOS 13+ for automatic language detection).

Usage:
    from text_detector import detect_text
    results = detect_text(pil_image)   # [{text, conf, box: (x1,y1,x2,y2)}]
"""

import os
import tempfile

import Vision
from Foundation import NSURL


def detect_text(pil_image, min_conf=0.3):
    """Run Vision OCR on a PIL image. Returns list of dicts with pixel boxes.

    Hands Vision a temp file URL rather than in-memory NSData — the
    dataWithBytes_length_ bridge SIGBUSes when torch is loaded in-process.
    """
    img_w, img_h = pil_image.size

    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        pil_image.save(tmp_path, format="PNG")
        return _detect_from_file(tmp_path, img_w, img_h, min_conf)
    finally:
        os.unlink(tmp_path)


def _detect_from_file(path, img_w, img_h, min_conf):
    url = NSURL.fileURLWithPath_(path)
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    request.setUsesLanguageCorrection_(True)
    if request.respondsToSelector_("setAutomaticallyDetectsLanguage:"):
        request.setAutomaticallyDetectsLanguage_(True)

    ok, error = handler.performRequests_error_([request], None)
    if not ok:
        raise RuntimeError(f"Vision OCR failed: {error}")

    results = []
    for obs in request.results() or []:
        candidates = obs.topCandidates_(1)
        if not candidates:
            continue
        text = str(candidates[0].string()).strip()
        conf = float(obs.confidence())
        if not text or conf < min_conf:
            continue

        # Vision bounding boxes are normalized with origin at bottom-left
        bb = obs.boundingBox()
        x, y = bb.origin.x, bb.origin.y
        w, h = bb.size.width, bb.size.height
        x1 = int(x * img_w)
        y1 = int((1.0 - y - h) * img_h)
        x2 = int((x + w) * img_w)
        y2 = int((1.0 - y) * img_h)

        results.append({"text": text, "conf": conf, "box": (x1, y1, x2, y2)})

    return results


if __name__ == "__main__":
    import argparse
    import json
    import sys

    from PIL import Image

    parser = argparse.ArgumentParser(description="Vision OCR")
    parser.add_argument("image")
    parser.add_argument("--json", action="store_true",
                        help="Emit JSON (for subprocess use by geo_inspector — "
                             "Vision SIGBUSes in-process with torch/MPS)")
    parser.add_argument("--min-conf", type=float, default=0.3)
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    results = detect_text(img, min_conf=args.min_conf)
    if args.json:
        json.dump(results, sys.stdout)
    else:
        for r in results:
            print(f"  {r['conf']:.2f}  {r['box']}  {r['text']!r}")
