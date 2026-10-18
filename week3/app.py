# -*- coding: utf-8 -*-
"""
Week 3 - the web demo.


Usage
-----
    python app.py                 # http://127.0.0.1:5000
    python app.py --port 8000
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

# Import order matters: `predict` adds week1/ to sys.path as a side effect (that is
# where config.py lives), so it has to be imported before `config`. Swapping these
# two lines raises ModuleNotFoundError on startup.
import predict as engine
import config

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

# Load the model once, at import time. Requests then never pay the (multi-second)
# load cost - and if no checkpoint exists the failure is immediate and explicit
# instead of appearing as a 500 on the first upload.
try:
    MODEL, MODEL_PATH = engine.load_model()
    STARTUP_ERROR: str | None = None
except Exception as exc:                                    # noqa: BLE001 - surfaced in the UI
    MODEL, MODEL_PATH, STARTUP_ERROR = None, None, str(exc)


# --------------------------------------------------------------------------- helpers
def read_upload() -> tuple[Image.Image | None, str | None]:
    """Return ``(image, error)``. Everything is validated before inference runs."""
    file = request.files.get("image")
    if file is None or not file.filename:
        return None, "Choose an image first."

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(e.lstrip(".").upper() for e in ALLOWED_EXTENSIONS))
        return None, f"Unsupported file type ({suffix or 'unknown'}). Please use {allowed}."

    try:
        image = Image.open(io.BytesIO(file.read()))
        image.load()                       # force the decode so bad files fail here
    except (UnidentifiedImageError, OSError, ValueError):
        return None, "That file could not be read as an image."

    return image, None


# --------------------------------------------------------------------------- routes
@app.get("/")
def index():
    """Serve the page.

    """
    return render_template("index.html")


@app.post("/api/predict")
def predict_api():
    """Classify one uploaded photo and answer with JSON. The page calls this directly."""
    if MODEL is None:
        return jsonify({"ok": False, "error": STARTUP_ERROR or "no model loaded"}), 503

    image, error = read_upload()
    if error:
        return jsonify({"ok": False, "error": error}), 400

    result = engine.predict_image(image, model=MODEL)
    return jsonify({"ok": True, "checkpoint": MODEL_PATH.name, **result})


@app.get("/healthz")
def healthz():
    return jsonify(
        {
            "status": "ok" if MODEL is not None else "no-model",
            "checkpoint": str(MODEL_PATH) if MODEL_PATH else None,
            "classes": config.CLASS_NAMES,
            "input_size": list(config.IMG_SIZE),
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            # Surfaced so the page can say *why* no model is loaded.
            "error": STARTUP_ERROR,
        }
    )


@app.errorhandler(413)
def too_large(_error):

    limit_mb = MAX_UPLOAD_BYTES // (1024 * 1024)
    return jsonify({"ok": False, "error": f"That file is larger than the {limit_mb} MB limit."}), 413


# --------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description="Run the waste-classification web demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    print("Waste Classification Assistant - web demo")
    if MODEL is not None:
        print(f"  checkpoint : {MODEL_PATH}")
        print(f"  classes    : {', '.join(config.CLASS_NAMES)}")
        print(f"  input size : {config.IMG_SIZE[0]}x{config.IMG_SIZE[1]}")
        print(f"  open       : http://{args.host}:{args.port}")
    else:
        print(f"  no model loaded: {STARTUP_ERROR}")

    # use_reloader=False on purpose: the reloader re-imports this module in a child
    # process, which would load TensorFlow and the model a second time.
    app.run(host=args.host, port=args.port, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
