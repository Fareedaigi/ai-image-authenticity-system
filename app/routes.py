from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError
from flask import Blueprint, current_app, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge

from .analysis import analyze_image

bp = Blueprint("main", __name__)

ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "GIF", "BMP", "TIFF"}
# Guard against decompression bombs (~ 16k x 10k pixels)
Image.MAX_IMAGE_PIXELS = 160_000_000


@bp.get("/")
def index():
    return render_template("index.html",
                           max_mb=current_app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024))


@bp.get("/api/health")
def health():
    return {"status": "ok"}


@bp.post("/api/analyze")
def analyze():
    file = request.files.get("image")
    if file is None or not file.filename:
        return jsonify(error="No image uploaded (expected form field 'image')."), 400
    data = file.read()
    if not data:
        return jsonify(error="The uploaded file is empty."), 400
    try:
        fmt = (Image.open(io.BytesIO(data)).format or "").upper()
        if fmt not in ALLOWED_FORMATS:
            return jsonify(error=f"Unsupported image format '{fmt or 'unknown'}'. "
                                 "Use JPEG, PNG, WebP, GIF, BMP or TIFF."), 415
        report = analyze_image(data, file.filename)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        return jsonify(error=f"Could not read the file as an image: {exc}"), 415
    return jsonify(report)


@bp.errorhandler(RequestEntityTooLarge)
def too_large(_):
    mb = current_app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
    return jsonify(error=f"File too large (max {mb} MB)."), 413
