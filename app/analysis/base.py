"""Shared types and helpers for the analyzers.

Every analyzer returns an :class:`AnalyzerResult`. Evidence is expressed as a
*log-likelihood ratio* (LLR): positive values push towards "AI-generated",
negative values towards "authentic camera capture", 0 means no evidence.
The scoring module combines the LLRs of all analyzers, which keeps the fusion
step simple and lets strong evidence (e.g. a signed C2PA manifest naming an
AI generator) dominate weak statistical hints.
"""
from __future__ import annotations

import base64
import io
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageOps

# Pixel analyses run on at most this many pixels on the long side. We *crop*
# rather than resize where possible, because resampling destroys the very
# high-frequency traces we are looking for.
MAX_ANALYSIS_SIDE = 2048


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def ramp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Piecewise-linear map of x from [x0, x1] onto [y0, y1], clamped."""
    if x1 == x0:
        return y1
    t = clamp((x - x0) / (x1 - x0), 0.0, 1.0)
    return y0 + t * (y1 - y0)


@dataclass
class Finding:
    text: str
    # "ai" (suspicious), "real" (supports authenticity) or "info"
    kind: str = "info"

    def to_dict(self) -> dict:
        return {"text": self.text, "kind": self.kind}


@dataclass
class AnalyzerResult:
    key: str
    name: str
    description: str
    applicable: bool = True
    llr: float = 0.0
    # Relative trust in this analyzer for the current image (0..1)
    weight: float = 1.0
    findings: list[Finding] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    visual: str | None = None  # data: URL of a diagnostic image
    visual_caption: str | None = None

    def add(self, text: str, kind: str = "info") -> None:
        self.findings.append(Finding(text, kind))

    @property
    def ai_probability(self) -> float | None:
        if not self.applicable:
            return None
        return sigmoid(self.llr)

    def to_dict(self) -> dict:
        p = self.ai_probability
        return {
            "key": self.key,
            "name": self.name,
            "description": self.description,
            "applicable": self.applicable,
            "llr": round(self.llr, 3),
            "weight": round(self.weight, 3),
            "ai_probability": None if p is None else round(p, 4),
            "findings": [f.to_dict() for f in self.findings],
            "details": self.details,
            "visual": self.visual,
            "visual_caption": self.visual_caption,
        }


class ImageContext:
    """Decoded image plus the raw bytes, shared by all analyzers."""

    def __init__(self, data: bytes, filename: str = "upload"):
        self.data = data
        self.filename = filename
        img = Image.open(io.BytesIO(data))
        self.format = (img.format or "UNKNOWN").upper()
        self.pil_original = img  # keeps .info / exif / quantization
        img.load()
        self.width, self.height = img.size
        self.mode = img.mode

        # Respect EXIF orientation for display / face detection.
        upright = ImageOps.exif_transpose(img.copy())
        if upright.mode not in ("RGB", "L"):
            if "A" in upright.getbands() or upright.mode == "P":
                upright = upright.convert("RGBA")
                bg = Image.new("RGB", upright.size, (255, 255, 255))
                bg.paste(upright, mask=upright.split()[-1])
                upright = bg
            else:
                upright = upright.convert("RGB")
        if upright.mode == "L":
            upright = upright.convert("RGB")

        self.rgb_full = np.asarray(upright, dtype=np.uint8)
        self.rgb = center_crop(self.rgb_full, MAX_ANALYSIS_SIDE)
        self.gray = rgb_to_gray(self.rgb)

    @property
    def is_jpeg(self) -> bool:
        return self.format in ("JPEG", "MPO")

    def jpeg_quality_estimate(self) -> int | None:
        q = getattr(self.pil_original, "quantization", None)
        if not q or not self.is_jpeg:
            return None
        return estimate_jpeg_quality(q)


def center_crop(arr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = arr.shape[:2]
    ch, cw = min(h, max_side), min(w, max_side)
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return arr[y0:y0 + ch, x0:x0 + cw]


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def to_data_url(arr: np.ndarray | Image.Image, max_side: int = 900,
                fmt: str = "PNG") -> str:
    img = arr if isinstance(arr, Image.Image) else Image.fromarray(arr)
    if max(img.size) > max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.LANCZOS)
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.convert("RGB").save(buf, "JPEG", quality=85)
        mime = "image/jpeg"
    else:
        img.save(buf, "PNG", optimize=True)
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def heatmap(values: np.ndarray, vmin: float | None = None,
            vmax: float | None = None) -> np.ndarray:
    """Map a 2-D float array to an RGB 'inferno-like' colour image."""
    v = values.astype(np.float32)
    lo = np.nanmin(v) if vmin is None else vmin
    hi = np.nanmax(v) if vmax is None else vmax
    t = np.clip((v - lo) / (hi - lo + 1e-9), 0, 1)
    t = np.nan_to_num(t)
    stops = np.array([
        [0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99],
        [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164],
    ], dtype=np.float32)
    idx = t * (len(stops) - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, len(stops) - 1)
    f = (idx - i0)[..., None]
    out = stops[i0] * (1 - f) + stops[i1] * f
    return out.astype(np.uint8)


# Standard IJG (libjpeg) luminance table, used for quality estimation.
_IJG_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99,
], dtype=np.float64)


def ijg_table(quality: int) -> np.ndarray:
    quality = int(clamp(quality, 1, 100))
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    t = np.floor((_IJG_LUMA * scale + 50) / 100)
    return np.clip(t, 1, 255)


def estimate_jpeg_quality(qtables: dict) -> int | None:
    luma = qtables.get(0)
    if luma is None:
        return None
    # Compare sorted values so the estimate doesn't depend on coefficient
    # ordering (natural vs zig-zag).
    luma = np.sort(np.array(list(luma), dtype=np.float64))
    best_q, best_err = None, float("inf")
    for q in range(1, 101):
        err = np.abs(np.sort(ijg_table(q)) - luma).mean()
        if err < best_err:
            best_q, best_err = q, err
    return best_q


def jpeg_tables_are_standard(qtables: dict) -> bool:
    luma = qtables.get(0)
    if luma is None:
        return False
    s = np.sort(np.array(list(luma), dtype=np.float64))
    return any(np.array_equal(np.sort(ijg_table(q)), s) for q in range(1, 101))
