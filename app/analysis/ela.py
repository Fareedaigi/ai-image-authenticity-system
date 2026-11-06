"""Error Level Analysis (ELA).

The image is re-saved as JPEG at a fixed quality and the difference to the
original is measured. Regions that share one compression history respond
alike; pasted or AI in-painted regions often respond differently and light up.
ELA is primarily a *manipulation* (local edit) detector rather than a
whole-image generation detector, so it carries a low weight in the score.
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image

from .base import AnalyzerResult, ImageContext, to_data_url
from .noise import block_view, noise_maps

QUALITY = 90


def analyze_ela(ctx: ImageContext) -> AnalyzerResult:
    res = AnalyzerResult(
        key="ela",
        name="Error Level Analysis",
        description="Re-compresses the image and highlights regions with a different "
                    "compression history — a classic sign of local edits or in-painting.",
        weight=0.6,
    )
    rgb = ctx.rgb
    if min(rgb.shape[:2]) < 64:
        res.applicable = False
        return res
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=QUALITY)
    recompressed = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.float32)
    diff = np.abs(rgb.astype(np.float32) - recompressed).max(axis=2)

    _, _, energy, mean = noise_maps(ctx.gray)
    ela_b = block_view(diff).mean(axis=(2, 3))
    hh, ww = min(ela_b.shape[0], energy.shape[0]), min(ela_b.shape[1], energy.shape[1])
    ela_b, energy, mean = ela_b[:hh, :ww], energy[:hh, :ww], mean[:hh, :ww]
    ok = (mean > 12) & (mean < 243)

    # ELA naturally grows with texture; model that and look at what is left.
    x = np.log1p(energy[ok])
    y = np.log1p(ela_b[ok])
    if len(x) < 30:
        res.applicable = False
        return res
    a, b = np.polyfit(x, y, 1)
    resid = np.full(ela_b.shape, np.nan, dtype=np.float32)
    resid[ok] = y - (a * x + b)
    rv = resid[ok]
    mad = np.median(np.abs(rv - np.median(rv))) * 1.4826 + 1e-6
    z = (resid - np.median(rv)) / mad
    outliers = np.nan_to_num(z) > 4.5
    # Keep only spatially clustered outliers (>=6 connected blocks)
    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(outliers.astype(np.uint8), 8)
    clusters = [stats[i, cv2.CC_STAT_AREA] for i in range(1, n_lab) if stats[i, cv2.CC_STAT_AREA] >= 6]
    clustered_frac = float(sum(clusters) / max(ok.sum(), 1))

    llr = 0.0
    if 0.02 < clustered_frac < 0.35 and clusters:
        llr = 0.5 + min(len(clusters), 4) * 0.1
        res.add(f"{len(clusters)} region(s) ({clustered_frac:.1%} of the image) respond "
                "differently to re-compression — possible local editing, compositing or "
                "AI in-painting.", "ai")
    else:
        res.add("Error levels are consistent across the image (no localised editing "
                "detected).", "real" if ctx.is_jpeg else "info")
        llr = -0.1 if ctx.is_jpeg else 0.0
    if not ctx.is_jpeg:
        res.add("Source is not a JPEG, so ELA mainly reflects texture here.", "info")
        res.weight = 0.35
    res.llr = llr
    res.details = {
        "recompression_quality": QUALITY,
        "anomalous_regions": len(clusters),
        "anomalous_area_fraction": round(clustered_frac, 4),
        "mean_error_level": round(float(diff.mean()), 3),
    }

    scale = 255.0 / max(np.percentile(diff, 99.5), 1.0)
    vis = np.clip(np.abs(rgb.astype(np.float32) - recompressed) * scale, 0, 255).astype(np.uint8)
    res.visual = to_data_url(vis, fmt="JPEG")
    res.visual_caption = ("Amplified re-compression error. Uniform noise is normal; distinct "
                          "bright or dark patches with sharp borders deserve a closer look.")
    return res
