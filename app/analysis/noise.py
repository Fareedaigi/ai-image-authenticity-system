"""Noise residual & texture analysis.

Every camera sensor adds photon (shot) and read noise, so even perfectly flat
areas of a real photo (sky, walls, skin) show a fine, fairly uniform grain.
Diffusion models synthesise "noise-like" texture where they think it belongs,
but flat regions often come out unnaturally clean, and textured surfaces get a
smooth, waxy look. Large differences of noise level between regions can also
indicate splicing or AI in-painting.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import AnalyzerResult, ImageContext, heatmap, ramp, to_data_url

BLOCK = 16


def block_view(a: np.ndarray, b: int = BLOCK) -> np.ndarray:
    h, w = a.shape[0] // b * b, a.shape[1] // b * b
    a = a[:h, :w]
    return a.reshape(h // b, b, w // b, b).swapaxes(1, 2)


def noise_maps(gray: np.ndarray):
    """Return per-block (noise sigma, gradient energy, mean intensity)."""
    g8 = np.clip(gray, 0, 255).astype(np.uint8)
    g = gray.astype(np.float32)
    resid = g - cv2.medianBlur(g8, 3).astype(np.float32)
    # Second residual from a smooth denoiser: mean absolute residual is a
    # stable noise estimator even for 8-bit quantised data.
    resid2 = g - cv2.GaussianBlur(g, (0, 0), 1.0)
    gx = cv2.Sobel(cv2.GaussianBlur(g, (0, 0), 1.5), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(cv2.GaussianBlur(g, (0, 0), 1.5), cv2.CV_32F, 0, 1)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    sigma = np.abs(block_view(resid)).mean(axis=(2, 3)) * 1.25
    sigma2 = np.abs(block_view(resid2)).mean(axis=(2, 3)) * 1.25
    energy = block_view(grad).mean(axis=(2, 3))
    mean = block_view(g).mean(axis=(2, 3))
    return sigma, sigma2, energy, mean


def analyze_noise(ctx: ImageContext) -> AnalyzerResult:
    res = AnalyzerResult(
        key="noise",
        name="Noise & Texture",
        description="Measures sensor-noise grain in flat regions, its consistency across the "
                    "image, and the 'waxy' over-smoothness typical of generated textures.",
    )
    g = ctx.gray
    if min(g.shape) < 96:
        res.applicable = False
        res.add("Image too small for noise analysis.", "info")
        return res

    sigma, sigma2, energy, mean = noise_maps(g)
    unclipped = (mean > 12) & (mean < 243)
    if unclipped.sum() < 20:
        res.applicable = False
        res.add("Image is almost entirely over/under-exposed.", "info")
        return res

    e_vals = energy[unclipped]
    flat_thr = np.percentile(e_vals, 30)
    flat = unclipped & (energy <= flat_thr)
    content = unclipped & (energy >= np.percentile(e_vals, 60))

    flat_sigma = sigma2[flat]
    flat_noise = float(np.median(flat_sigma))
    clean_fraction = float(np.mean(flat_sigma < 0.35))
    noise_cv = float(np.std(flat_sigma) / (np.mean(flat_sigma) + 1e-6))
    # fine detail relative to edge strength in textured regions
    detail_ratio = float(np.median(sigma[content] / (energy[content] + 1e-3)))

    llr = 0.0
    if flat_noise < 0.35:
        llr += 1.2
        res.add(f"Flat regions are almost noise-free (σ≈{flat_noise:.2f}). Real camera "
                "photos nearly always carry visible sensor grain.", "ai")
    elif flat_noise < 0.8:
        llr += ramp(flat_noise, 0.8, 0.35, 0.2, 0.9)
        res.add(f"Low noise in flat regions (σ≈{flat_noise:.2f}).", "ai")
    elif flat_noise > 1.6:
        llr -= ramp(flat_noise, 1.6, 3.5, 0.5, 1.0)
        res.add(f"Clear sensor-like grain in flat regions (σ≈{flat_noise:.2f}).", "real")
    else:
        llr -= 0.2
        res.add(f"Moderate noise in flat regions (σ≈{flat_noise:.2f}).", "info")

    if clean_fraction > 0.5:
        llr += 0.4
        res.add(f"{clean_fraction:.0%} of flat blocks are perfectly smooth.", "ai")

    if noise_cv > 1.1:
        llr += 0.5
        res.add(f"Noise level varies strongly between regions (CV {noise_cv:.2f}) — "
                "possible compositing, in-painting or heavy local retouching.", "ai")
    elif noise_cv < 0.45:
        res.add(f"Noise is consistent across the image (CV {noise_cv:.2f}).", "real")
        llr -= 0.2

    if detail_ratio < 0.035:
        llr += 0.5
        res.add("Textured surfaces lack fine micro-detail relative to their edges "
                "(smooth, 'plastic' rendering).", "ai")
    elif detail_ratio > 0.09:
        llr -= 0.3
        res.add("Textures contain rich micro-detail.", "real")

    # Compression and down-scaling erase noise; trust this analyzer less then.
    q = ctx.jpeg_quality_estimate()
    weight = 1.0
    if q is not None and q < 85:
        weight = ramp(q, 60, 85, 0.3, 0.8)
        res.add(f"JPEG quality ≈{q} removes fine noise — this check is down-weighted.", "info")
    if max(ctx.width, ctx.height) < 700:
        weight *= 0.6
        res.add("Low resolution (probably down-scaled) — noise evidence is weaker.", "info")
    res.weight = weight
    res.llr = float(np.clip(llr, -1.5, 2.2))
    res.details = {
        "flat_region_noise_sigma": round(flat_noise, 3),
        "clean_flat_block_fraction": round(clean_fraction, 3),
        "noise_variation_cv": round(noise_cv, 3),
        "detail_to_edge_ratio": round(detail_ratio, 4),
        "blocks_analysed": int(unclipped.sum()),
    }

    vis_vals = np.where(unclipped, sigma2, np.nan)
    vis = heatmap(vis_vals, 0, max(2.5, float(np.nanpercentile(vis_vals, 98))))
    vis = cv2.resize(vis, (vis.shape[1] * BLOCK, vis.shape[0] * BLOCK),
                     interpolation=cv2.INTER_NEAREST)
    res.visual = to_data_url(vis)
    res.visual_caption = ("Local noise level (dark = noise-free, bright = grainy). Real photos "
                          "show grain everywhere; suspicious regions stand out as dark islands.")
    return res
