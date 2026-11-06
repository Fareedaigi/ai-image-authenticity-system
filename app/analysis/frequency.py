"""Frequency-domain (Fourier spectrum) analysis.

Generative models build images with up-sampling layers (transposed
convolutions, pixel-shuffle, VAE decoders working on 8x8 latent patches).
These leave periodic traces that are invisible in pixel space but appear as
sharp, regularly spaced peaks in the 2-D power spectrum. Natural photographs
instead show a smooth, roughly power-law (1/f^2) fall-off with a flat
sensor-noise floor at the highest frequencies.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import AnalyzerResult, ImageContext, heatmap, ramp, to_data_url

MAX_N = 1024


def _radial_profile(power: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = power.shape[0]
    fy, fx = np.indices(power.shape)
    fy = fy - n // 2
    fx = fx - n // 2
    r = np.sqrt(fx ** 2 + fy ** 2).astype(np.int32)
    counts = np.bincount(r.ravel())
    sums = np.bincount(r.ravel(), power.ravel())
    prof = sums / np.maximum(counts, 1)
    return prof, r, np.stack([fx, fy])


def analyze_frequency(ctx: ImageContext) -> AnalyzerResult:
    res = AnalyzerResult(
        key="frequency",
        name="Frequency Spectrum",
        description="Searches the Fourier spectrum for periodic up-sampling artefacts left "
                    "by GAN / diffusion decoders and checks the natural 1/f fall-off.",
    )
    g = ctx.gray
    h, w = g.shape
    n = min(h, w, MAX_N)
    n -= n % 2
    if n < 128:
        res.applicable = False
        res.add("Image too small for reliable spectral analysis.", "info")
        return res
    y0, x0 = (h - n) // 2, (w - n) // 2
    patch = g[y0:y0 + n, x0:x0 + n].astype(np.float64)
    patch -= patch.mean()
    win = np.outer(np.hanning(n), np.hanning(n))
    spec = np.fft.fftshift(np.fft.fft2(patch * win))
    logp = np.log10(np.abs(spec) ** 2 + 1e-8)

    prof, r, (fx, fy) = _radial_profile(logp)
    nyq = n // 2
    freqs = np.arange(len(prof)) / n  # cycles / pixel

    # --- power-law slope in the mid band ------------------------------------
    band = (freqs >= 0.02) & (freqs <= 0.25)
    slope, intercept = np.polyfit(np.log10(freqs[band]), prof[band], 1)
    # --- high-frequency tail vs. extrapolated power law (informational) ------
    tail = (freqs >= 0.35) & (freqs <= 0.5)
    expected = slope * np.log10(freqs[tail]) + intercept
    hf_residual = float(np.mean(prof[tail] - expected))

    # --- periodic peak detection ---------------------------------------------
    # Whiten with a local background estimate: robust everywhere, including
    # the spectrum corners where radial bins contain only a handful of samples.
    background = cv2.blur(logp.astype(np.float32), (17, 17), borderType=cv2.BORDER_REFLECT)
    resid = logp - background
    valid = r > n * 0.04  # includes the corners (diagonal Nyquist checkerboards)
    valid &= (np.abs(fx) > 2) & (np.abs(fy) > 2)  # axis cross = edge effects
    step = n / 8.0
    on_grid8 = (np.abs(((fx + nyq) % step) - 0) < 1.5) | (np.abs(((fx + nyq) % step) - step) < 1.5)
    on_grid8_y = (np.abs(((fy + nyq) % step) - 0) < 1.5) | (np.abs(((fy + nyq) % step) - step) < 1.5)
    grid8 = on_grid8 | on_grid8_y
    if ctx.is_jpeg:
        valid &= ~grid8  # JPEG 8x8 blocking produces legitimate peaks here
    vals = resid[valid]
    med = np.median(vals)
    mad = np.median(np.abs(vals - med)) * 1.4826 + 1e-9
    z = np.zeros_like(resid)
    z[valid] = (resid[valid] - med) / mad

    thresh = 7.0
    peaks = np.argwhere(z > thresh)
    # cluster neighbouring bins, count each symmetric pair once
    clusters: list[tuple[int, int, float]] = []
    for yy, xx in peaks[np.argsort(-z[z > thresh])]:
        if any(abs(yy - cy) <= 3 and abs(xx - cx) <= 3 for cy, cx, _ in clusters):
            continue
        sy, sx = 2 * (n // 2) - yy, 2 * (n // 2) - xx
        if any(abs(sy - cy) <= 3 and abs(sx - cx) <= 3 for cy, cx, _ in clusters):
            continue
        clusters.append((int(yy), int(xx), float(z[yy, xx])))
        if len(clusters) >= 40:
            break
    n_peaks = len(clusters)
    max_z = float(z.max()) if n_peaks else float(z[valid].max(initial=0))
    grid_peaks = sum(1 for cy, cx, _ in clusters if grid8[cy, cx])

    # --- evidence ---------------------------------------------------------------
    llr = 0.0
    if n_peaks >= 6:
        llr += ramp(max_z, 7, 14, 0.9, 1.8)
        res.add(f"{n_peaks} periodic spectral peaks detected (max {max_z:.1f}σ) — typical "
                "of up-sampling artefacts in generated images, but also of strongly "
                "repetitive textures (fabric, tiles, screens).", "ai")
    elif n_peaks >= 2:
        llr += 0.5
        res.add(f"{n_peaks} isolated spectral peaks (max {max_z:.1f}σ).", "ai")
    else:
        llr -= 0.3
        res.add("No periodic peaks — spectrum looks free of up-sampling grids.", "real")
    if not ctx.is_jpeg and grid_peaks >= 2:
        res.add(f"{grid_peaks} peaks lie on the 8-pixel grid (VAE patch size / prior JPEG "
                "compression).", "ai")
        llr += 0.3

    if slope < -3.3:
        llr += ramp(slope, -3.3, -4.2, 0.3, 0.9)
        res.add(f"Spectrum falls off steeply (slope {slope:.2f}) — unusually little fine "
                "detail, seen in generated or heavily smoothed/up-scaled images.", "ai")
    elif slope > -2.4:
        res.add(f"Spectral slope {slope:.2f}: rich fine detail.", "real")
        llr -= 0.2
    else:
        res.add(f"Spectral slope {slope:.2f} is within the natural-image range.", "info")

    res.llr = float(np.clip(llr, -1.5, 2.5))
    res.details = {
        "analysed_patch": f"{n}×{n}",
        "spectral_slope": round(float(slope), 3),
        "hf_residual": round(hf_residual, 3),
        "periodic_peaks": n_peaks,
        "max_peak_sigma": round(max_z, 2),
        "peaks_on_8px_grid": grid_peaks,
        "jpeg_grid_excluded": ctx.is_jpeg,
    }

    # --- visual: normalised log-spectrum, peaks marked ------------------------------
    disp = np.clip(resid, -1.5, 3.0)
    vis = heatmap(disp)
    for cy, cx, _ in clusters[:20]:
        for yy, xx in ((cy, cx), (2 * (n // 2) - cy, 2 * (n // 2) - cx)):
            y_a, y_b = max(yy - 6, 0), min(yy + 7, n)
            x_a, x_b = max(xx - 6, 0), min(xx + 7, n)
            vis[y_a:y_b, [x_a, x_b - 1]] = (0, 255, 200)
            vis[[y_a, y_b - 1], x_a:x_b] = (0, 255, 200)
    res.visual = to_data_url(vis, max_side=512)
    res.visual_caption = ("Whitened power spectrum (centre = low frequencies). Bright isolated "
                          "dots, boxed in cyan, are periodic artefacts.")
    return res
