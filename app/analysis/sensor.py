"""Camera pipeline (demosaicing / CFA) trace detection.

Almost every camera sensor sits behind a Bayer colour filter array: each
photosite records only one of R, G or B, and the camera *interpolates*
(demosaics) the missing two. Interpolated green pixels lie on one of the two
quincunx (checkerboard) lattices and are much better predicted by their
neighbours than the originally measured ones. Finding this 2x2 periodic
signature is strong evidence of a real camera; generators never produce it.

The trace is destroyed by down-scaling and heavy compression, so its
*absence* is only very weak evidence for AI.
"""
from __future__ import annotations

import numpy as np

from .base import AnalyzerResult, ImageContext, ramp

B = 32


def cfa_lattice_ratios(green: np.ndarray) -> np.ndarray:
    g = green.astype(np.float32)
    pred = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]) / 4.0
    err = (g[1:-1, 1:-1] - pred) ** 2
    h, w = (err.shape[0] // B) * B, (err.shape[1] // B) * B
    err = err[:h, :w]
    yy, xx = np.indices(err.shape)
    lattice_a = ((yy + xx) % 2 == 0)
    blocks_e = err.reshape(h // B, B, w // B, B).swapaxes(1, 2).reshape(-1, B * B)
    blocks_m = lattice_a.reshape(h // B, B, w // B, B).swapaxes(1, 2).reshape(-1, B * B)
    va = (blocks_e * blocks_m).sum(1) / blocks_m.sum(1)
    vb = (blocks_e * ~blocks_m).sum(1) / (~blocks_m).sum(1)
    keep = (va + vb) > 2.0  # ignore flat blocks
    return np.log((va[keep] + 1e-3) / (vb[keep] + 1e-3))


def analyze_sensor(ctx: ImageContext) -> AnalyzerResult:
    res = AnalyzerResult(
        key="sensor",
        name="Camera Sensor Traces",
        description="Looks for the Bayer-filter demosaicing pattern that real camera sensors "
                    "imprint on every photo.",
    )
    if min(ctx.rgb.shape[:2]) < 128:
        res.applicable = False
        res.add("Image too small.", "info")
        return res
    ratios = cfa_lattice_ratios(ctx.rgb[..., 1])
    if len(ratios) < 30:
        res.applicable = False
        res.add("Not enough textured area to test for demosaicing traces.", "info")
        return res

    med = float(np.median(ratios))
    agreement = float(max(np.mean(ratios > 0), np.mean(ratios < 0)))
    strength = abs(med)
    llr = 0.0
    if strength > 0.15 and agreement > 0.75:
        llr = -ramp(strength, 0.15, 0.6, 1.0, 2.5)
        res.add(f"Bayer demosaicing pattern detected (strength {strength:.2f}, consistent in "
                f"{agreement:.0%} of regions) — strong sign of a real camera capture.", "real")
    elif strength > 0.07 and agreement > 0.65:
        llr = -0.5
        res.add(f"Weak demosaicing pattern detected (strength {strength:.2f}).", "real")
    else:
        llr = 0.15
        res.add("No demosaicing pattern. Expected for generated images, but also for "
                "photos that were resized or recompressed (e.g. via social media).", "info")
    res.llr = llr
    res.details = {
        "cfa_strength": round(strength, 4),
        "lattice_agreement": round(agreement, 3),
        "blocks_tested": int(len(ratios)),
    }
    return res
