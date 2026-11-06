"""Evidence fusion: turns analyzer outputs into one authenticity score."""
from __future__ import annotations

from .base import AnalyzerResult, clamp, sigmoid

# How much each analyzer is trusted in general (multiplied with the
# per-image weight that the analyzer itself reports).
MODULE_WEIGHTS = {
    "metadata": 1.0,
    "frequency": 0.8,
    "noise": 0.9,
    "sensor": 1.0,
    "ela": 1.0,      # ELA already reports a low weight of its own
    "faces": 0.8,
}

# Pixel statistics alone should never claim near-certainty.
PIXEL_EVIDENCE_CAP = 3.0

VERDICTS = [
    (0.85, "Very likely AI-generated", "ai-high"),
    (0.65, "Likely AI-generated", "ai"),
    (0.45, "Inconclusive", "neutral"),
    (0.25, "Likely authentic", "real"),
    (0.0, "Very likely authentic", "real-high"),
]

PLATFORM_NOTES = {
    "Anthropic": ("Anthropic's Claude models do not natively generate photographic images; "
                  "a Claude reference usually means the image was produced by code or a tool "
                  "orchestrated by Claude."),
    "ByteDance (Seedance/Seedream)": ("Seedance is ByteDance's video model and Seedream its "
                                      "image model; stills from either typically carry an "
                                      "AIGC label or ByteDance C2PA data when unmodified."),
    "Google": ("Google AI images also carry an invisible SynthID watermark, which can only be "
               "verified with Google's own detector."),
    "OpenAI": "OpenAI attaches C2PA Content Credentials to images from ChatGPT and its API.",
}


def fuse(results: list[AnalyzerResult]) -> dict:
    meta = next((r for r in results if r.key == "metadata"), None)
    contributions = []
    pixel_total = 0.0
    meta_total = 0.0
    for r in results:
        if not r.applicable:
            contributions.append({"key": r.key, "name": r.name, "contribution": None})
            continue
        c = r.llr * r.weight * MODULE_WEIGHTS.get(r.key, 1.0)
        contributions.append({"key": r.key, "name": r.name, "contribution": round(c, 3)})
        if r.key == "metadata":
            meta_total += c
        else:
            pixel_total += c

    pixel_total = clamp(pixel_total, -PIXEL_EVIDENCE_CAP, PIXEL_EVIDENCE_CAP)
    total = meta_total + pixel_total
    p_ai = sigmoid(total)

    strong_marker = bool(meta and meta.details.get("strong_ai_marker"))
    if strong_marker or abs(total) >= 3.0:
        confidence = "High"
    elif abs(total) >= 1.5:
        confidence = "Medium"
    else:
        confidence = "Low"

    verdict, tone = next((v, t) for thr, v, t in VERDICTS if p_ai >= thr)

    # --- platform attribution ---------------------------------------------
    attribution = {"platform": None, "basis": "none", "candidates": [], "note": None}
    cands = (meta.details.get("platform_candidates") if meta else None) or []
    if cands:
        top = cands[0]
        attribution.update(platform=top["platform"], basis="metadata",
                           candidates=cands, note=PLATFORM_NOTES.get(top["platform"]))
    elif p_ai >= 0.6 and meta and meta.details.get("dimension_hint_platforms"):
        hints = meta.details["dimension_hint_platforms"]
        attribution.update(platform=" / ".join(hints), basis="dimensions",
                           candidates=[{"platform": h, "evidence": ["native output size"],
                                        "strength": 0.8} for h in hints])
    # --- headline reasons ------------------------------------------------------
    applicable = [r for r in results if r.applicable]
    ranked = sorted(applicable, key=lambda r: -abs(r.llr * r.weight))
    reasons = []
    want = "ai" if p_ai >= 0.5 else "real"
    for r in ranked:
        for f in r.findings:
            if f.kind == want and len(reasons) < 4:
                reasons.append({"analyzer": r.name, "text": f.text})
    return {
        "ai_probability": round(p_ai, 4),
        "authenticity_score": round((1 - p_ai) * 100),
        "verdict": verdict,
        "tone": tone,
        "confidence": confidence,
        "total_llr": round(total, 3),
        "metadata_llr": round(meta_total, 3),
        "pixel_llr": round(pixel_total, 3),
        "contributions": contributions,
        "attribution": attribution,
        "key_reasons": reasons,
    }
