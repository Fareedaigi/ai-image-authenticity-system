"""Image authenticity analysis pipeline."""
from __future__ import annotations

import logging
import time

from .base import AnalyzerResult, ImageContext
from .ela import analyze_ela
from .faces import analyze_faces
from .frequency import analyze_frequency
from .metadata import analyze_metadata
from .noise import analyze_noise
from .scoring import fuse
from .sensor import analyze_sensor

log = logging.getLogger(__name__)

ANALYZERS = [
    analyze_metadata,
    analyze_frequency,
    analyze_noise,
    analyze_sensor,
    analyze_faces,
    analyze_ela,
]

DISCLAIMER = ("This score is a probabilistic estimate from forensic heuristics, not proof. "
              "Metadata can be stripped or forged, and re-compression, resizing or filters "
              "can hide or mimic the pixel-level traces this tool measures.")


def analyze_image(data: bytes, filename: str = "upload") -> dict:
    t0 = time.perf_counter()
    ctx = ImageContext(data, filename)
    results: list[AnalyzerResult] = []
    for fn in ANALYZERS:
        t = time.perf_counter()
        try:
            r = fn(ctx)
        except Exception as exc:  # one failing analyzer must not break the report
            log.exception("analyzer %s failed", fn.__name__)
            r = AnalyzerResult(key=fn.__name__.removeprefix("analyze_"),
                               name=fn.__name__.removeprefix("analyze_").title(),
                               description="", applicable=False)
            r.add(f"Analyzer error: {exc}", "info")
        r.details["runtime_ms"] = round((time.perf_counter() - t) * 1000)
        results.append(r)

    summary = fuse(results)
    return {
        "filename": filename,
        "image": {
            "format": ctx.format,
            "width": ctx.width,
            "height": ctx.height,
            "mode": ctx.mode,
            "bytes": len(data),
        },
        "summary": summary,
        "analyzers": [r.to_dict() for r in results],
        "disclaimer": DISCLAIMER,
        "runtime_ms": round((time.perf_counter() - t0) * 1000),
    }
