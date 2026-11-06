# AI Image Authenticity Checker

A web application that analyses an uploaded image and estimates how likely it is to be
**AI-generated** (e.g. by OpenAI ChatGPT / DALL·E, Google Gemini / Imagen, ByteDance
Seedance / Seedream, Midjourney, Stable Diffusion, FLUX) versus an **authentic camera
capture**. It produces an authenticity score from 0–100, a verdict with a confidence
level, a best guess at the source platform, and an explained breakdown of every check.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run.py                      # http://localhost:5000
```

Run the tests with `pytest`.

Images are processed in memory and never written to disk. Upload limit: 25 MB
(JPEG, PNG, WebP, GIF, BMP, TIFF).

## How it works

Each analyzer outputs a **log-likelihood ratio** (positive = evidence for AI, negative =
evidence for a real photo) plus findings and a diagnostic image. The fusion step
(`app/analysis/scoring.py`) adds them up, with per-analyzer weights, and turns the total
into a probability. Pixel-level evidence is capped at ±3 so statistics alone never claim
near-certainty; provenance metadata (signed manifests, generation parameters) can.

| Analyzer | File | What it looks for |
|---|---|---|
| Metadata & Provenance | `metadata.py` | C2PA Content Credentials (OpenAI, Google, Adobe, Microsoft), IPTC `DigitalSourceType` (`trainedAlgorithmicMedia`…), China's AIGC implicit labels (ByteDance / Seedream / Dreamina), generator names, Stable Diffusion / ComfyUI parameters, camera EXIF (make, exposure, MakerNote, GPS), generator-native output sizes, JPEG quantisation tables |
| Frequency Spectrum | `frequency.py` | Periodic peaks in the Fourier spectrum left by up-sampling layers / VAE decoders; abnormal spectral fall-off |
| Noise & Texture | `noise.py` | Missing sensor grain in flat areas, inconsistent noise between regions (splicing / in-painting), "waxy" lack of micro-detail |
| Camera Sensor Traces | `sensor.py` | The Bayer colour-filter demosaicing pattern that real cameras imprint; strong evidence of authenticity when present |
| Facial Consistency | `faces.py` | Per face: mismatched corneal catch-lights, eye size/colour asymmetry, skin far smoother than the scene, left/right sharpness imbalance |
| Error Level Analysis | `ela.py` | Regions with a different compression history (local edits / in-painting) |

### Platform attribution

Naming the *specific* platform is only reliable from metadata — e.g. a C2PA manifest whose
claim generator is ChatGPT, a "Made with Google AI" credential, or an AIGC label from
ByteDance. Without metadata the app says the source is not identifiable (optionally showing
a weak hint from generator-native image dimensions). Notes:

* **Google** images also carry an invisible SynthID watermark that only Google's detector
  can verify.
* **Anthropic** Claude models don't natively generate images, so "Anthropic" is only
  reported when metadata references it.
* **Seedance** is ByteDance's video model; its stills and Seedream images are recognised
  through ByteDance/AIGC metadata.

## API

`POST /api/analyze` — multipart form with field `image`. Returns JSON:

```jsonc
{
  "summary": {
    "authenticity_score": 6,           // 0 = certainly AI, 100 = certainly authentic
    "ai_probability": 0.94,
    "verdict": "Very likely AI-generated",
    "confidence": "High",
    "attribution": { "platform": "OpenAI", "basis": "metadata", "candidates": [...] },
    "contributions": [{ "key": "metadata", "contribution": 9.0 }, ...],
    "key_reasons": [...]
  },
  "analyzers": [{ "key": "frequency", "llr": 1.7, "findings": [...], "details": {...}, "visual": "data:image/png;base64,..." }, ...],
  "disclaimer": "..."
}
```

`GET /api/health` returns `{"status": "ok"}`.

## Limitations — read before relying on a result

* The pixel-level checks are hand-tuned forensic heuristics. They were validated on
  synthetic camera/generator simulations (`tests/synthetic.py`) and a few real photos,
  **not** on a large benchmark of real AI outputs. Calibrating the thresholds (or replacing
  the fusion with a trained classifier) on a labelled dataset is the most valuable next step.
* Screenshots, resizing, social-media re-compression and filters erase the traces these
  checks rely on, so such images often land in *Inconclusive*.
* Metadata can be stripped (common) or forged (possible). C2PA signatures are detected but
  **not cryptographically verified**; adding the official `c2pa` library would fix that.
* Face checks use OpenCV Haar cascades, which only find fairly large frontal faces.

## Project layout

```
app/
  __init__.py          Flask app factory
  routes.py            HTML page + JSON API
  analysis/            analyzers, shared helpers, scoring
  templates/index.html
  static/css, static/js
tests/                 pytest suite + synthetic image generators
run.py                 dev server entry point
```
