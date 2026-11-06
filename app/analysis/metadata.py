"""Metadata & provenance analysis.

Looks at everything an image file carries besides its pixels:

* EXIF camera data (make/model, exposure settings, MakerNote, GPS...)
* C2PA / Content Credentials manifests (JUMBF boxes) used by OpenAI, Google,
  Adobe, Microsoft and others to label AI output
* IPTC ``DigitalSourceType`` values such as ``trainedAlgorithmicMedia``
* China's AIGC implicit-label metadata (used by ByteDance/Seedream/Dreamina...)
* Generation parameters left by Stable Diffusion UIs, ComfyUI, InvokeAI...
* Output dimensions typical for specific generators
* JPEG quantisation tables

Metadata is the only signal that can attribute an image to a *specific*
platform with high confidence. Its absence proves nothing: most social
networks and messengers strip metadata on upload, and metadata can be forged.
"""
from __future__ import annotations

import re
import struct
import zlib
from dataclasses import dataclass

from PIL import ExifTags

from .base import AnalyzerResult, ImageContext, clamp, jpeg_tables_are_standard


# --------------------------------------------------------------------------
# Raw metadata extraction (format-aware so we never match pixel data)
# --------------------------------------------------------------------------

def _jpeg_segments(data: bytes) -> list[tuple[str, bytes]]:
    segs: list[tuple[str, bytes]] = []
    i = 2
    n = len(data)
    while i + 4 <= n:
        if data[i] != 0xFF:
            break
        marker = data[i + 1]
        if marker == 0xFF:  # fill byte
            i += 1
            continue
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if marker == 0xDA:  # start of scan: metadata is over
            break
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        payload = data[i + 4:i + 2 + length]
        if 0xE0 <= marker <= 0xEF:
            segs.append((f"APP{marker - 0xE0}", payload))
        elif marker == 0xFE:
            segs.append(("COM", payload))
        i += 2 + length
    return segs


_PNG_SKIP = {b"IHDR", b"IDAT", b"PLTE", b"IEND", b"tRNS", b"fdAT", b"fcTL"}


def _png_chunks(data: bytes) -> tuple[list[tuple[str, bytes]], dict[str, str]]:
    segs: list[tuple[str, bytes]] = []
    text: dict[str, str] = {}
    i = 8
    n = len(data)
    while i + 8 <= n:
        length = struct.unpack(">I", data[i:i + 4])[0]
        ctype = data[i + 4:i + 8]
        body = data[i + 8:i + 8 + length]
        i += 12 + length
        if ctype in _PNG_SKIP:
            continue
        try:
            if ctype == b"tEXt":
                k, _, v = body.partition(b"\0")
                text[k.decode("latin-1")] = v.decode("latin-1", "replace")
                body = k + b"\0" + v
            elif ctype == b"zTXt":
                k, _, rest = body.partition(b"\0")
                v = zlib.decompress(rest[1:])
                text[k.decode("latin-1")] = v.decode("latin-1", "replace")
                body = k + b"\0" + v
            elif ctype == b"iTXt":
                k, _, rest = body.partition(b"\0")
                flag, _method = rest[0], rest[1]
                rest = rest[2:]
                _lang, _, rest = rest.partition(b"\0")
                _tkey, _, v = rest.partition(b"\0")
                if flag:
                    v = zlib.decompress(v)
                text[k.decode("latin-1")] = v.decode("utf-8", "replace")
                body = k + b"\0" + v
        except Exception:  # malformed chunk: keep raw bytes
            pass
        segs.append((ctype.decode("latin-1"), body))
    return segs, text


_WEBP_SKIP = {b"VP8 ", b"VP8L", b"ALPH", b"ANMF", b"ANIM", b"VP8X"}


def _webp_chunks(data: bytes) -> list[tuple[str, bytes]]:
    segs = []
    i = 12
    n = len(data)
    while i + 8 <= n:
        fourcc = data[i:i + 4]
        size = struct.unpack("<I", data[i + 4:i + 8])[0]
        body = data[i + 8:i + 8 + size]
        i += 8 + size + (size & 1)
        if fourcc not in _WEBP_SKIP:
            segs.append((fourcc.decode("latin-1").strip(), body))
    return segs


def extract_metadata_blocks(ctx: ImageContext) -> tuple[list[tuple[str, bytes]], dict[str, str]]:
    data = ctx.data
    text: dict[str, str] = {}
    if data[:2] == b"\xff\xd8":
        segs = _jpeg_segments(data)
    elif data[:8] == b"\x89PNG\r\n\x1a\n":
        segs, text = _png_chunks(data)
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        segs = _webp_chunks(data)
    else:
        # Unknown container (TIFF, GIF, BMP...): fall back to scanning the
        # start and end of the file, where metadata normally lives.
        segs = [("RAW", data[:1 << 20] + (data[-(1 << 20):] if len(data) > 2 << 20 else b""))]
    return segs, text


# --------------------------------------------------------------------------
# Signature tables
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Sig:
    platform: str
    pattern: str  # regex, applied to lower-cased latin-1/utf-8 text
    label: str
    llr: float


PLATFORM_SIGNATURES: list[Sig] = [
    # OpenAI
    Sig("OpenAI", r"openai", "‘OpenAI’ in metadata", 5.0),
    Sig("OpenAI", r"chatgpt", "‘ChatGPT’ in metadata", 5.0),
    Sig("OpenAI", r"dall(?:[\- ]|\xe2?\xb7)?e\b", "‘DALL·E’ in metadata", 5.0),
    Sig("OpenAI", r"gpt-?4o|gpt-image", "GPT image model named in metadata", 5.0),
    Sig("OpenAI", r"\bsora\b", "‘Sora’ in metadata", 4.0),
    # Google
    Sig("Google", r"made with google ai", "‘Made with Google AI’ credit", 6.0),
    Sig("Google", r"\bgemini\b", "‘Gemini’ in metadata", 5.0),
    Sig("Google", r"\bimagen\b", "‘Imagen’ in metadata", 5.0),
    Sig("Google", r"synthid", "SynthID watermark referenced in metadata", 6.0),
    Sig("Google", r"nano[ \-]?banana", "‘Nano Banana’ (Gemini image) in metadata", 5.0),
    # Anthropic
    Sig("Anthropic", r"anthropic", "‘Anthropic’ in metadata", 4.0),
    Sig("Anthropic", r"claude\.ai|claude[ \-](?:opus|sonnet|haiku|fable|\d)", "Claude model named in metadata", 4.0),
    # ByteDance (Seedance / Seedream / Dreamina / Jimeng / Doubao)
    Sig("ByteDance (Seedance/Seedream)", r"seedance", "‘Seedance’ in metadata", 5.0),
    Sig("ByteDance (Seedance/Seedream)", r"seedream", "‘Seedream’ in metadata", 5.0),
    Sig("ByteDance (Seedance/Seedream)", r"dreamina", "‘Dreamina’ in metadata", 5.0),
    Sig("ByteDance (Seedance/Seedream)", r"jimeng|\xe5\x8d\xb3\xe6\xa2\xa6", "‘Jimeng / 即梦’ in metadata", 5.0),
    Sig("ByteDance (Seedance/Seedream)", r"doubao|\xe8\xb1\x86\xe5\x8c\x85", "‘Doubao / 豆包’ in metadata", 4.0),
    Sig("ByteDance (Seedance/Seedream)", r"bytedance", "‘ByteDance’ in metadata", 2.0),
    # Others
    Sig("Midjourney", r"midjourney", "‘Midjourney’ in metadata", 5.0),
    Sig("Stable Diffusion", r"stable[ \-_]?diffusion|stability\.?ai|dreamstudio|\bsdxl\b", "Stable Diffusion named in metadata", 5.0),
    Sig("Stable Diffusion", r"comfyui|\"class_type\"", "ComfyUI workflow embedded", 6.0),
    Sig("Stable Diffusion", r"invokeai|automatic1111|fooocus|novelai", "Stable Diffusion front-end named in metadata", 5.0),
    Sig("Adobe Firefly", r"firefly", "‘Adobe Firefly’ in metadata", 5.0),
    Sig("Black Forest Labs (FLUX)", r"black forest labs|flux\.1|flux[\-_ ](?:dev|pro|schnell|kontext)", "FLUX model named in metadata", 5.0),
    Sig("Microsoft", r"bing image creator|microsoft designer|image creator from microsoft", "Microsoft image generator named", 5.0),
    Sig("Meta", r"imagine with meta|meta ai", "Meta AI named in metadata", 4.0),
    Sig("xAI (Grok)", r"\bgrok\b", "‘Grok’ in metadata", 4.0),
    Sig("Ideogram", r"ideogram", "‘Ideogram’ in metadata", 5.0),
    Sig("Leonardo.ai", r"leonardo\.ai", "‘Leonardo.ai’ in metadata", 5.0),
    Sig("Runway", r"runwayml|runway gen", "Runway named in metadata", 4.0),
    Sig("Kling", r"\bkling\b|kuaishou", "Kling named in metadata", 4.0),
]

# IPTC digital source type vocabulary (also used inside C2PA manifests)
SOURCE_TYPES = [
    ("compositewithtrainedalgorithmicmedia", "IPTC/C2PA source type: composite with AI-generated elements (AI-edited)", 4.0),
    ("trainedalgorithmicmedia", "IPTC/C2PA source type: trainedAlgorithmicMedia (fully AI-generated)", 7.0),
    ("compositesynthetic", "IPTC source type: composite synthetic", 2.5),
    ("algorithmicmedia", "IPTC source type: algorithmicMedia (computer generated)", 2.5),
    ("digitalcapture", "IPTC/C2PA source type: digitalCapture (camera capture)", -2.5),
    ("computationalcapture", "IPTC/C2PA source type: computationalCapture (smartphone camera)", -2.0),
    ("negativefilm", "IPTC source type: scanned negative film", -1.5),
    ("positivefilm", "IPTC source type: scanned positive film", -1.5),
]

GEN_PARAM_KEYS = {"parameters", "prompt", "workflow", "negative_prompt",
                  "sd-metadata", "invokeai_metadata", "dream", "generation_data"}

# Native output sizes of popular generators (w, h) -> platforms
KNOWN_AI_SIZES: dict[tuple[int, int], list[str]] = {}


def _add_sizes(platform: str, sizes: list[tuple[int, int]], both: bool = True):
    for w, h in sizes:
        for s in ({(w, h), (h, w)} if both else {(w, h)}):
            KNOWN_AI_SIZES.setdefault(s, [])
            if platform not in KNOWN_AI_SIZES[s]:
                KNOWN_AI_SIZES[s].append(platform)


_add_sizes("OpenAI (GPT-image / DALL·E 3)", [(1024, 1024), (1024, 1536), (1792, 1024)])
_add_sizes("Google (Gemini / Imagen)", [(1024, 1024), (832, 1248), (864, 1184), (896, 1152),
                                        (768, 1344), (1536, 672), (1408, 768), (1280, 896)])
_add_sizes("Stable Diffusion / FLUX", [(512, 512), (512, 768), (768, 768), (1024, 1024),
                                       (1152, 896), (1216, 832), (1344, 768), (1536, 640)])
_add_sizes("Midjourney", [(1024, 1024), (2048, 2048), (1456, 816), (1232, 928), (1344, 896),
                          (1792, 1024), (2912, 1632)])
_add_sizes("ByteDance (Seedance/Seedream)", [(2048, 2048), (2304, 1728), (2496, 1664),
                                              (2560, 1440), (3024, 1296)])

EXIF_NAMES = {v: k for k, v in ExifTags.TAGS.items()}


# --------------------------------------------------------------------------
# Analyzer
# --------------------------------------------------------------------------

def _printable(s: str, limit: int = 160) -> str:
    s = re.sub(r"[^\x20-\x7e\u00a0-\uffff]+", " ", s).strip()
    return s[:limit] + ("…" if len(s) > limit else "")


def _cbor_text_after(blob: bytes, pos: int) -> str | None:
    """Decode a CBOR text string (major type 3) starting at ``pos``."""
    if pos >= len(blob):
        return None
    b = blob[pos]
    if 0x60 <= b < 0x78:
        n, start = b - 0x60, pos + 1
    elif b == 0x78 and pos + 1 < len(blob):
        n, start = blob[pos + 1], pos + 2
    elif b == 0x79 and pos + 2 < len(blob):
        n, start = int.from_bytes(blob[pos + 1:pos + 3], "big"), pos + 3
    else:
        return None
    raw = blob[start:start + min(n, 200)]
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _c2pa_strings(blob: bytes) -> list[str]:
    """Pull human-readable claim-generator / software-agent strings from a
    C2PA manifest (CBOR-encoded). Falls back to the next printable run."""
    out: list[str] = []
    for key in (b"claim_generator", b"softwareAgent", b"name"):
        for m in re.finditer(re.escape(key), blob):
            val = _cbor_text_after(blob, m.end())
            if val is None or not val.isprintable():
                run = re.search(rb"[\x20-\x7e]{4,}", blob[m.end():m.end() + 120])
                val = run.group().decode("ascii") if run else None
            if not val:
                continue
            val = val.strip()
            if (len(val) >= 3 and val not in out and not val.startswith(("c2pa", "http", "_"))):
                out.append(val[:100])
    return out[:8]


def analyze_metadata(ctx: ImageContext) -> AnalyzerResult:
    r = AnalyzerResult(
        key="metadata",
        name="Metadata & Provenance",
        description="EXIF camera data, C2PA Content Credentials, IPTC source type, "
                    "AI-generator signatures and generation parameters.",
    )
    segs, png_text = extract_metadata_blocks(ctx)
    blob = b"\n".join(p for _, p in segs)
    low = blob.lower().decode("latin-1")
    platforms: dict[str, dict] = {}
    llr = 0.0
    strong_ai = False

    def hit_platform(name: str, why: str, strength: float):
        p = platforms.setdefault(name, {"platform": name, "evidence": [], "strength": 0.0})
        if why not in p["evidence"]:
            p["evidence"].append(why)
        p["strength"] = max(p["strength"], strength)

    # ---- C2PA -------------------------------------------------------------
    has_c2pa = (b"c2pa" in blob.lower() and (b"jumb" in blob or b"jumd" in blob)) \
        or any(name == "caBX" for name, _ in segs)
    c2pa_generators: list[str] = []
    if has_c2pa:
        c2pa_generators = _c2pa_strings(blob)
        r.add("Content Credentials (C2PA) manifest present. Note: the cryptographic "
              "signature is not verified by this tool.", "info")
        if c2pa_generators:
            r.add("C2PA claim generator / agents: " + ", ".join(c2pa_generators), "info")

    # ---- IPTC digital source type ------------------------------------------
    source_type = None
    for token, label, val in SOURCE_TYPES:
        if token in low:
            source_type = token
            r.add(label, "ai" if val > 0 else "real")
            llr += val
            if val >= 4:
                strong_ai = True
            break  # tokens are ordered so the most specific matches first

    # ---- AIGC implicit label (China, GB 45438-2025) ------------------------
    if re.search(r"aigc", low):
        m = re.search(r"\"?label\"?\s*[:=]\s*\"?(\d)", low)
        if m is None or m.group(1) == "1":  # no explicit label value, or "1" = AI
            r.add("AIGC implicit label found (Chinese AI-content labelling standard).", "ai")
            llr += 5.0
            strong_ai = True
            producer = re.search(r"contentproducer\"?\s*[:=]\s*\"?([^\",}]{2,60})", low)
            if producer:
                r.add(f"AIGC content producer: {producer.group(1)}", "info")
                if any(t in producer.group(1) for t in ("bytedance", "douyin", "jimeng", "doubao")):
                    hit_platform("ByteDance (Seedance/Seedream)", "AIGC label producer", 6.0)

    # ---- platform signatures -------------------------------------------------
    sig_llr = 0.0
    for sig in PLATFORM_SIGNATURES:
        if re.search(sig.pattern, low):
            hit_platform(sig.platform, sig.label, sig.llr)
            sig_llr = max(sig_llr, sig.llr)
    # "Google" alone is ambiguous (Pixel phones write Make=Google); only count
    # it when combined with an AI source type.
    if "google" in low and source_type and "trainedalgorithmic" in source_type:
        hit_platform("Google", "Google-signed manifest with AI source type", 6.0)
    if "microsoft" in low and source_type and "trainedalgorithmic" in source_type:
        hit_platform("Microsoft", "Microsoft-signed manifest with AI source type", 5.0)
    if sig_llr:
        names = ", ".join(sorted(platforms))
        r.add(f"AI generator signature(s) found: {names}", "ai")
        llr += sig_llr
        strong_ai = True

    # ---- generation parameters ----------------------------------------------
    gen_keys = sorted(k for k in png_text if k.lower() in GEN_PARAM_KEYS)
    if gen_keys or re.search(r"steps:\s*\d+.{0,200}sampler", low, re.S):
        r.add("Diffusion generation parameters embedded "
              f"({', '.join(gen_keys) or 'Steps/Sampler/CFG text'}).", "ai")
        llr += 6.0
        strong_ai = True
        hit_platform("Stable Diffusion", "Generation parameters (prompt / sampler / steps)", 6.0)
        snippet = png_text.get("parameters") or png_text.get("prompt") or ""
        if snippet:
            r.details["generation_parameters"] = _printable(snippet, 400)

    # ---- EXIF camera evidence -----------------------------------------------
    exif_summary: dict[str, str] = {}
    camera_llr = 0.0
    try:
        exif = ctx.pil_original.getexif()
    except Exception:
        exif = {}
    if exif:
        base = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.items()}
        try:
            sub = {ExifTags.TAGS.get(k, str(k)): v for k, v in exif.get_ifd(0x8769).items()}
        except Exception:
            sub = {}
        try:
            gps = exif.get_ifd(0x8825)
        except Exception:
            gps = {}
        for k in ("Make", "Model", "Software", "DateTime", "Artist", "ImageDescription", "Copyright"):
            if k in base:
                exif_summary[k] = _printable(str(base[k]), 120)
        for k in ("DateTimeOriginal", "ExposureTime", "FNumber", "ISOSpeedRatings",
                  "FocalLength", "LensMake", "LensModel", "BodySerialNumber"):
            if k in sub:
                exif_summary[k] = _printable(str(sub[k]), 120)
        if "MakerNote" in sub:
            exif_summary["MakerNote"] = f"present ({len(bytes(sub['MakerNote'])) if isinstance(sub['MakerNote'], (bytes, bytearray)) else '?'} bytes)"
        if gps:
            exif_summary["GPS"] = "present"

        make = str(base.get("Make", "")).strip()
        model = str(base.get("Model", "")).strip()
        if make and model:
            camera_llr -= 0.8
            r.add(f"Camera identified in EXIF: {make} {model}", "real")
        exposure = [k for k in ("ExposureTime", "FNumber", "ISOSpeedRatings", "FocalLength") if k in sub]
        if len(exposure) >= 2:
            camera_llr -= 1.0
            r.add("Exposure settings recorded (" + ", ".join(exposure) + ")", "real")
        if "MakerNote" in sub:
            camera_llr -= 1.0
            r.add("Manufacturer MakerNote present — rarely written by generators or forged.", "real")
        if "LensModel" in sub:
            camera_llr -= 0.3
        if gps:
            camera_llr -= 0.4
            r.add("GPS location embedded.", "real")
        soft = str(base.get("Software", ""))
        if re.search(r"photoshop|lightroom|gimp|snapseed|affinity|capture one|picsart|canva", soft, re.I):
            r.add(f"Edited with: {_printable(soft, 60)}", "info")
    camera_llr = clamp(camera_llr, -3.0, 0.0)
    if strong_ai and camera_llr < 0:
        r.add("Camera EXIF coexists with AI markers — EXIF may have been copied or the "
              "photo was AI-edited.", "info")
        camera_llr *= 0.3
    llr += camera_llr

    # ---- dimensions -----------------------------------------------------------
    w, h = ctx.width, ctx.height
    size_platforms = KNOWN_AI_SIZES.get((w, h), [])
    r.details["dimension_hint_platforms"] = size_platforms
    if camera_llr > -1.5:
        if size_platforms:
            llr += 0.6
            r.add(f"{w}×{h} is a native output size of: {', '.join(size_platforms)}", "ai")
        elif w % 64 == 0 and h % 64 == 0 and max(w, h) <= 2048:
            llr += 0.3
            r.add(f"{w}×{h}: both sides are multiples of 64, common for diffusion models.", "ai")

    # ---- JPEG tables ------------------------------------------------------------
    q = ctx.jpeg_quality_estimate()
    if q is not None:
        tables = getattr(ctx.pil_original, "quantization", {}) or {}
        standard = jpeg_tables_are_standard(tables)
        r.details["jpeg_quality_estimate"] = q
        r.details["jpeg_standard_tables"] = standard
        r.add(f"JPEG quality ≈ {q} ({'standard libjpeg' if standard else 'custom/vendor'} "
              "quantisation tables)", "info")
        if not standard and camera_llr < 0:
            llr -= 0.3  # vendor tables consistent with in-camera JPEG

    # ---- nothing at all? ----------------------------------------------------------
    meaningful = [n for n, p in segs if n not in ("APP0", "ICCP", "APP2", "iCCP", "pHYs", "sRGB",
                                                    "gAMA", "cHRM", "bKGD", "tIME")]
    if not meaningful and not exif:
        llr += 0.4
        r.add("No EXIF/XMP/provenance metadata. Common for AI output — but also for "
              "screenshots and images re-shared via social media.", "ai")

    r.llr = clamp(llr, -4.0, 9.0)
    r.details.update({
        "format": ctx.format,
        "dimensions": f"{w}×{h}",
        "mode": ctx.mode,
        "file_size_bytes": len(ctx.data),
        "c2pa_present": bool(has_c2pa),
        "c2pa_generators": c2pa_generators,
        "digital_source_type": source_type,
        "exif": exif_summary,
        "png_text_keys": sorted(png_text),
        "metadata_blocks": sorted({n for n, _ in segs}),
        "platform_candidates": sorted(platforms.values(), key=lambda p: -p["strength"]),
        "strong_ai_marker": strong_ai,
    })
    return r
