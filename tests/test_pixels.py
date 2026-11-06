import numpy as np

from app.analysis import analyze_image
from app.analysis.base import ImageContext, estimate_jpeg_quality
from app.analysis.frequency import analyze_frequency
from app.analysis.noise import analyze_noise
from app.analysis.sensor import analyze_sensor
from app.analysis.faces import analyze_faces

from .helpers import camera_like, encode, generated_like

CAMERA = camera_like()
GENERATED = generated_like()


def ctx(arr, fmt="PNG", **kw):
    return ImageContext(encode(arr, fmt, **kw))


def test_sensor_detects_bayer_demosaicing():
    assert analyze_sensor(ctx(CAMERA)).llr <= -1.0
    assert analyze_sensor(ctx(GENERATED)).llr > 0


def test_frequency_finds_periodic_artifacts():
    gen = analyze_frequency(ctx(GENERATED))
    cam = analyze_frequency(ctx(CAMERA))
    assert gen.details["periodic_peaks"] >= 2
    assert cam.details["periodic_peaks"] == 0
    assert gen.llr > cam.llr
    assert gen.visual.startswith("data:image/png;base64,")


def test_noise_prefers_grainy_camera_image():
    cam = analyze_noise(ctx(CAMERA))
    clean = analyze_noise(ctx(np.clip(GENERATED.astype(int), 0, 255).astype(np.uint8), "JPEG", quality=90))
    assert cam.llr < clean.llr


def test_small_image_marks_analyzers_not_applicable():
    tiny = ctx(camera_like(64, 64))
    assert not analyze_frequency(tiny).applicable
    assert not analyze_noise(tiny).applicable


def test_no_faces_in_abstract_image():
    r = analyze_faces(ctx(CAMERA))
    assert not r.applicable
    assert r.details["faces_detected"] == 0


def test_jpeg_quality_estimate():
    for q in (60, 85, 95):
        c = ctx(CAMERA[:128, :128], "JPEG", quality=q)
        assert abs(estimate_jpeg_quality(c.pil_original.quantization) - q) <= 1


def test_end_to_end_ordering():
    cam = analyze_image(encode(CAMERA, "JPEG", quality=95))["summary"]
    gen = analyze_image(encode(GENERATED))["summary"]
    assert cam["ai_probability"] < 0.4
    assert gen["ai_probability"] > 0.6
    assert 0 <= cam["authenticity_score"] <= 100


def test_handles_rgba_grayscale_and_palette():
    rgba = np.dstack([CAMERA[:200, :200], np.full((200, 200), 128, np.uint8)])
    for arr, mode in [(rgba, None), (CAMERA[:200, :200, 0], None)]:
        r = analyze_image(encode(arr))
        assert "summary" in r
    from PIL import Image
    import io
    buf = io.BytesIO()
    Image.fromarray(CAMERA[:200, :200]).convert("P").save(buf, "GIF")
    assert analyze_image(buf.getvalue())["image"]["format"] == "GIF"
