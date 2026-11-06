from app.analysis import analyze_image
from app.analysis.base import ImageContext
from app.analysis.metadata import analyze_metadata

from .helpers import (camera_like, encode, fake_c2pa_jpeg, generated_like, insert_jpeg_segment,
                      jpeg_with_camera_exif, png_with_text, xmp_packet)

SMALL = camera_like(256, 320)


def meta(data):
    return analyze_metadata(ImageContext(data))


def top_platform(r):
    cands = r.details["platform_candidates"]
    return cands[0]["platform"] if cands else None


def test_stable_diffusion_parameters_in_png():
    data = png_with_text(SMALL, parameters="a cat, Steps: 30, Sampler: Euler a, CFG scale: 7")
    r = meta(data)
    assert r.llr >= 5
    assert top_platform(r) == "Stable Diffusion"
    assert "generation_parameters" in r.details


def test_comfyui_workflow():
    r = meta(png_with_text(SMALL, workflow='{"nodes":[{"class_type":"KSampler"}]}'))
    assert r.llr >= 5
    assert top_platform(r) == "Stable Diffusion"


def test_c2pa_openai_attribution():
    data = fake_c2pa_jpeg(SMALL, "ChatGPT", "trainedAlgorithmicMedia")
    r = meta(data)
    assert r.details["c2pa_present"]
    assert r.details["digital_source_type"] == "trainedalgorithmicmedia"
    assert top_platform(r) == "OpenAI"
    assert r.llr >= 8


def test_c2pa_google_ai():
    r = meta(fake_c2pa_jpeg(SMALL, "Google C2PA Core Generator Library", "trainedAlgorithmicMedia"))
    assert top_platform(r) == "Google"


def test_pixel_camera_c2pa_is_not_attributed_to_google_ai():
    r = meta(fake_c2pa_jpeg(SMALL, "Google Pixel Camera", "computationalCapture"))
    assert top_platform(r) is None
    assert r.llr < 0


def test_ai_edited_composite():
    r = meta(fake_c2pa_jpeg(SMALL, "Adobe Photoshop", "compositeWithTrainedAlgorithmicMedia"))
    assert r.details["digital_source_type"] == "compositewithtrainedalgorithmicmedia"
    assert r.llr >= 4


def test_bytedance_aigc_label():
    xmp = xmp_packet('<rdf:Description AIGC=\'{"Label":"1","ContentProducer":"bytedance-jimeng"}\'/>')
    data = insert_jpeg_segment(encode(SMALL, "JPEG", quality=92), 0xE1, xmp)
    r = meta(data)
    assert r.llr >= 5
    assert top_platform(r) == "ByteDance (Seedance/Seedream)"


def test_seedream_and_gemini_signatures():
    for word, platform in [("Seedream 4.0", "ByteDance (Seedance/Seedream)"),
                           ("Gemini 2.5 Flash Image", "Google"),
                           ("Created by Anthropic tooling", "Anthropic"),
                           ("Midjourney v7", "Midjourney")]:
        xmp = xmp_packet(f"<rdf:Description><dc:creator>{word}</dc:creator></rdf:Description>")
        r = meta(insert_jpeg_segment(encode(SMALL, "JPEG", quality=92), 0xE1, xmp))
        assert top_platform(r) == platform, word


def test_camera_exif_supports_authenticity():
    r = meta(jpeg_with_camera_exif(SMALL))
    assert r.llr <= -2
    assert r.details["exif"]["Make"] == "Canon"
    assert any(f.kind == "real" for f in r.findings)


def test_plain_google_make_is_not_ai():
    r = meta(jpeg_with_camera_exif(SMALL, make="Google", model="Pixel 9"))
    assert top_platform(r) is None
    assert r.llr < 0


def test_known_generator_dimensions_hint():
    r = meta(encode(generated_like(1536, 1024)))
    assert "OpenAI (GPT-image / DALL·E 3)" in r.details["dimension_hint_platforms"]
    assert r.llr > 0


def test_metadata_overrides_pixels_in_final_score():
    report = analyze_image(fake_c2pa_jpeg(camera_like(512, 512), "ChatGPT", "trainedAlgorithmicMedia"))
    s = report["summary"]
    assert s["ai_probability"] > 0.9
    assert s["attribution"]["platform"] == "OpenAI"
    assert s["confidence"] == "High"


def test_c2pa_claim_generator_is_decoded():
    r = meta(fake_c2pa_jpeg(SMALL, "ChatGPT", "trainedAlgorithmicMedia"))
    assert r.details["c2pa_generators"][0] == "ChatGPT"
