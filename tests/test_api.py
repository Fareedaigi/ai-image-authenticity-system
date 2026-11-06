import io

from .helpers import camera_like, encode, png_with_text


def test_index(client):
    r = client.get("/")
    assert r.status_code == 200
    assert b"Image Authenticity Checker" in r.data


def test_health(client):
    assert client.get("/api/health").json == {"status": "ok"}


def test_missing_file(client):
    assert client.post("/api/analyze").status_code == 400


def test_not_an_image(client):
    r = client.post("/api/analyze", data={"image": (io.BytesIO(b"hello world"), "x.txt")})
    assert r.status_code == 415
    assert "error" in r.json


def test_analyze_returns_report(client):
    data = png_with_text(camera_like(300, 400), parameters="Steps: 20, Sampler: DPM++ 2M")
    r = client.post("/api/analyze", data={"image": (io.BytesIO(data), "sd.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    j = r.json
    assert j["filename"] == "sd.png"
    assert {a["key"] for a in j["analyzers"]} == {"metadata", "frequency", "noise", "sensor", "faces", "ela"}
    assert j["summary"]["attribution"]["platform"] == "Stable Diffusion"
    assert j["summary"]["ai_probability"] > 0.9


def test_too_large(client):
    client.application.config["MAX_CONTENT_LENGTH"] = 1000
    data = encode(camera_like(200, 200))
    r = client.post("/api/analyze", data={"image": (io.BytesIO(data), "a.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 413
