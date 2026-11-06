"""Facial consistency analysis.

Generated faces frequently betray themselves through physically implausible
details. For every detected face we check:

* **Corneal specular highlights** – both eyes reflect the same light sources,
  so the bright catch-lights should have a similar shape and position. GAN and
  diffusion faces often show mismatched highlights (Hu et al., 2021).
* **Eye pair consistency** – size and colour of the two detected eyes.
* **Skin vs. scene noise** – generated (or beauty-filtered) skin is far
  smoother than the sensor noise elsewhere in the picture.
* **Left/right texture balance** – AI faces sometimes render one side of the
  face with noticeably different sharpness.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import AnalyzerResult, ImageContext, clamp, to_data_url

_CASCADES = cv2.data.haarcascades
_face = cv2.CascadeClassifier(_CASCADES + "haarcascade_frontalface_default.xml")
_face_alt = cv2.CascadeClassifier(_CASCADES + "haarcascade_frontalface_alt2.xml")
_eye = cv2.CascadeClassifier(_CASCADES + "haarcascade_eye.xml")

MAX_SIDE = 1600
MAX_FACES = 6


def _hi_mask(eye: np.ndarray) -> np.ndarray | None:
    """Mask of the specular highlight (catch-light), or None if there is none."""
    eye = cv2.resize(eye, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    thr = max(np.percentile(eye, 97), eye.mean() + 1.5 * eye.std())
    mask = eye >= thr
    if mask.sum() < 3 or eye.max() - np.median(eye) < 40:
        return None
    return mask


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 1.0


def _sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _noise_sigma(gray: np.ndarray) -> float:
    g = gray.astype(np.float32)
    return float(np.mean(np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.0))) * 1.25)


def _overlap(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter)


def _eye_pair(eyes, face_w: int):
    """Pick a plausible left/right eye pair: side by side, similar height,
    clearly separated and not overlapping."""
    best = None
    for i in range(len(eyes)):
        for j in range(i + 1, len(eyes)):
            a, b = sorted((eyes[i], eyes[j]), key=lambda e: e[0])
            (ax, ay, aw, ah), (bx, by, bw, bh) = a, b
            sep = (bx + bw / 2) - (ax + aw / 2)
            if sep < 0.25 * face_w or sep > 0.75 * face_w or bx < ax + aw * 0.8:
                continue
            if abs((ay + ah / 2) - (by + bh / 2)) > 0.15 * face_w:
                continue
            score = aw * ah + bw * bh
            if best is None or score > best[0]:
                best = (score, (a, b))
    return best[1] if best else None


def _detect_faces(gray: np.ndarray, min_face: int) -> list:
    """Haar detection, cross-checked by a second cascade and an eye detector
    to suppress the false positives Haar cascades are prone to."""
    eq = cv2.equalizeHist(gray)
    primary = _face.detectMultiScale(eq, scaleFactor=1.1, minNeighbors=6,
                                     minSize=(min_face, min_face))
    if len(primary) == 0:
        return []
    alt = _face_alt.detectMultiScale(eq, scaleFactor=1.1, minNeighbors=4,
                                     minSize=(min_face, min_face))
    confirmed = []
    for f in primary:
        if not any(_overlap(f, a) > 0.4 for a in alt):
            continue
        x, y, w, h = f
        upper = gray[y:y + h // 2 + h // 10, x:x + w]
        if len(_eye.detectMultiScale(upper, 1.1, 5, minSize=(max(8, w // 10),) * 2)) == 0:
            continue
        confirmed.append(f)
    return confirmed


def analyze_faces(ctx: ImageContext) -> AnalyzerResult:
    res = AnalyzerResult(
        key="faces",
        name="Facial Consistency",
        description="Detects faces and checks eye reflections, eye symmetry, skin texture "
                    "and left/right rendering consistency.",
    )
    rgb = ctx.rgb_full
    scale = min(1.0, MAX_SIDE / max(rgb.shape[:2]))
    if scale < 1.0:
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    min_face = max(40, int(min(gray.shape) * 0.06))
    faces = _detect_faces(gray, min_face)
    if len(faces) == 0:
        res.applicable = False
        res.add("No frontal faces detected — facial checks not applicable.", "info")
        res.details = {"faces_detected": 0}
        return res

    faces = sorted(faces, key=lambda f: -f[2] * f[3])[:MAX_FACES]
    scene_noise = _noise_sigma(gray)
    vis = rgb.copy()
    per_face = []
    total = 0.0

    for idx, (x, y, w, h) in enumerate(faces, 1):
        face_g = gray[y:y + h, x:x + w]
        info: dict = {"box": [int(x / scale), int(y / scale), int(w / scale), int(h / scale)]}
        llr = 0.0
        notes = []

        # --- eyes ---------------------------------------------------------
        upper = face_g[: h // 2 + h // 10]
        eyes = _eye.detectMultiScale(upper, 1.1, 5, minSize=(max(8, w // 10),) * 2)
        eyes = _eye_pair(eyes, w)
        if eyes:
            (ax, ay, aw, ah), (bx, by, bw, bh) = eyes
            size_ratio = float(max(aw, bw) / max(min(aw, bw), 1))
            ea = face_g[ay:ay + ah, ax:ax + aw]
            eb = face_g[by:by + bh, bx:bx + bw]
            ma, mb = _hi_mask(ea), _hi_mask(eb)
            iou = _iou(ma, mb) if ma is not None and mb is not None else None
            ca = rgb[y + ay:y + ay + ah, x + ax:x + ax + aw].reshape(-1, 3).mean(0)
            cb = rgb[y + by:y + by + bh, x + bx:x + bx + bw].reshape(-1, 3).mean(0)
            color_diff = float(np.abs(ca - cb).mean())
            info.update(highlight_iou=None if iou is None else round(iou, 3),
                        eye_size_ratio=round(size_ratio, 3),
                        eye_color_diff=round(color_diff, 2))
            if iou is None:
                notes.append(("No clear catch-lights in both eyes — reflection check skipped.", "info"))
            elif iou < 0.15:
                llr += 0.9
                notes.append(("Catch-lights in the two eyes don't match — physically "
                              "inconsistent reflections are a known AI tell.", "ai"))
            elif iou > 0.4:
                llr -= 0.4
                notes.append(("Eye reflections are consistent.", "real"))
            if size_ratio > 1.45:
                llr += 0.3
                notes.append((f"Detected eyes differ in size by {size_ratio:.2f}×.", "ai"))
            if color_diff > 25:
                llr += 0.3
                notes.append(("Noticeable colour difference between the eyes.", "ai"))
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(vis, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (80, 200, 255), 1)
        else:
            notes.append(("Could not locate both eyes — eye checks skipped.", "info"))

        # --- skin texture vs scene ------------------------------------------
        cheeks = face_g[h // 2: int(h * 0.85), w // 5: w - w // 5]
        skin_noise = _noise_sigma(cheeks) if cheeks.size else 0.0
        ratio = skin_noise / (scene_noise + 1e-3)
        info.update(skin_noise=round(skin_noise, 3), scene_noise=round(scene_noise, 3),
                    skin_to_scene_noise=round(ratio, 3))
        if scene_noise > 0.8 and ratio < 0.35:
            llr += 0.7
            notes.append(("Skin is far smoother than the rest of the image (airbrushed / "
                          "generated skin, or a beauty filter).", "ai"))
        elif ratio > 0.7 and skin_noise > 0.8:
            llr -= 0.3
            notes.append(("Skin carries the same grain as the rest of the photo.", "real"))

        # --- left / right sharpness balance -------------------------------------
        left, right = face_g[:, : w // 2], face_g[:, w - w // 2:]
        sl, sr = _sharpness(left), _sharpness(right)
        lr = max(sl, sr) / max(min(sl, sr), 1e-3)
        info["left_right_sharpness_ratio"] = round(lr, 3)
        if lr > 3.0:
            llr += 0.4
            notes.append((f"One half of the face is {lr:.1f}× sharper than the other.", "ai"))

        llr = clamp(llr, -1.0, 2.0)
        info["llr"] = round(llr, 3)
        info["notes"] = [n for n, _ in notes]
        per_face.append(info)
        total += llr
        for n, kind in notes:
            res.add(f"Face {idx}: {n}", kind)
        colour = (255, 80, 80) if llr > 0.5 else (80, 220, 120) if llr < 0 else (255, 200, 60)
        cv2.rectangle(vis, (x, y), (x + w, y + h), colour, max(2, int(w / 80)))
        cv2.putText(vis, str(idx), (x + 4, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

    res.add(f"{len(faces)} face(s) analysed.", "info")
    # Average with a bonus for multiple suspicious faces, capped
    res.llr = clamp(total / len(faces) + 0.15 * max(0, sum(f["llr"] > 0.5 for f in per_face) - 1),
                    -1.0, 2.0)
    res.details = {"faces_detected": len(faces), "faces": per_face}
    res.visual = to_data_url(vis, fmt="JPEG")
    res.visual_caption = "Detected faces (red = suspicious, amber = unclear, green = consistent)."
    return res
