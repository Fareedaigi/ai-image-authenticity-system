"""Synthetic test images that mimic camera captures and generated images."""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image


def _scene(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth 'natural-ish' scene: gradients, blobs and some texture."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = 90 + 70 * np.sin(xx / 97.0) * np.cos(yy / 131.0)
    img[..., 1] = 110 + 60 * np.cos(xx / 71.0 + 1)
    img[..., 2] = 120 + 50 * np.sin(yy / 53.0)
    for _ in range(25):
        cy, cx, r = rng.integers(0, h), rng.integers(0, w), rng.integers(15, 90)
        col = rng.integers(30, 225, 3).tolist()
        cv2.circle(img, (int(cx), int(cy)), int(r), col, -1, lineType=cv2.LINE_AA)
    tex = cv2.GaussianBlur(rng.normal(0, 25, (h, w)).astype(np.float32), (0, 0), 2.0)
    img += tex[..., None]
    return np.clip(img, 0, 255)


def camera_like(h=768, w=1024, seed=0) -> np.ndarray:
    """Scene -> Bayer mosaic -> shot/read noise -> demosaic (like a camera)."""
    rng = np.random.default_rng(seed)
    scene = _scene(h, w, rng)
    bayer = np.zeros((h, w), np.float32)
    bayer[0::2, 0::2] = scene[0::2, 0::2, 2]   # B
    bayer[0::2, 1::2] = scene[0::2, 1::2, 1]   # G
    bayer[1::2, 0::2] = scene[1::2, 0::2, 1]   # G
    bayer[1::2, 1::2] = scene[1::2, 1::2, 0]   # R
    bayer += rng.normal(0, 1, bayer.shape) * np.sqrt(np.maximum(bayer, 1) * 0.08) + rng.normal(0, 1.5, bayer.shape)
    bayer = np.clip(bayer, 0, 255).astype(np.uint8)
    return cv2.cvtColor(bayer, cv2.COLOR_BayerBG2RGB)


def generated_like(h=1024, w=1024, seed=1) -> np.ndarray:
    """Noise-free render decoded from an 8x-downsampled 'latent' with a faint
    periodic decoder pattern, like a latent-diffusion VAE output."""
    rng = np.random.default_rng(seed)
    scene = cv2.GaussianBlur(_scene(h, w, rng), (0, 0), 1.5)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    pattern = (np.cos(2 * np.pi * xx / 8) * np.cos(2 * np.pi * yy / 8)
               + 0.5 * np.cos(np.pi * (xx + yy)))
    out = scene + 1.2 * pattern[..., None]
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def encode(arr: np.ndarray, fmt: str = "PNG", **kw) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, fmt, **kw)
    return buf.getvalue()
