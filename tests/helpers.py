"""Builders for images carrying specific metadata."""
from __future__ import annotations

import io
import struct

from PIL import Image, PngImagePlugin

from .synthetic import camera_like, encode, generated_like


def png_with_text(arr, **text) -> bytes:
    info = PngImagePlugin.PngInfo()
    for k, v in text.items():
        info.add_text(k, v)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG", pnginfo=info)
    return buf.getvalue()


def jpeg_with_camera_exif(arr, make="Canon", model="EOS R5") -> bytes:
    img = Image.fromarray(arr)
    exif = Image.Exif()
    exif[271] = make
    exif[272] = model
    sub = exif.get_ifd(0x8769)
    sub[33434] = 1 / 250   # ExposureTime
    sub[33437] = 2.8       # FNumber
    sub[34855] = 200       # ISO
    sub[37386] = 50.0      # FocalLength
    sub[37500] = b"Canon\0" + bytes(200)  # MakerNote
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95, exif=exif.tobytes())
    return buf.getvalue()


def insert_jpeg_segment(jpeg: bytes, marker: int, payload: bytes) -> bytes:
    seg = bytes([0xFF, marker]) + struct.pack(">H", len(payload) + 2) + payload
    return jpeg[:2] + seg + jpeg[2:]


def xmp_packet(body: str) -> bytes:
    return (b"http://ns.adobe.com/xap/1.0/\0<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>"
            b"<x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>"
            + body.encode() + b"</rdf:RDF></x:xmpmeta><?xpacket end='w'?>")


def fake_c2pa_jpeg(arr, generator: str, source_type: str) -> bytes:
    """JPEG with an APP11 JUMBF box resembling a C2PA manifest (unsigned)."""
    jumbf = (b"JP\x00\x01\x00\x00\x00\x01" + b"\x00\x00\x00\x40jumb\x00\x00\x00\x20jumdc2pa"
             + b"\x00" * 8 + b"\x6fclaim_generator" + bytes([0x78, len(generator)])
             + generator.encode() + b"\x6bc2pa.actions digitalSourceType http://cv.iptc.org/newscodes/digitalsourcetype/"
             + source_type.encode())
    return insert_jpeg_segment(encode(arr, "JPEG", quality=92), 0xEB, jumbf)


__all__ = ["camera_like", "generated_like", "encode", "png_with_text", "jpeg_with_camera_exif",
           "insert_jpeg_segment", "xmp_packet", "fake_c2pa_jpeg"]
