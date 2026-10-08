from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zxingcpp
from PIL import Image, ImageOps


ImageInput = Image.Image | str | Path | np.ndarray


def extract_qr_payload(image: ImageInput | None) -> str | None:
    """Decode the first non-empty QR payload from a PIL image, path, or array."""
    if image is None:
        return None

    try:
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened_image:
                source: Any = ImageOps.exif_transpose(opened_image).convert("RGB")
        elif isinstance(image, Image.Image):
            source = ImageOps.exif_transpose(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            source = image
        else:
            return None

        payloads: list[str] = []
        for barcode in zxingcpp.read_barcodes(source, formats=zxingcpp.BarcodeFormat.QRCode):
            payload = barcode.text.strip()
            if payload and payload not in payloads:
                payloads.append(payload)
        if payloads:
            return "\n".join(payloads)
    except (OSError, TypeError, ValueError):
        return None
    return None
