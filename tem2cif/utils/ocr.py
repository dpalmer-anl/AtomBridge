"""
OCR utility with graceful fallback.

Tries EasyOCR, then Tesseract via pytesseract. If neither is available,
returns an empty string so upstream logic can continue.
"""

from __future__ import annotations

from typing import Any


def _to_pil(img: Any):  # pragma: no cover - best effort conversion
    try:
        from PIL import Image
        import numpy as np

        if isinstance(img, Image.Image):
            return img
        if isinstance(img, np.ndarray):
            mode = "RGB"
            if img.ndim == 2:
                mode = "L"
            elif img.ndim == 3 and img.shape[2] == 4:
                mode = "RGBA"
            return Image.fromarray(img.astype("uint8"), mode=mode)
    except Exception:
        pass
    return None


def ocr_text(image: Any) -> str:
    # Try EasyOCR
    try:  # pragma: no cover - optional dep
        import easyocr

        reader = easyocr.Reader(["en"], gpu=False)
        # easyocr can accept path or numpy array; ensure array
        pil = _to_pil(image)
        if pil is None:
            return ""
        import numpy as np

        arr = np.array(pil.convert("RGB"))
        result = reader.readtext(arr, detail=0, paragraph=True)
        return "\n".join(result).strip()
    except Exception:
        pass

    # Try pytesseract
    try:  # pragma: no cover - optional dep
        from PIL import Image
        import pytesseract

        pil = _to_pil(image)
        if pil is None:
            return ""
        text = pytesseract.image_to_string(pil)
        return (text or "").strip()
    except Exception:
        pass

    # Fallback
    return ""
