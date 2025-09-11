from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tem2cif.state import S


def _to_np(img: Any):
    try:
        import numpy as np
        from PIL import Image

        if isinstance(img, np.ndarray):
            return img
        if isinstance(img, Image.Image):
            return np.array(img)
    except Exception:
        pass
    return None


def _crop(img: Any, bbox: Optional[List[int]]):
    if img is None:
        return None
    if not bbox:
        return _to_np(img)
    try:
        import numpy as np

        arr = _to_np(img)
        if arr is None:
            return None
        x1, y1, b3, b4 = bbox
        if b3 > 0 and b4 > 0:
            x2, y2 = x1 + b3, y1 + b4
        else:
            x2, y2 = b3, b4
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        h, w = arr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return arr
        return arr[y1:y2, x1:x2]
    except Exception:
        return _to_np(img)


def _gray(img_np):
    try:
        import cv2
        import numpy as np

        if img_np is None:
            return None
        if img_np.ndim == 2:
            return img_np
        if img_np.ndim == 3:
            return cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) if img_np.shape[2] == 3 else img_np[:, :, 0]
    except Exception:
        pass
    return img_np


def _annotation_penalty(crop_np) -> float:
    try:
        import numpy as np
        import cv2
        from tem2cif.utils.ocr import ocr_text

        if crop_np is None:
            return 1.0

        h, w = crop_np.shape[:2]
        area = max(1, h * w)

        text = ""
        try:
            text = ocr_text(crop_np) or ""
        except Exception:
            text = ""
        tokens = [t for t in text.split() if any(c.isalnum() for c in t)]
        ocr_density = min(1.0, len(tokens) / max(1.0, area / 50000.0))

        if crop_np.ndim == 3 and crop_np.shape[2] >= 3:
            b, g, r = crop_np[:, :, 0], crop_np[:, :, 1], crop_np[:, :, 2]
            diff = (abs(r.astype(float) - g.astype(float)) + abs(g.astype(float) - b.astype(float)))
            non_gray = (diff > 30.0).sum() / float(area)
        else:
            non_gray = 0.0

        gimg = _gray(crop_np)
        try:
            edges = cv2.Canny(gimg, 80, 160)
            lines = cv2.HoughLinesP(edges, 1, 3.14159 / 180.0, threshold=60, minLineLength=min(w, h) // 10, maxLineGap=5)
            line_cnt = 0 if lines is None else len(lines)
        except Exception:
            line_cnt = 0
        line_score = min(1.0, line_cnt / max(1.0, area / 100000.0))

        return float(max(0.0, min(1.0, (ocr_density + non_gray + line_score) / 3.0)))
    except Exception:
        return 0.5


def _lattice_strength(crop_np) -> float:
    try:
        import numpy as np
        import cv2

        if crop_np is None:
            return 0.0
        g = _gray(crop_np)
        g = g.astype("float32")
        g = (g - g.min()) / (g.max() - g.min() + 1e-6)

        F = np.fft.fftshift(np.fft.fft2(g))
        mag = np.abs(F)
        h, w = mag.shape
        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        rmax = np.sqrt((cy) ** 2 + (cx) ** 2)
        rin, rout = 0.08 * rmax, 0.45 * rmax
        band = (R >= rin) & (R <= rout)
        band_mag = mag * band

        vals = band_mag[band]
        if vals.size == 0:
            return 0.0
        mu, sd = float(vals.mean()), float(vals.std())
        contrast = sd / (mu + 1e-6)
        contrast_n = max(0.0, min(1.0, contrast / 5.0))

        thr = mu + 2.0 * sd
        peaks = (band_mag > thr).sum()
        peak_norm = max(0.0, min(1.0, peaks / max(1.0, (h * w) / 4000.0)))

        try:
            lap = cv2.Laplacian((g * 255).astype("uint8"), cv2.CV_64F)
            sharp = float(lap.var())
            sharp_n = max(0.0, min(1.0, sharp / 200.0))
        except Exception:
            sharp_n = contrast_n

        score = 0.6 * contrast_n + 0.4 * max(peak_norm, sharp_n)
        return float(max(0.0, min(1.0, score)))
    except Exception:
        return 0.0


def choose_best_image(state: S) -> S:
    panel_candidates: List[Dict[str, Any]] = state.get("panel_candidates", []) or []
    fig_images: List[Any] = state.get("fig_images", []) or []
    fig_meta: List[Dict[str, Any]] = state.get("fig_meta", []) or []

    best: Optional[Dict[str, Any]] = None
    best_score = -1e9

    for i, cand in enumerate(panel_candidates):
        fig_id = cand.get("fig_id")
        bbox = cand.get("bbox")
        page = cand.get("page")

        # Try to locate corresponding figure image; fall back by index
        img = None
        for j, meta in enumerate(fig_meta):
            if str(meta.get("fig_id", meta.get("id", j))) == str(fig_id):
                img = meta.get("image") or (fig_images[j] if j < len(fig_images) else None)
                break
        if img is None and i < len(fig_images):
            img = fig_images[i]

        crop = _crop(img, bbox)

        ann = _annotation_penalty(crop)
        lat = _lattice_strength(crop)
        quality = 0.5 * lat - 0.5 * ann

        if quality > best_score:
            best_score = quality
            best = {
                "image": crop,
                "bbox": bbox,
                "fig_id": fig_id,
                "page": page,
                "quality_score": float(max(0.0, min(1.0, (quality + 1.0) / 2.0))),
            }

    if best is None:
        img0 = fig_images[0] if fig_images else None
        crop0 = _crop(img0, None)
        best = {
            "image": crop0,
            "bbox": None,
            "fig_id": str(fig_meta[0].get("fig_id", 0)) if fig_meta else "0",
            "page": int(fig_meta[0].get("page", 0)) if fig_meta else 0,
            "quality_score": 0.0,
        }

    state["focus_bundle"] = best
    return state
