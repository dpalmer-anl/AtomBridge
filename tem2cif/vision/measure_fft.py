from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import math

from tem2cif.state import S, Peak, ImageMetrics


def _to_gray_float(img: Any):
    try:
        import numpy as np
        from PIL import Image
        import cv2

        if isinstance(img, Image.Image):
            img = np.array(img)
        if isinstance(img, np.ndarray):
            arr = img
        else:
            return None
        if arr.ndim == 2:
            g = arr
        elif arr.ndim == 3:
            # Heuristic: assume BGR if OpenCV read, else RGB; convert robustly
            # Use cv2 to convert safely
            try:
                g = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            except Exception:
                g = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        else:
            return None
        g = g.astype("float32")
        g -= g.min()
        rng = g.max() - g.min()
        if rng > 0:
            g /= rng
        return g
    except Exception:
        return None


def _bilateral_or_gaussian(g):
    try:
        import cv2

        return cv2.bilateralFilter((g * 255).astype("uint8"), d=7, sigmaColor=50, sigmaSpace=7).astype(
            "float32"
        ) / 255.0
    except Exception:
        try:
            import cv2

            return cv2.GaussianBlur(g, (5, 5), 0)
        except Exception:
            return g


def _hanning_window(h: int, w: int):
    try:
        import numpy as np

        wy = np.hanning(h)
        wx = np.hanning(w)
        return wy[:, None] * wx[None, :]
    except Exception:
        return None


def _fft_mag(g):
    try:
        import numpy as np

        F = np.fft.fftshift(np.fft.fft2(g))
        mag = np.abs(F)
        return mag
    except Exception:
        return None


def _band_mask(h: int, w: int, rin_frac: float = 0.06, rout_frac: float = 0.48):
    try:
        import numpy as np

        cy, cx = h // 2, w // 2
        Y, X = np.ogrid[:h, :w]
        R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
        rmax = float(max(cy, cx))
        rin, rout = rin_frac * rmax, rout_frac * rmax
        mask = (R >= rin) & (R <= rout)
        return mask, R
    except Exception:
        return None, None


def _local_maxima_peaks(mag, mask, max_peaks: int = 32) -> List[Tuple[float, float, float]]:
    """Return list of peaks as (y, x, val)."""
    try:
        import numpy as np
        import cv2

        band = mag.copy()
        if mask is not None:
            band[~mask] = 0

        # Smooth to reduce noise
        try:
            band = cv2.GaussianBlur(band.astype("float32"), (3, 3), 0)
        except Exception:
            pass

        vals = band[band > 0]
        mu = float(vals.mean()) if vals.size else float(band.mean())
        sd = float(vals.std()) if vals.size else float(band.std())
        thr = mu + 2.5 * sd

        # Local maxima via dilation
        try:
            kernel = np.ones((3, 3), np.uint8)
            dil = cv2.dilate(band, kernel)
            maxima = (band == dil) & (band > thr)
        except Exception:
            maxima = (band > thr)

        ys, xs = np.where(maxima)
        peak_vals = band[ys, xs]
        order = np.argsort(-peak_vals)
        peaks = []
        for idx in order[:max_peaks]:
            y, x = int(ys[idx]), int(xs[idx])
            v = float(peak_vals[idx])
            # Subpixel centroid in 5x5 window
            y0, y1 = max(0, y - 2), min(band.shape[0], y + 3)
            x0, x1 = max(0, x - 2), min(band.shape[1], x + 3)
            win = band[y0:y1, x0:x1]
            if win.size:
                Y, X = np.mgrid[y0:y1, x0:x1]
                total = win.sum() + 1e-6
                yc = float((Y * win).sum() / total)
                xc = float((X * win).sum() / total)
            else:
                yc, xc = float(y), float(x)
            peaks.append((yc, xc, v))
        return peaks
    except Exception:
        return []


def _peaks_to_polar(peaks: List[Tuple[float, float, float]], shape) -> List[Tuple[float, float, float]]:
    import math

    h, w = shape
    cy, cx = h / 2.0, w / 2.0
    out = []
    for y, x, v in peaks:
        dy, dx = y - cy, x - cx
        r = math.hypot(dy, dx)
        ang = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 180.0  # map to [0,180)
        out.append((r, ang, v))
    return out


def _angle_clusters(angles: List[float]) -> List[str]:
    """Very light clustering into major orientations; returns label strings."""
    if not angles:
        return []
    try:
        import numpy as np

        bins = np.linspace(0, 180, 10, endpoint=False)
        hist, edges = np.histogram(angles, bins=bins)
        top = hist.argsort()[::-1][:2]
        labels = []
        for idx in top:
            center = (edges[idx] + edges[idx + 1]) / 2.0
            labels.append(f"angle~{int(round(center))}°")
        return labels
    except Exception:
        return [f"angle~{int(round(sum(angles)/len(angles)))}°"]


def measure_fft(state: S) -> S:
    fb = state.get("focus_bundle", {}) or {}
    img = fb.get("image")
    bbox = fb.get("bbox")
    fig_id = fb.get("fig_id")
    quality = fb.get("quality_score")

    g = _to_gray_float(img)
    if g is None:
        state["image_metrics"] = {
            "scale_A_per_px": 0.0,
            "zone_axis_candidates": [],
            "peaks": [],
            "uncertainties": {"d_A": 0.0, "theta_deg": 0.0},
            "provenance": {"fig": fig_id, "panel_bbox": bbox, "quality_score": quality},
        }
        return state

    # denoise
    g = _bilateral_or_gaussian(g)

    # window
    h, w = g.shape[:2]
    win = _hanning_window(h, w)
    if win is not None:
        g = g * win

    # FFT magnitude and band-pass
    mag = _fft_mag(g)
    mask, R = _band_mask(h, w)
    if mag is None or mask is None or R is None:
        state["image_metrics"] = {
            "scale_A_per_px": 0.0,
            "zone_axis_candidates": [],
            "peaks": [],
            "uncertainties": {"d_A": 0.0, "theta_deg": 0.0},
            "provenance": {"fig": fig_id, "panel_bbox": bbox, "quality_score": quality},
        }
        return state

    # Peak picking
    peaks_yxv = _local_maxima_peaks(mag, mask, max_peaks=24)
    polar = _peaks_to_polar(peaks_yxv, mag.shape)

    # Convert to d_A using optional scale; attempt to fetch scale from focus_bundle or fig_meta
    scale = 0.0
    # from state if already detected previously
    if isinstance(state.get("image_metrics", {}).get("scale_A_per_px"), (int, float)):
        try:
            scale = float(state.get("image_metrics", {}).get("scale_A_per_px", 0.0))
        except Exception:
            scale = 0.0
    # possibly from focus bundle/meta
    if not scale:
        try:
            scale = float(fb.get("scale_A_per_px", 0.0))
        except Exception:
            scale = 0.0

    Rmax = float(max(h, w))
    peaks: List[Peak] = []
    for r, ang, v in polar:
        dA = (scale * Rmax / r) if (scale and r > 0) else 0.0
        peaks.append({"d_A": float(dA), "theta_deg": float(ang), "intensity": float(v)})

    # Sort peaks by intensity descending
    peaks.sort(key=lambda p: p["intensity"], reverse=True)

    # Angle clustering for zone-axis candidates (placeholder labels)
    zone_axis_candidates = _angle_clusters([p["theta_deg"] for p in peaks[:12]])

    # Uncertainties: heuristic
    uncertainties = {"d_A": 0.05 * (peaks[0]["d_A"] if peaks else 0.0), "theta_deg": 2.0}

    metrics: ImageMetrics = {
        "scale_A_per_px": float(scale),
        "zone_axis_candidates": zone_axis_candidates,
        "peaks": peaks,
        "uncertainties": uncertainties,
        "provenance": {"fig": fig_id, "panel_bbox": bbox, "quality_score": quality},
    }

    state["image_metrics"] = metrics
    return state
