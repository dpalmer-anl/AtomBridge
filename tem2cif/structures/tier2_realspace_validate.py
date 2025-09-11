from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tem2cif.state import S, CIFScore


def _read_atoms(path: str):
    try:
        from ase.io import read

        return read(path)
    except Exception:
        return None


def _to_gray_float(img: Any):
    try:
        import numpy as np
        import cv2

        if img is None:
            return None
        if isinstance(img, np.ndarray):
            arr = img
        else:
            try:
                from PIL import Image

                if isinstance(img, Image.Image):
                    import numpy as np

                    arr = np.array(img)
                else:
                    return None
            except Exception:
                return None
        if arr.ndim == 3 and arr.shape[2] >= 3:
            g = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        elif arr.ndim == 2:
            g = arr
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


def _bandpass_realspace(g, lo: float = 0.01, hi: float = 0.25):
    try:
        import numpy as np

        if g is None:
            return None
        h, w = g.shape[:2]
        G = np.fft.fftshift(np.fft.fft2(g))
        Y, X = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2
        R = ((Y - cy) ** 2 + (X - cx) ** 2) ** 0.5
        rmax = (cy**2 + cx**2) ** 0.5
        mask = (R >= lo * rmax) & (R <= hi * rmax)
        Gf = G * mask
        gf = np.fft.ifft2(np.fft.ifftshift(Gf)).real
        gf -= gf.min()
        rng = gf.max() - gf.min()
        if rng > 0:
            gf /= rng
        return gf.astype("float32")
    except Exception:
        return g


def _ssim(a, b) -> float:
    try:
        from skimage.metrics import structural_similarity as ssim
        import numpy as np

        if a is None or b is None:
            return 0.0
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        if h <= 8 or w <= 8:
            return 0.0
        a = a[:h, :w]
        b = b[:h, :w]
        return float(ssim(a, b, data_range=1.0))
    except Exception:
        # Fallback: normalized cross-correlation
        try:
            import numpy as np

            if a is None or b is None:
                return 0.0
            h = min(a.shape[0], b.shape[0])
            w = min(a.shape[1], b.shape[1])
            if h <= 8 or w <= 8:
                return 0.0
            a = a[:h, :w]
            b = b[:h, :w]
            a = (a - a.mean()) / (a.std() + 1e-6)
            b = (b - b.mean()) / (b.std() + 1e-6)
            return float((a * b).mean())
        except Exception:
            return 0.0


def _fringe_error(exp_img, sim_img) -> float:
    """Compare fringe spacing via FFT peak radii in normalized units."""
    try:
        import numpy as np

        def peak_radii(g):
            if g is None:
                return []
            h, w = g.shape[:2]
            F = np.fft.fftshift(np.fft.fft2(g))
            mag = np.abs(F)
            cy, cx = h // 2, w // 2
            Y, X = np.ogrid[:h, :w]
            R = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
            rmax = np.sqrt(cy**2 + cx**2)
            band = (R >= 0.05 * rmax) & (R <= 0.5 * rmax)
            vals = mag * band
            thr = vals[band].mean() + 2.5 * vals[band].std()
            yx = np.where(vals > thr)
            rs = R[yx] / (rmax + 1e-6)
            rs.sort()
            return rs[:50]

        re = peak_radii(exp_img)
        rs = peak_radii(sim_img)
        if len(re) == 0 or len(rs) == 0:
            return float("inf")
        m = min(len(re), len(rs))
        re = re[:m]
        rs = rs[:m]
        return float(np.mean(np.abs(re - rs)))
    except Exception:
        return float("inf")


def _simulate_hrtem_abtem(atoms, thickness_nm: float, defocus_nm: float):  # pragma: no cover - optional
    """Attempt an abTEM real-space HRTEM simulation. Returns image or None."""
    try:
        # Placeholder: return None to gracefully skip when abTEM is unavailable.
        # To enable, integrate abTEM TEMSimulator with given thickness/defocus here.
        return None
    except Exception:
        return None


def tier2_realspace_validate(state: S) -> S:
    # Experimental real-space image
    exp_img = (state.get("focus_bundle", {}) or {}).get("image")
    exp_g = _to_gray_float(exp_img)
    exp_bp = _bandpass_realspace(exp_g)

    updated: List[CIFScore] = []
    for sc in state.get("fft_scored_cifs", []) or []:
        sc = dict(sc)
        passed_t1 = bool((sc.get("tier1", {}) or {}).get("pass"))
        if not passed_t1:
            # propagate without tier2
            updated.append(sc)
            continue

        atoms = _read_atoms(sc.get("path", ""))
        best = {"ssim": 0.0, "fringe_err": float("inf"), "thickness_nm": None, "defocus_nm": None}
        for thickness_nm in [2, 4, 6, 8]:
            for defocus_nm in [-20, -10, 0, 10, 20]:
                sim = _simulate_hrtem_abtem(atoms, thickness_nm, defocus_nm)
                if sim is None:
                    continue
                sim_g = _to_gray_float(sim)
                sim_bp = _bandpass_realspace(sim_g)
                ssim_val = _ssim(exp_bp, sim_bp)
                fe = _fringe_error(exp_bp, sim_bp)
                if ssim_val > best["ssim"]:
                    best = {
                        "ssim": float(ssim_val),
                        "fringe_err": float(fe),
                        "thickness_nm": float(thickness_nm),
                        "defocus_nm": float(defocus_nm),
                    }

        # If no simulation available, keep conservative defaults
        t2 = dict(sc.get("tier2", {}))
        if best["thickness_nm"] is None:
            t2.update({"ssim": 0.5, "fringe_err": float("inf")})
        else:
            t2.update(best)
        sc["tier2"] = t2
        updated.append(sc)

    state["rs_scored_cifs"] = updated
    return state
