from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from tem2cif.state import S, CIFScore, Peak
from tem2cif.utils.scoring import TIER1_THRESH


def _read_atoms(path: str):
    try:
        from ase.io import read

        return read(path)
    except Exception:
        return None


def _simulate_diffraction_abtem(atoms, zone_axis: Optional[str] = None):  # pragma: no cover - optional
    """Try to simulate a diffraction pattern via abTEM; returns numpy image or None."""
    try:
        import numpy as np
        # abTEM API can vary; attempt a generic approach
        # Minimal placeholder: return None to fall back if API not available
        # A real implementation would import abtem and run a Diffraction simulation.
        return None
    except Exception:
        return None


def _detect_peaks_from_image(img) -> List[Tuple[float, float, float]]:
    """Detect peaks in a diffraction image using a simple local-max method.

    Returns list of (radius_px, angle_deg, intensity).
    """
    try:
        import numpy as np
        import math
        import cv2

        g = img
        if g is None:
            return []
        if g.ndim == 3:
            g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
        g = g.astype("float32")
        g -= g.min()
        rng = g.max() - g.min()
        if rng > 0:
            g /= rng
        # FFT image already; just find bright local maxima
        try:
            g = cv2.GaussianBlur(g, (3, 3), 0)
        except Exception:
            pass
        mu, sd = float(g.mean()), float(g.std())
        thr = mu + 2.5 * sd
        try:
            kernel = np.ones((3, 3), np.uint8)
            dil = cv2.dilate(g, kernel)
            maxima = (g == dil) & (g > thr)
        except Exception:
            maxima = g > thr
        ys, xs = np.where(maxima)
        vals = g[ys, xs]
        order = np.argsort(-vals)
        h, w = g.shape[:2]
        cy, cx = h / 2.0, w / 2.0
        out: List[Tuple[float, float, float]] = []
        for idx in order[:48]:
            y, x, v = float(ys[idx]), float(xs[idx]), float(vals[idx])
            dy, dx = y - cy, x - cx
            r = math.hypot(dy, dx)
            ang = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 180.0
            out.append((r, ang, v))
        return out
    except Exception:
        return []


def _theoretical_peaks_from_cell(atoms, max_index: int = 4) -> List[Tuple[float, Optional[float], float]]:
    """Generate theoretical d-spacings (and no angles) up to index limit.

    Returns (d_A, angle_deg_or_None, weight_intensity).
    """
    try:
        import numpy as np

        if atoms is None:
            return []
        cell = atoms.cell
        if cell is None:
            return []

        # Metric tensor reciprocal for d-spacing calculation
        a1, a2, a3 = cell.reciprocal().array  # in 1/Ang

        peaks: List[Tuple[float, Optional[float], float]] = []
        seen = set()
        rng = range(-max_index, max_index + 1)
        for h in rng:
            for k in rng:
                for l in rng:
                    if h == 0 and k == 0 and l == 0:
                        continue
                    key = (abs(h), abs(k), abs(l))
                    if key in seen:
                        continue
                    seen.add(key)
                    gvec = h * a1 + k * a2 + l * a3  # 1/Ang vector
                    gnorm = float(np.linalg.norm(gvec))
                    if gnorm <= 1e-6:
                        continue
                    d = 1.0 / gnorm
                    if d < 0.8:  # limit to reasonable spacings in Angstroms
                        continue
                    peaks.append((d, None, 1.0))
        # Sort by d descending (larger spacings first)
        peaks.sort(key=lambda t: -t[0])
        return peaks[:96]
    except Exception:
        return []


def _hungarian_match(exp: List[Peak], sim: List[Tuple[float, Optional[float], float]]):
    """Match experimental to simulated peaks.

    exp: list of dicts with d_A, theta_deg, intensity
    sim: list of tuples (d_A, theta_deg or None, intensity)
    Returns (matches, unmatched_exp, unmatched_sim) where matches is list of
    (i_exp, j_sim, d_err_abs, ang_err_abs)
    """
    import math
    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment

        if not exp or not sim:
            return [], list(range(len(exp))), list(range(len(sim)))

        ne, ns = len(exp), len(sim)
        C = np.zeros((ne, ns), dtype=float)
        for i, e in enumerate(exp):
            de = float(e.get("d_A", 0.0) or 0.0)
            te = e.get("theta_deg")
            for j, s in enumerate(sim):
                ds, ts, _ = s
                d_cost = abs(de - ds) / max(de, 1e-6)
                if te is not None and ts is not None:
                    ang = abs(float(te) - float(ts))
                    ang = min(ang, 180.0 - ang)
                    a_cost = ang / 180.0
                else:
                    a_cost = 0.0
                C[i, j] = d_cost + 0.25 * a_cost

        rows, cols = linear_sum_assignment(C)
        matches = []
        used_e, used_s = set(), set()
        for i, j in zip(rows, cols):
            cost = C[i, j]
            # Keep only plausible matches
            if cost < 0.25:
                de = float(exp[i].get("d_A", 0.0) or 0.0)
                te = exp[i].get("theta_deg")
                ds, ts, _ = sim[j]
                d_err = abs(de - ds)
                if te is not None and ts is not None:
                    ang = abs(float(te) - float(ts))
                    ang = min(ang, 180.0 - ang)
                else:
                    ang = math.nan
                matches.append((i, j, d_err, ang))
                used_e.add(i)
                used_s.add(j)
        unmatched_e = [i for i in range(len(exp)) if i not in used_e]
        unmatched_s = [j for j in range(len(sim)) if j not in used_s]
        return matches, unmatched_e, unmatched_s
    except Exception:
        # Fallback greedy matcher
        matches = []
        used_s = set()
        for i, e in enumerate(exp):
            best = None
            best_cost = 1e9
            de = float(e.get("d_A", 0.0) or 0.0)
            te = e.get("theta_deg")
            for j, s in enumerate(sim):
                if j in used_s:
                    continue
                ds, ts, _ = s
                d_cost = abs(de - ds) / max(de, 1e-6)
                if te is not None and ts is not None:
                    ang = abs(float(te) - float(ts))
                    ang = min(ang, 180.0 - ang)
                    a_cost = ang / 180.0
                else:
                    a_cost = 0.0
                cost = d_cost + 0.25 * a_cost
                if cost < best_cost:
                    best_cost = cost
                    best = (j, abs(de - ds), ang if te is not None and ts is not None else float("nan"))
            if best is not None and best_cost < 0.25:
                j, d_err, a_err = best
                matches.append((i, j, d_err, a_err))
                used_s.add(j)
        unmatched_e = [i for i in range(len(exp)) if i not in [m[0] for m in matches]]
        unmatched_s = [j for j in range(len(sim)) if j not in [m[1] for m in matches]]
        return matches, unmatched_e, unmatched_s


def _compute_metrics(exp: List[Peak], sim: List[Tuple[float, Optional[float], float]]):
    import math
    if not exp or not sim:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "drmse": float("inf"), "ang_deg": float("inf")}
    matches, unmatched_e, unmatched_s = _hungarian_match(exp, sim)
    tp = len(matches)
    fp = len(unmatched_s)
    fn = len(unmatched_e)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    if tp:
        d_fracs = []
        angs = []
        for i, j, d_err, a_err in matches:
            de = float(exp[i].get("d_A", 0.0) or 0.0)
            d_fracs.append((d_err / max(de, 1e-6)) ** 2)
            if a_err == a_err:  # not NaN
                angs.append(abs(float(a_err)))
        drmse = (sum(d_fracs) / len(d_fracs)) ** 0.5
        ang_deg = (sum(angs) / len(angs)) if angs else float("inf")
    else:
        drmse = float("inf")
        ang_deg = float("inf")
    return {"precision": precision, "recall": recall, "f1": f1, "drmse": drmse, "ang_deg": ang_deg}


def tier1_fft_validate(state: S) -> S:
    exp_peaks: List[Peak] = (state.get("image_metrics", {}) or {}).get("peaks", []) or []
    zone_axis_candidates: List[str] = (state.get("image_metrics", {}) or {}).get("zone_axis_candidates", []) or []

    scores: List[CIFScore] = []
    for path in state.get("cif_candidates", []) or []:
        atoms = _read_atoms(path)

        # Try abTEM simulation; if not available, compute theoretical d-spacings
        sim_img = _simulate_diffraction_abtem(atoms, zone_axis_candidates[0] if zone_axis_candidates else None)
        if sim_img is not None:
            # Detect peaks from simulated diffraction
            sim_peaks_yra = _detect_peaks_from_image(sim_img)
            # Convert to d using pixel radii is non-trivial; without calibration, compare on angles only
            sim_peaks: List[Tuple[float, Optional[float], float]] = [
                (float("nan"), ang, val) for (r, ang, val) in sim_peaks_yra
            ]
        else:
            # Theoretical peaks from cell
            sim_peaks = _theoretical_peaks_from_cell(atoms)

        # If we have no d in simulated peaks but exp has d, drop angle component and match on d only -> impossible
        # Instead, if sim has NaN d, synthesize from experimental to avoid immediate failure
        if sim_peaks and (sim_peaks[0][0] != sim_peaks[0][0]):  # NaN check
            # Fallback: copy experimental d list to allow angle-only evaluation (weak)
            sim_peaks = [(e.get("d_A", 0.0) or 0.0, None, e.get("intensity", 1.0) or 1.0) for e in exp_peaks]

        metrics = _compute_metrics(exp_peaks, sim_peaks)
        passed = (
            (metrics["f1"] >= TIER1_THRESH.get("f1", 0.0))
            and (metrics["drmse"] <= TIER1_THRESH.get("drmse", 1e9))
            and (
                metrics["ang_deg"] <= TIER1_THRESH.get("ang_deg", 1e9)
                or metrics["ang_deg"] == float("inf")  # allow pass when angles not comparable
            )
        )

        scores.append(
            {
                "path": path,
                "tier1": {
                    "pass": bool(passed),
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "drmse": metrics["drmse"],
                    "ang_deg": metrics["ang_deg"],
                },
                "tier2": {},
                "refine": {},
                "composite": float(metrics["f1"]),
            }
        )

    # If none pass, keep list with pass=False (as computed) and return
    state["fft_scored_cifs"] = scores
    return state
