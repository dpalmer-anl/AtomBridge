from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image


def _load_gray(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    return np.array(img)


def _fft_reciprocal_vectors(img: np.ndarray, top_k: int = 20):
    """Return list of reciprocal vectors (fx, fy) in cycles/pixel using FFT peaks."""
    import cv2  # type: ignore
    H, W = img.shape[:2]
    F = np.fft.fftshift(np.fft.fft2(img.astype(np.float32)))
    P = np.log1p(np.abs(F))
    cx, cy = W // 2, H // 2
    Y, X = np.ogrid[:H, :W]
    r0 = max(3, int(0.02 * min(H, W)))
    mask_center = (X - cx) ** 2 + (Y - cy) ** 2 <= r0 * r0
    P = P.copy()
    P[mask_center] = 0.0
    # Normalize to uint8
    Pn = P - P.min()
    if Pn.max() > 0:
        Pn = (Pn / Pn.max()) * 255.0
    P8 = Pn.astype(np.uint8)
    P8 = cv2.GaussianBlur(P8, (5, 5), 0)
    pts = cv2.goodFeaturesToTrack(P8, maxCorners=500, qualityLevel=0.01, minDistance=5)
    cand = []  # (power, fx, fy)
    if pts is not None:
        for p in pts:
            x, y = float(p[0][0]), float(p[0][1])
            if (x - cx) ** 2 + (y - cy) ** 2 <= (r0 * 1.5) ** 2:
                continue
            fx = (x - cx) / float(W)
            fy = (y - cy) / float(H)
            f = (fx * fx + fy * fy) ** 0.5
            if f < 1e-3:
                continue
            xi, yi = int(round(x)), int(round(y))
            x0, x1 = max(0, xi - 2), min(W, xi + 3)
            y0, y1 = max(0, yi - 2), min(H, yi + 3)
            power = float(P[y0:y1, x0:x1].sum())
            cand.append((power, fx, fy))
    cand.sort(key=lambda t: -t[0])
    uniq = []
    for _, fx, fy in cand:
        if fx < 0:
            fx, fy = -fx, -fy
        if not any(abs(fx - ux) < 1e-4 and abs(fy - uy) < 1e-4 for (ux, uy) in uniq):
            uniq.append((fx, fy))
        if len(uniq) >= top_k:
            break
    return [np.array([fx, fy], dtype=float) for fx, fy in uniq]


def measure_lattice_vectors(image_path: str, pixel_to_nm: float) -> Dict:
    """Estimate two in-plane lattice vectors from a grayscale STEM/TEM crop.
    Returns dict with a_nm, b_nm, angle_deg, n_atoms, overlay_path.
    pixel_to_nm: nm per pixel (e.g., 0.02 means 50 px per nm)
    """
    import cv2  # type: ignore
    from sklearn.cluster import DBSCAN  # type: ignore
    # Peak detection: prefer skimage, else fallback to OpenCV or morphology
    try:
        from skimage.feature import peak_local_max  # type: ignore
    except Exception:
        peak_local_max = None  # type: ignore

    img = _load_gray(image_path)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    # Detect bright atom centers
    if peak_local_max is not None:
        coords = peak_local_max(enhanced, min_distance=5, threshold_rel=0.6, exclude_border=True)
    else:
        # Fallback 1: Shi-Tomasi corners as proxies for bright spots
        pts = cv2.goodFeaturesToTrack(enhanced, maxCorners=5000, qualityLevel=0.01, minDistance=5)
        if pts is not None and len(pts) > 0:
            coords = np.array([[int(p[0][1]), int(p[0][0])] for p in pts], dtype=int)
        else:
            # Fallback 2: morphological local maxima above threshold
            thr = np.percentile(enhanced, 95)
            kernel = np.ones((3, 3), np.uint8)
            dil = cv2.dilate(enhanced, kernel)
            mask = (enhanced == dil) & (enhanced >= thr)
            ys, xs = np.where(mask)
            coords = np.stack([ys, xs], axis=1) if ys.size else np.empty((0, 2), dtype=int)
    if len(coords) < 10:
        raise RuntimeError(f"Too few detected atoms: {len(coords)}")

    # coords in (row, col) -> (y, x). Switch to (x, y)
    pts = coords[:, ::-1].astype(float)

    # Nearest neighbors without SciPy: use sklearn NearestNeighbors
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    k = min(7, len(pts))
    nbrs = NearestNeighbors(n_neighbors=k, algorithm="kd_tree").fit(pts)
    distances, indices = nbrs.kneighbors(pts)

    neighbor_vectors: List[np.ndarray] = []
    all_nn = distances[:, 1]
    for i, neigh in enumerate(indices):
        for j in neigh[1:]:
            v = pts[j] - pts[i]
            neighbor_vectors.append(v)
    V = np.array(neighbor_vectors)
    if V.size == 0:
        raise RuntimeError("No neighbor vectors formed.")

    median_nn = float(np.median(all_nn))
    eps = max(1.0, median_nn * 0.4)
    min_samples = max(5, int(len(pts) * 0.25))
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(V)
    labels = db.labels_
    uniq = [u for u in set(labels) if u != -1]
    if len(uniq) < 2:
        # FFT fallback
        recips = _fft_reciprocal_vectors(img)
        if len(recips) >= 2:
            r1 = recips[0]
            r2 = None
            for g in recips[1:]:
                cosang = float(np.dot(r1, g) / (np.linalg.norm(r1) * np.linalg.norm(g)))
                if abs(cosang) < 0.95:
                    r2 = g
                    break
            if r2 is None:
                raise RuntimeError("Could not find two primary lattice directions.")
            f1 = float(np.linalg.norm(r1))
            f2 = float(np.linalg.norm(r2))
            len_a_nm = (1.0 / f1) * pixel_to_nm
            len_b_nm = (1.0 / f2) * pixel_to_nm
            ang_star = float(np.degrees(np.arccos(np.clip(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)), -1.0, 1.0))))
            angle = 180.0 - ang_star
            # Overlay using reciprocal vectors for visualization
            overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            cx, cy = (img.shape[1] // 2, img.shape[0] // 2)
            scale_draw = min(img.shape[:2]) * 0.5
            p1 = (int(cx + r1[0] * scale_draw), int(cy + r1[1] * scale_draw))
            p2 = (int(cx + r2[0] * scale_draw), int(cy + r2[1] * scale_draw))
            cv2.arrowedLine(overlay, (cx, cy), p1, (0, 200, 0), 2, line_type=cv2.LINE_AA)
            cv2.arrowedLine(overlay, (cx, cy), p2, (0, 255, 255), 2, line_type=cv2.LINE_AA)
            out_path = str(Path(image_path).with_suffix("") ) + "_lattice_fft.png"
            cv2.imwrite(out_path, overlay)
            return {
                "a_nm": float(len_a_nm),
                "b_nm": float(len_b_nm),
                "gamma_deg": float(angle),
                "n_atoms": int(len(pts)),
                "overlay_path": out_path,
            }
        else:
            raise RuntimeError("Could not find two primary lattice directions.")

    centers = []
    for u in uniq:
        m = V[labels == u].mean(axis=0)
        centers.append(m)
    centers.sort(key=lambda x: np.linalg.norm(x))
    a = centers[0]
    b = None
    for c in centers[1:]:
        cos = float(np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c)))
        if abs(cos) < 0.95:
            b = c
            break
    if b is None:
        # As another fallback, use FFT if DBSCAN produced collinear directions
        recips = _fft_reciprocal_vectors(img)
        if len(recips) >= 2:
            r1 = recips[0]
            r2 = None
            for g in recips[1:]:
                cosang = float(np.dot(r1, g) / (np.linalg.norm(r1) * np.linalg.norm(g)))
                if abs(cosang) < 0.95:
                    r2 = g
                    break
            if r2 is not None:
                f1 = float(np.linalg.norm(r1))
                f2 = float(np.linalg.norm(r2))
                a_nm = (1.0 / f1) * pixel_to_nm
                b_nm = (1.0 / f2) * pixel_to_nm
                ang_star = float(np.degrees(np.arccos(np.clip(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)), -1.0, 1.0))))
                angle = 180.0 - ang_star
                overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                cx, cy = (img.shape[1] // 2, img.shape[0] // 2)
                scale_draw = min(img.shape[:2]) * 0.5
                p1 = (int(cx + r1[0] * scale_draw), int(cy + r1[1] * scale_draw))
                p2 = (int(cx + r2[0] * scale_draw), int(cy + r2[1] * scale_draw))
                cv2.arrowedLine(overlay, (cx, cy), p1, (0, 200, 0), 2, line_type=cv2.LINE_AA)
                cv2.arrowedLine(overlay, (cx, cy), p2, (0, 255, 255), 2, line_type=cv2.LINE_AA)
                out_path = str(Path(image_path).with_suffix("") ) + "_lattice_fft.png"
                cv2.imwrite(out_path, overlay)
                return {
                    "a_nm": float(a_nm),
                    "b_nm": float(b_nm),
                    "gamma_deg": float(angle),
                    "n_atoms": int(len(pts)),
                    "overlay_path": out_path,
                }
        raise RuntimeError("Second non-collinear vector not found.")

    a_nm = np.linalg.norm(a) * pixel_to_nm
    b_nm = np.linalg.norm(b) * pixel_to_nm
    angle = float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1.0, 1.0))))

    # overlay visualization
    overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for x, y in pts:
        cv2.circle(overlay, (int(x), int(y)), 3, (255, 0, 0), 1, lineType=cv2.LINE_AA)
    center_idx = int(np.argmin(np.linalg.norm(pts - pts.mean(axis=0), axis=1)))
    sp = tuple(pts[center_idx].astype(int))
    cv2.arrowedLine(overlay, sp, tuple((pts[center_idx] + a).astype(int)), (0, 255, 0), 2, line_type=cv2.LINE_AA)
    cv2.arrowedLine(overlay, sp, tuple((pts[center_idx] + b).astype(int)), (0, 255, 255), 2, line_type=cv2.LINE_AA)
    out_path = str(Path(image_path).with_suffix("") ) + "_lattice.png"
    cv2.imwrite(out_path, overlay)

    return {
        "a_nm": float(a_nm),
        "b_nm": float(b_nm),
        "gamma_deg": float(angle),
        "n_atoms": int(len(pts)),
        "overlay_path": out_path,
    }


def minimal_cif_from_lattice(a_nm: float, b_nm: float, gamma_deg: float, out_path: str) -> str:
    """Create a minimal CIF with a 2D lattice in the ab-plane and a placeholder atom.
    c is set to 10 Å. A single 'X' atom is placed at 0,0,c/2 to allow a valid CIF.
    """
    from ase import Atoms  # type: ignore
    from ase.io import write as ase_write  # type: ignore

    a = a_nm * 10.0
    b = b_nm * 10.0
    c = 10.0
    gamma = gamma_deg
    # Build cell from lengths+angles (alpha=beta=90, gamma provided)
    # Use 2D cell with pbc in a,b
    atoms = Atoms(symbols="X", positions=[[0.0, 0.0, c / 2.0]], pbc=[True, True, False])
    # Construct cell matrix from lengths and gamma
    import numpy as np
    gamma_rad = np.deg2rad(gamma)
    cell = np.array([[a, 0, 0], [b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0], [0, 0, c]])
    atoms.set_cell(cell)
    atoms.set_pbc([True, True, False])
    ase_write(out_path, atoms, format="cif")
    return out_path
