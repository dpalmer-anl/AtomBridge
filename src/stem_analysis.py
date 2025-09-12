from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image


def _load_gray(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    return np.array(img)


def measure_lattice_vectors(image_path: str, pixel_to_nm: float) -> Dict:
    """Estimate two in-plane lattice vectors from a grayscale STEM/TEM crop.
    Returns dict with a_nm, b_nm, angle_deg, n_atoms, overlay_path.
    pixel_to_nm: nm per pixel (e.g., 0.02 means 50 px per nm)
    """
    import cv2  # type: ignore
    from scipy.spatial import KDTree  # type: ignore
    from sklearn.cluster import DBSCAN  # type: ignore
    from skimage.feature import peak_local_max  # type: ignore

    img = _load_gray(image_path)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img)

    coords = peak_local_max(enhanced, min_distance=5, threshold_rel=0.6, exclude_border=True)
    if len(coords) < 10:
        raise RuntimeError(f"Too few detected atoms: {len(coords)}")

    # coords in (row, col) -> (y, x). Switch to (x, y)
    pts = coords[:, ::-1].astype(float)

    tree = KDTree(pts)
    k = min(7, len(pts))
    distances, indices = tree.query(pts, k=k)

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

