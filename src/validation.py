from __future__ import annotations
from typing import List, Tuple, Dict, Any
import numpy as np
from ase.io import read as ase_read


def _pairwise_dists_2d(points: np.ndarray, top_k: int = 2000) -> np.ndarray:
    n = points.shape[0]
    if n < 2:
        return np.zeros((0,), dtype=float)
    dists = []
    for i in range(n):
        di = np.linalg.norm(points[i + 1 :] - points[i], axis=1)
        dists.append(di)
    d = np.concatenate(dists) if dists else np.zeros((0,), dtype=float)
    d = np.sort(d)
    if d.size > top_k:
        d = d[:top_k]
    return d


def compare_image_coords_to_cif(cif_path: str, image_coords: List[Tuple[float, float]], tolerance: float = 0.15) -> Dict[str, Any]:
    """Compare 2D pairwise distance distribution from image coords to that from CIF.
    - Project CIF atoms onto the a-b plane (XY) and compute pairwise distances.
    - Compute scale factor to best match distributions (median ratio).
    - Compute RMSE on matched sorted distances.
    Returns metrics and pass/fail.
    """
    atoms = ase_read(cif_path)
    pos = atoms.get_positions()  # Nx3 in Angstroms
    xy = pos[:, :2]
    xy = xy - xy.mean(axis=0, keepdims=True)
    cif_d = _pairwise_dists_2d(xy)

    if len(image_coords) < 2:
        return {"status": "error", "reason": "Not enough image coordinates"}
    img = np.array(image_coords, dtype=float)
    img = img - img.mean(axis=0, keepdims=True)
    img_d = _pairwise_dists_2d(img)

    if cif_d.size == 0 or img_d.size == 0:
        return {"status": "error", "reason": "Empty distance set"}

    # Match by length: take min(len)) and compute scale via median ratio of top distances
    m = min(cif_d.size, img_d.size)
    cif_sel = cif_d[:m]
    img_sel = img_d[:m]
    # avoid zeros
    nz = (img_sel > 1e-8) & (cif_sel > 1e-8)
    if not np.any(nz):
        return {"status": "error", "reason": "No non-zero distances"}
    scale = np.median(cif_sel[nz] / img_sel[nz])
    img_scaled = img_sel * scale
    rmse = float(np.sqrt(np.mean((cif_sel - img_scaled) ** 2)))
    mae = float(np.mean(np.abs(cif_sel - img_scaled)))
    rel = float(mae / (np.mean(cif_sel) + 1e-8))
    passed = rel <= tolerance

    return {
        "status": "ok",
        "rmse": rmse,
        "mae": mae,
        "relative_error": rel,
        "scale": float(scale),
        "n_pairs": int(m),
        "pass": passed,
    }

