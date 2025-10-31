from __future__ import annotations

from pathlib import Path
from ase.io import read

from src.validation import compare_cif_lattices, compare_image_coords_to_cif


ROOT = Path(__file__).resolve().parents[1]


def test_compare_image_coords_to_cif_passes_with_scaled_coords():
    cif_path = ROOT / "LiCoO2.cif"
    atoms = read(cif_path)
    xy = atoms.get_positions()[:, :2]
    xy = xy - xy.mean(axis=0, keepdims=True)
    coords = (xy * 1.7).tolist()

    res = compare_image_coords_to_cif(str(cif_path), coords, tolerance=0.2)

    assert res["status"] == "ok"
    assert res["pass"] is True
    assert res["relative_error"] < 1e-8
    assert res["n_pairs"] > 0


def test_compare_image_coords_to_cif_errors_on_insufficient_points():
    cif_path = ROOT / "LiCoO2.cif"
    res = compare_image_coords_to_cif(str(cif_path), [(0.0, 0.0)])
    assert res["status"] == "error"
    assert "Not enough" in res["reason"]


def test_compare_cif_lattices_with_similar_structures_passes():
    cif_a = ROOT / "LiCoO2.cif"
    cif_b = ROOT / "LiCoO2_R-3m_pristine_1x1x1.cif"
    res = compare_cif_lattices(str(cif_a), str(cif_b), tol_len=0.05, tol_ang=1.0)
    assert res["status"] == "ok"
    assert res["pass"] is True
