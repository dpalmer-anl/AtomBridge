from __future__ import annotations

import os
from itertools import product
from typing import Any, Dict, List, Tuple

from tem2cif.state import S, DescriptionJSON


def _ensure_out_dir(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
    except Exception:
        pass
    return path


def _build_atoms_from_description(desc: DescriptionJSON):
    """Construct an ASE Atoms from description; degrade gracefully if ASE missing.

    Returns (atoms_or_none, cellpar_tuple) where cellpar is (a,b,c,alpha,beta,gamma).
    """
    cell = desc.get("cell", {}) or {}
    a = float(cell.get("a", 3.5) or 3.5)
    b = float(cell.get("b", a) or a)
    c = float(cell.get("c", a) or a)
    alpha = float(cell.get("alpha", 90.0) or 90.0)
    beta = float(cell.get("beta", 90.0) or 90.0)
    gamma = float(cell.get("gamma", 90.0) or 90.0)
    cellpar = (a, b, c, alpha, beta, gamma)

    try:
        from ase import Atoms
        from ase.cell import Cell
        from ase.data import chemical_symbols
        from ase.formula import Formula

        # Expand species list from formula (fallback to 'X')
        formula = (desc.get("formula") or "X").strip()
        try:
            comp = Formula(formula).count()
        except Exception:
            comp = {"X": 1}

        species: List[str] = []
        for elem, count in comp.items():
            # Guard invalid symbols
            if elem not in chemical_symbols:
                elem = "X"
            species.extend([elem] * int(max(1, int(round(count)))) )

        # Limit to a manageable basis size (e.g., 8 sites)
        max_sites = 8
        total = min(len(species), max_sites) or 1
        species = species[:total] if species else ["X"]

        # Canonical fractional coordinates for up to 8 atoms
        frac_coords = [
            (0.0, 0.0, 0.0),
            (0.5, 0.5, 0.5),
            (0.0, 0.5, 0.5),
            (0.5, 0.0, 0.5),
            (0.5, 0.5, 0.0),
            (0.25, 0.25, 0.25),
            (0.75, 0.75, 0.75),
            (0.25, 0.75, 0.5),
        ][: total]

        cell_obj = Cell.fromcellpar(cellpar)
        atoms = Atoms(symbols="".join(species), scaled_positions=frac_coords, cell=cell_obj, pbc=True)
        return atoms, cellpar
    except Exception:
        return None, cellpar


def _standardize_with_spglib(atoms):
    try:
        import numpy as np
        import spglib

        if atoms is None:
            return None
        lattice = atoms.cell.array
        positions = atoms.get_scaled_positions()
        numbers = atoms.get_atomic_numbers()
        std = spglib.standardize_cell((lattice, positions, numbers), to_primitive=True, no_idealize=False)
        if std is None:
            return atoms
        lat, pos, nums = std
        from ase import Atoms

        std_atoms = Atoms(numbers=nums, scaled_positions=pos, cell=lat, pbc=True)
        return std_atoms
    except Exception:
        return atoms


def _write_cif(path: str, atoms, cellpar: Tuple[float, float, float, float, float, float]):
    """Try to write CIF via ASE; fallback to minimal CIF text."""
    try:
        from ase.io import write

        write(path, atoms, format="cif")
        return
    except Exception:
        pass

    # Fallback minimal CIF
    a, b, c, alpha, beta, gamma = cellpar
    content = (
        "data_generated\n"
        f"_cell_length_a    {a:.6f}\n"
        f"_cell_length_b    {b:.6f}\n"
        f"_cell_length_c    {c:.6f}\n"
        f"_cell_angle_alpha {alpha:.6f}\n"
        f"_cell_angle_beta  {beta:.6f}\n"
        f"_cell_angle_gamma {gamma:.6f}\n"
        "_symmetry_space_group_name_H-M   'P1'\n"
    )
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        pass


def _strained_cells(cellpar: Tuple[float, float, float, float, float, float]):
    a, b, c, alpha, beta, gamma = cellpar
    # Enumerate light strains on a,b,c; keep angles fixed here
    factors = [0.98, 0.99, 1.0, 1.01, 1.02]
    combos = []
    # Limit to at most 9 candidates by sampling around baseline
    preferred = [(1.0, 1.0, 1.0), (0.99, 1.0, 1.01), (1.01, 0.99, 1.0), (1.0, 1.01, 0.99), (1.02, 1.0, 1.0), (1.0, 1.02, 1.0), (1.0, 1.0, 1.02), (0.98, 1.0, 1.0), (1.0, 0.98, 1.0)]
    seen = set()
    for fa, fb, fc in preferred:
        key = (fa, fb, fc)
        if key in seen:
            continue
        seen.add(key)
        combos.append((fa, fb, fc))
    return [
        (a * fa, b * fb, c * fc, alpha, beta, gamma)
        for (fa, fb, fc) in combos
    ]


def gen_structures(state: S) -> S:
    desc: DescriptionJSON = state.get("final_description") or {}
    if not desc:
        # Fallback to draft description if final not set yet
        draft = state.get("draft_description", {}) or {}
        desc = draft.get("description_json", {}) or {}

    out_dir = _ensure_out_dir(state.get("out_dir", "out"))

    atoms, cellpar = _build_atoms_from_description(desc)
    atoms = _standardize_with_spglib(atoms)

    candidates: List[str] = []

    # Base structure
    base_path = os.path.join(out_dir, "candidate_base.cif")
    _write_cif(base_path, atoms, cellpar)
    candidates.append(base_path)

    # Strained variants
    try:
        if atoms is not None:
            from ase import Atoms
            from ase.cell import Cell

            for idx, cpar in enumerate(_strained_cells(cellpar), start=1):
                strained = atoms.copy()
                strained.set_cell(Cell.fromcellpar(cpar), scale_atoms=True)
                path = os.path.join(out_dir, f"candidate_strain_{idx:02d}.cif")
                _write_cif(path, strained, cpar)
                candidates.append(path)
        else:
            # No ASE; still emit several CIF stubs with strained cell params
            for idx, cpar in enumerate(_strained_cells(cellpar), start=1):
                path = os.path.join(out_dir, f"candidate_strain_{idx:02d}.cif")
                _write_cif(path, None, cpar)
                candidates.append(path)
    except Exception:
        pass

    state["cif_candidates"] = candidates
    return state
