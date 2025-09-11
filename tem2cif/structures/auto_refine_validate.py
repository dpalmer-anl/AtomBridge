from __future__ import annotations

import os
from typing import Dict, List, Optional

from tem2cif.state import S, CIFScore
from tem2cif.utils.scoring import WEIGHTS


def _read_atoms(path: str):
    try:
        from ase.io import read

        return read(path)
    except Exception:
        return None


def _standardize_spglib(atoms):
    try:
        import spglib
        if atoms is None:
            return None, False
        lattice = atoms.cell.array
        positions = atoms.get_scaled_positions()
        numbers = atoms.get_atomic_numbers()
        std = spglib.standardize_cell((lattice, positions, numbers), to_primitive=True, no_idealize=False)
        if std is None:
            return atoms, False
        lat, pos, nums = std
        from ase import Atoms

        std_atoms = Atoms(numbers=nums, scaled_positions=pos, cell=lat, pbc=True)
        return std_atoms, True
    except Exception:
        return atoms, False


def _density_g_cm3(atoms) -> Optional[float]:
    try:
        import numpy as np

        if atoms is None or atoms.cell is None:
            return None
        mass_amu = float(sum(atoms.get_masses()))
        vol_A3 = float(atoms.get_volume())
        NA = 6.02214076e23
        g = mass_amu / NA
        cm3 = vol_A3 * 1e-24
        if cm3 <= 0:
            return None
        return g / cm3
    except Exception:
        return None


def _stoich(atoms):
    try:
        from collections import Counter
        if atoms is None:
            return {}, False
        syms = list(atoms.get_chemical_symbols())
        counts = Counter(syms)
        chem_ok = all(s != "X" for s in syms)
        return dict(counts), chem_ok
    except Exception:
        return {}, False


def _composite_score(sc: Dict) -> float:
    # Compose a score from TEM-only metrics + refinement flags; ignore XRD
    w = WEIGHTS.copy() if isinstance(WEIGHTS, dict) else {"tier1": 0.5, "tier2": 0.5}
    components = {
        "tier1": float(((sc.get("tier1") or {}).get("f1") or 0.0)),
        "tier2": float(((sc.get("tier2") or {}).get("ssim") or 0.0)),
        "sg": 1.0 if ((sc.get("refine") or {}).get("sg_standardized")) else 0.0,
        "chem": 1.0 if ((sc.get("refine") or {}).get("chem_ok")) else 0.0,
        # "xrd": excluded by user request (TEM only)
    }
    used_keys = [k for k in ["tier1", "tier2", "sg", "chem"] if k in w]
    total_w = sum(w[k] for k in used_keys) or 1.0
    score = sum(w[k] * components[k] for k in used_keys) / total_w
    return float(max(0.0, min(1.0, score)))


def auto_refine_validate(state: S) -> S:
    refined: List[CIFScore] = []
    for sc in state.get("rs_scored_cifs", []) or []:
        sc = dict(sc)
        atoms = _read_atoms(sc.get("path", ""))
        atoms_std, sg_ok = _standardize_spglib(atoms)
        dens = _density_g_cm3(atoms_std)
        stoich_map, chem_ok = _stoich(atoms_std)

        ref = dict(sc.get("refine", {}))
        ref.update({
            "sg_standardized": bool(sg_ok),
            "density_g_cm3": float(dens) if dens is not None else None,
            "stoich": stoich_map,
            "chem_ok": bool(chem_ok),
        })
        sc["refine"] = ref
        sc["composite"] = _composite_score(sc)
        refined.append(sc)

    # Sort by composite descending
    refined.sort(key=lambda x: x.get("composite", 0.0), reverse=True)
    state["rs_scored_cifs"] = refined
    return state
