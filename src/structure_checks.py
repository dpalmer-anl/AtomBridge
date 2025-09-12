from __future__ import annotations

from typing import Tuple, Dict

import numpy as np


def check_atom_distances(atoms, cutoff: float = 0.5):
    """Check if any two atoms in an ASE Atoms object are closer than cutoff.
    Raises ValueError listing offending pairs if found.
    """
    try:
        from ase import Atoms  # noqa: F401
    except Exception:
        pass
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, np.inf)
    close_pairs = np.argwhere(dists < cutoff)
    if close_pairs.size > 0:
        messages = []
        for i, j in close_pairs:
            if i < j:
                messages.append(
                    f"Atoms {i} ({atoms[i].symbol}) and {j} ({atoms[j].symbol}) are too close: {dists[i, j]:.3f} Å"
                )
        raise ValueError("Too-close atoms detected:\n" + "\n".join(messages))


def distance_summary(atoms, cutoff: float = 0.5) -> Dict[str, float | int]:
    """Return summary statistics for interatomic distances.
    - min_distance: global minimum pairwise distance (Å)
    - num_pairs_below_cutoff: number of unique pairs with distance < cutoff
    - cutoff: the cutoff used (Å)
    """
    d = atoms.get_all_distances(mic=True)
    # consider upper triangle only to avoid double counting
    iu = np.triu_indices_from(d, k=1)
    vals = d[iu]
    min_d = float(np.min(vals)) if vals.size else float("inf")
    num_bad = int(np.sum(vals < cutoff))
    return {"min_distance": min_d, "num_pairs_below_cutoff": num_bad, "cutoff": float(cutoff)}


def validate_m3gnet(cif_path: str):
    """Run a quick M3GNET relaxation via pymatgen interface; returns (energy, relaxed_structure).
    Heavy dependency; safe to call in a try/except and treat as optional.
    """
    try:
        from pymatgen.core import Structure  # type: ignore
        from m3gnet.models import Relaxer  # type: ignore
    except Exception as e:  # library missing
        raise RuntimeError(f"M3GNET validation unavailable: {e}")
    structure = Structure.from_file(cif_path)
    relaxer = Relaxer()
    result = relaxer.relax(structure)
    energy = result["trajectory"].energies[0]
    relaxed_structure = result["final_structure"]
    return float(energy), relaxed_structure
