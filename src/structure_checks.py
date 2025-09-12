from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import os
import sys
import subprocess
from pathlib import Path


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


def _ensure_m3gnet_available(auto_install: bool = True) -> None:
    """Ensure pymatgen + m3gnet are importable. Optionally try a conda
    installation of m3gnet into the current prefix (no deps) if missing.
    """
    try:
        import pymatgen  # noqa: F401
        import m3gnet  # noqa: F401
        return
    except Exception as first_err:
        if not auto_install:
            raise RuntimeError(f"M3GNET validation unavailable: {first_err}")

        # Try installing with conda into the active prefix
        conda_exe = os.environ.get("CONDA_EXE", "conda")
        # Detect prefix: prefer CONDA_PREFIX, else sys.prefix if it looks like conda
        prefix = os.environ.get("CONDA_PREFIX")
        if not prefix:
            # Heuristic: conda envs have conda-meta
            if (Path(sys.prefix) / "conda-meta").exists():
                prefix = sys.prefix

        if not prefix:
            raise RuntimeError(
                "M3GNET validation unavailable: not running in a conda environment and packages are missing. "
                "Please run `conda install --no-deps m3gnet` in this environment."
            )

        try:
            proc = subprocess.run(
                [conda_exe, "install", "-y", "--no-deps", "-p", prefix, "m3gnet"],
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as e:
            raise RuntimeError(
                f"Attempted conda install of m3gnet failed to launch: {e}. "
                "Please run `conda install --no-deps m3gnet` manually."
            )

        if proc.returncode != 0:
            raise RuntimeError(
                "Conda install of m3gnet failed (rc=%s). Stdout tail:\n%s\nStderr tail:\n%s"
                % (proc.returncode, proc.stdout[-800:], proc.stderr[-800:])
            )

        # Retry import
        try:
            import pymatgen  # noqa: F401
            import m3gnet  # noqa: F401
        except Exception as e:
            raise RuntimeError(f"m3gnet/pymatgen still unavailable after conda install: {e}")


def validate_m3gnet(cif_path: str):
    """Run a quick M3GNET relaxation via pymatgen interface; returns (energy, relaxed_structure).
    Heavy dependency; safe to call in a try/except and treat as optional.
    """
    # Ensure availability, attempting conda install if needed
    _ensure_m3gnet_available(auto_install=True)
    from pymatgen.core import Structure  # type: ignore
    from m3gnet.models import Relaxer  # type: ignore
    structure = Structure.from_file(cif_path)
    relaxer = Relaxer()
    result = relaxer.relax(structure)
    energy = result["trajectory"].energies[0]
    relaxed_structure = result["final_structure"]
    return float(energy), relaxed_structure
