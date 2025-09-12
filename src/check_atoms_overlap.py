from ase import Atoms
import numpy as np

def check_atom_distances(atoms: Atoms, cutoff: float = 0.5):
    """
    Check if any two atoms in an ASE Atoms object are closer than cutoff.
    Raises a ValueError if such atoms are found.

    Parameters
    ----------
    atoms : ase.Atoms
        The atomic structure.
    cutoff : float, optional
        Distance threshold in Å (default: 0.5).

    Raises
    ------
    ValueError
        If any atoms are closer than the cutoff.
    """
    dists = atoms.get_all_distances(mic=True)
    np.fill_diagonal(dists, np.inf)

    # Find indices where distance < cutoff
    close_pairs = np.argwhere(dists < cutoff)

    if close_pairs.size > 0:
        messages = []
        for i, j in close_pairs:
            # only report each pair once (i < j)
            if i < j:
                messages.append(
                    f"Atoms {i} ({atoms[i].symbol}) and {j} ({atoms[j].symbol}) "
                    f"are too close: {dists[i, j]:.3f} Å"
                )
        raise ValueError("Too-close atoms detected:\n" + "\n".join(messages))


if __name__=="__main__":
    import ase.io
    atoms = ase.io.read("HT_LiCoO2_R-3m.cif")
    print(len(atoms))
    print(check_atom_distances(atoms))