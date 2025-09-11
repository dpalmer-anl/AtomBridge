from __future__ import annotations

from tem2cif.state import S


def segment_tem(state: S) -> S:
    """Stub: return focus bundle as-is.

    A full version would detect the lattice ROI. For now, keep the selected
    panel image unchanged.
    """
    return state
