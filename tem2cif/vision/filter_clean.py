from __future__ import annotations

from tem2cif.state import S


def filter_clean(state: S) -> S:
    """Stub: no-op filtering.

    Downstream FFT step performs denoising; this node can be expanded later
    for mask removal and contrast normalization.
    """
    return state
