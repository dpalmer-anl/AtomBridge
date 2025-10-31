"""
Legacy TEM measurement entrypoint (Streamlit-backed).

This module previously contained the standalone OpenCV desktop workflow.
The project now standardises on the Streamlit canvas utilities that live in
``src.image_workflow_streamlit`` (the Ritesh workflow).  To avoid breaking
older imports, we simply re-export those helpers here.
"""

from __future__ import annotations

try:  # Prefer absolute import when the project root is on sys.path
    from src.image_workflow_streamlit import (
        custom_select_roi_streamlit,
        get_scale_from_user_streamlit,
        measure_atomic_spacing_realspace,
    )
except ModuleNotFoundError:  # Fallback for direct execution from within src/
    from image_workflow_streamlit import (
        custom_select_roi_streamlit,
        get_scale_from_user_streamlit,
        measure_atomic_spacing_realspace,
    )

# Backwards-compatible aliases used by legacy callers
get_scale_from_user = get_scale_from_user_streamlit
custom_select_roi = custom_select_roi_streamlit
measure_atomic_spacing = measure_atomic_spacing_realspace

__all__ = [
    "get_scale_from_user_streamlit",
    "custom_select_roi_streamlit",
    "measure_atomic_spacing_realspace",
    "get_scale_from_user",
    "custom_select_roi",
    "measure_atomic_spacing",
]
