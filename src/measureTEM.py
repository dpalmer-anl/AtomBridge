"""
Streamlit TEM measurement helpers (Ritesh workflow re-export).

Historically this module duplicated the Streamlit canvas utilities.  The
project now keeps a single canonical implementation in
``src.image_workflow_streamlit``; this file simply re-exports those helpers so
existing imports like ``from measureTEM import ...`` continue to work.
"""

from __future__ import annotations

try:
    from src.image_workflow_streamlit import (
        custom_select_roi_streamlit,
        get_scale_from_user_streamlit,
        measure_atomic_spacing_realspace,
    )
except ModuleNotFoundError:
    from image_workflow_streamlit import (
        custom_select_roi_streamlit,
        get_scale_from_user_streamlit,
        measure_atomic_spacing_realspace,
    )

# Backwards-compatible aliases (shorter legacy names)
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
