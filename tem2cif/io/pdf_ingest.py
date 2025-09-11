from __future__ import annotations

import os
from typing import Any, Dict, List

from tem2cif.state import S


def ingest_paper(state: S) -> S:
    """Stub: populate minimal ingest fields if missing.

    In a full implementation, this would parse the PDF to extract text,
    captions, figure metadata and images. Here we ensure downstream steps
    have keys to operate on (possibly empty).
    """
    state.setdefault("doc_text", "")
    state.setdefault("captions", [])
    state.setdefault("fig_images", [])
    state.setdefault("fig_meta", [])

    # Ensure organized folder structure under out_dir
    out = state.get("out_dir", "out")
    try:
        os.makedirs(out, exist_ok=True)
        paper_dir = os.path.join(out, "paper")
        manual_dir = os.path.join(out, "manual")
        # New aggregate dirs to hold multiple inputs
        papers_dir = os.path.join(out, "papers")
        images_dir = os.path.join(out, "images")
        manual_single = os.path.join(manual_dir, "single")
        manual_multi = os.path.join(manual_dir, "multi")
        for d in [paper_dir, manual_dir, manual_single, manual_multi, papers_dir, images_dir]:
            os.makedirs(d, exist_ok=True)
        # Expose paths for downstream/manual use
        state["dirs"] = {
            "out": out,
            "paper": paper_dir,
            "papers": papers_dir,
            "manual": manual_dir,
            "manual_single": manual_single,
            "manual_multi": manual_multi,
            "images": images_dir,
        }
    except Exception:
        # Best-effort; continue without raising
        pass
    return state
