from __future__ import annotations

from typing import Any, Dict

from tem2cif.state import S, DescriptionJSON
from tem2cif.llm.prompts import SYNTHESIZE_DESCRIPTION_PROMPT


def synthesize_description(state: S) -> S:
    """Stub: fuse text priors with image metrics into a draft description.

    Creates a minimal description with any available evidence.
    """
    priors = state.get("priors_text", {}) or {}
    im = state.get("image_metrics", {}) or {}
    prov = (im.get("provenance") or {}) if isinstance(im, dict) else {}

    desc: DescriptionJSON = {
        "formula": priors.get("formula", ""),
        "spacegroup": priors.get("spacegroup", ""),
        "cell": priors.get("cell", {}),
        "Z": priors.get("Z", 0) or 0,
        "expected_d": priors.get("expected_d", []),
        "zone_axis": (priors.get("zone_axis") or (im.get("zone_axis_candidates") or [""])[0] or ""),
        "phase_labels": priors.get("phase_labels", []),
        "evidence": {
            "fig": prov.get("fig"),
            "page": prov.get("page"),
            "caption_quote": (priors.get("evidence", {}) or {}).get("caption_quote"),
        },
        "assumptions": priors.get("assumptions", []),
    }

    state["draft_description"] = {
        "summary": "Draft synthesized from text and FFT metrics.",
        "description_json": desc,
        "prompt": SYNTHESIZE_DESCRIPTION_PROMPT.strip(),
    }
    return state
