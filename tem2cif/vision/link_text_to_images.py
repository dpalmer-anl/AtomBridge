from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional
    Image = None  # type: ignore

from tem2cif.state import S

# Optional utilities: degrade gracefully if not implemented
try:  # pragma: no cover - optional dependency wrapper
    from tem2cif.utils.clip import panel_text_similarity  # (text: str, image) -> float in [0,1]
except Exception:  # pragma: no cover
    panel_text_similarity = None  # type: ignore

try:  # pragma: no cover - optional dependency wrapper
    from tem2cif.utils.ocr import ocr_text  # (image) -> str
except Exception:  # pragma: no cover
    ocr_text = None  # type: ignore


def _extract_references(texts: List[str]) -> Tuple[set, set]:
    """Extract figure numbers and (figure, panel) references from text.

    Returns:
        (fig_nums, fig_panel_pairs) where elements are strings like '3' and ('3','b').
    """
    fig_nums = set()
    fig_panels: set[Tuple[str, str]] = set()

    joined = "\n".join([t for t in texts if t])
    # Examples: Fig. 3b, Figure 3b, Fig 3(b), Fig. 3 (b)
    for m in re.finditer(r"(?:Fig(?:ure)?\.?\s*)(\d+)\s*(?:[\(\[]?([a-zA-Z])\)?\]?)?", joined):
        num = m.group(1)
        pan = m.group(2)
        if num:
            fig_nums.add(num)
        if num and pan:
            fig_panels.add((num, pan.lower()))

    # Standalone panel refs like: panel (b), (b) panel
    for m in re.finditer(r"panel\s*[\(\[]?([a-zA-Z])\)?\]?", joined, flags=re.IGNORECASE):
        pan = m.group(1)
        if pan:
            # Without explicit fig number; handled as wildcard panel reference
            fig_panels.add(("*", pan.lower()))

    return fig_nums, fig_panels


def _crop_image(img: Any, bbox: List[int]) -> Any:
    """Crop image given bbox [x, y, w, h] or [x1, y1, x2, y2]. Supports PIL or numpy.
    Returns the cropped image object of the same type when feasible.
    """
    if img is None or bbox is None:
        return img

    # Normalize bbox to (x1, y1, x2, y2)
    if len(bbox) == 4:
        x1, y1, b3, b4 = bbox
        # Detect if input is w,h (positive dims) or x2,y2 (coordinates)
        if b3 > 0 and b4 > 0 and (Image is None or not isinstance(img, Image.Image)):
            # Treat as width/height for numpy; we'll handle PIL next
            x2, y2 = x1 + b3, y1 + b4
        else:
            x2, y2 = b3, b4
    else:
        return img

    try:
        if Image is not None and isinstance(img, Image.Image):
            # If width/height provided, ensure x2,y2 are max dims
            if x2 <= x1 or y2 <= y1:
                x2, y2 = x1 + b3, y1 + b4
            return img.crop((int(x1), int(y1), int(x2), int(y2)))
    except Exception:
        pass

    # Fallback: assume numpy array HxWxC
    try:
        import numpy as np  # local import

        arr = img
        if isinstance(arr, np.ndarray):
            return arr[int(y1) : int(y2), int(x1) : int(x2)]
    except Exception:
        pass

    return img


def _norm(x: Optional[float]) -> float:
    try:
        if x is None:
            return 0.0
        # Clamp to [0,1]
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.0


def _keyword_score(text: str, keywords: List[str]) -> float:
    text_l = text.lower()
    total = 0
    for kw in keywords:
        kw = kw.strip().lower()
        if not kw:
            continue
        total += len(re.findall(re.escape(kw), text_l))
    # Heuristic normalization: 0..1 with diminishing returns
    return 1.0 - (0.5 ** max(total, 0)) if total > 0 else 0.0


def link_text_to_images(state: S) -> S:
    """Tie paper text/captions to figure panels and rank candidates.

    Inputs (from state): doc_text, captions, fig_images, fig_meta, optional focus_query.
    Output: state["panel_candidates"] = List[Dict[str, Any]] entries with
        {fig_id, page, bbox, clip_score, keyword_score, regx_hit}
    """
    doc_text = state.get("doc_text", "") or ""
    captions: List[str] = state.get("captions", []) or []
    fig_images: List[Any] = state.get("fig_images", []) or []
    fig_meta: List[Dict[str, Any]] = state.get("fig_meta", []) or []
    focus_query: Optional[str] = state.get("focus_query")

    # Extract references from doc text, captions and focus query
    ref_texts = [doc_text] + captions + ([focus_query] if focus_query else [])
    fig_nums, fig_panels = _extract_references(ref_texts)

    # Keywords from focus query (and quoted phrases in doc?) Keep simple per spec
    keywords: List[str] = []
    if focus_query:
        # Split on commas and whitespace; keep up to 6 keywords to avoid over-weighting
        parts = re.split(r"[\s,;]+", focus_query)
        keywords = [p for p in parts if p]

    candidates: List[Dict[str, Any]] = []

    for i, meta in enumerate(fig_meta):
        fig_id = str(meta.get("fig_id", meta.get("id", i)))
        page = int(meta.get("page", meta.get("page_index", -1)))

        # Gather panels
        panels = meta.get("panels") or []
        if not panels:
            # Some metadata may provide a single bbox at figure level
            single_bbox = meta.get("bbox")
            panels = [
                {"panel_id": meta.get("panel_id", None), "bbox": single_bbox}
            ] if single_bbox is not None else [{"panel_id": None, "bbox": None}]

        # Associated figure-level caption (fallback by index)
        caption_text = meta.get("caption") or (captions[i] if i < len(captions) else "")

        # Get the figure image for cropping
        fig_img = meta.get("image") or (fig_images[i] if i < len(fig_images) else None)

        for p in panels:
            bbox = p.get("bbox")
            panel_id = str(p.get("panel_id")).lower() if p.get("panel_id") is not None else None

            # Regex hit if (fig, panel) appears, or if figure-only matches and no panel ID
            regx_hit = 0
            if (fig_id, panel_id or "").__class__:  # no-op to satisfy type checker variants
                if (fig_id, (panel_id or "").lower()) in fig_panels:
                    regx_hit = 1
                elif fig_id in fig_nums and not panel_id:
                    regx_hit = 1
                elif ("*", (panel_id or "").lower()) in fig_panels:
                    regx_hit = 1

            # Crop panel for CLIP and OCR
            crop = _crop_image(fig_img, bbox) if fig_img is not None else None

            # CLIP similarity (0..1), fall back to 0 if unavailable
            clip_score = 0.0
            if panel_text_similarity is not None and crop is not None:
                try:
                    # Use caption text primarily; if empty, use doc context
                    text_for_clip = caption_text or doc_text[:2000]
                    clip_score = float(panel_text_similarity(text_for_clip, crop))
                except Exception:
                    clip_score = 0.0

            # OCR keyword score
            keyword_score = 0.0
            if keywords and ocr_text is not None and crop is not None:
                try:
                    ocr_str = ocr_text(crop)
                    keyword_score = _keyword_score(ocr_str or "", keywords)
                except Exception:
                    keyword_score = 0.0

            candidates.append(
                {
                    "fig_id": fig_id,
                    "page": page,
                    "bbox": bbox,
                    "clip_score": _norm(clip_score),
                    "keyword_score": _norm(keyword_score),
                    "regx_hit": int(regx_hit),
                }
            )

    # Rank candidates by composite score
    def rank_key(c: Dict[str, Any]) -> float:
        return 0.6 * _norm(c.get("clip_score")) + 0.3 * _norm(c.get("keyword_score")) + 0.1 * (
            1.0 if c.get("regx_hit") else 0.0
        )

    candidates.sort(key=rank_key, reverse=True)

    # Update state
    state["panel_candidates"] = candidates
    return state
