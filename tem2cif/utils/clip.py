"""
Lightweight wrapper for panel_text_similarity.

Attempts to use a local CLIP implementation if available. If not, returns
0.0 so downstream logic can rely on regex/keyword scoring.
"""

from __future__ import annotations

from typing import Any, Tuple


def _to_pil(img: Any):  # pragma: no cover - best effort conversion
    try:
        from PIL import Image
        import numpy as np

        if isinstance(img, Image.Image):
            return img
        if isinstance(img, np.ndarray):
            mode = "RGB"
            if img.ndim == 2:
                mode = "L"
            elif img.ndim == 3 and img.shape[2] == 4:
                mode = "RGBA"
            return Image.fromarray(img.astype("uint8"), mode=mode)
    except Exception:
        pass
    return None


def _load_clip():  # pragma: no cover - optional
    """Try to load a CLIP model from available packages.

    Returns (model, preprocess, device) or (None, None, None) on failure.
    """
    # Try open_clip first
    try:
        import torch
        import open_clip

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model = model.to(device)
        model.eval()
        return model, preprocess, device
    except Exception:
        pass

    # Try OpenAI CLIP
    try:
        import torch
        import clip  # type: ignore

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        return model, preprocess, device
    except Exception:
        pass

    return None, None, None


_MODEL, _PREPROC, _DEVICE = None, None, None


def _ensure_model():  # pragma: no cover - optional
    global _MODEL, _PREPROC, _DEVICE
    if _MODEL is None:
        _MODEL, _PREPROC, _DEVICE = _load_clip()
    return _MODEL, _PREPROC, _DEVICE


def panel_text_similarity(text: str, image: Any) -> float:
    """Return a similarity score in [0,1] between text and image panel.

    Uses CLIP if available; otherwise returns 0.0 as a neutral placeholder.
    """
    # Attempt model load lazily
    model, preprocess, device = _ensure_model()
    if model is None or preprocess is None or device is None:
        return 0.0

    # Convert image
    pil = _to_pil(image)
    if pil is None:
        return 0.0

    # Compute similarity via available backend
    try:  # pragma: no cover
        import torch

        with torch.no_grad():
            image_in = preprocess(pil).unsqueeze(0)
            image_in = image_in.to(device)

            # open_clip and clip share similar encode APIs
            if hasattr(model, "encode_image"):
                img_feat = model.encode_image(image_in)
            else:
                return 0.0

            # Text processing differs slightly; handle both
            if hasattr(model, "encode_text"):
                if "open_clip" in str(type(model)):  # crude check
                    import open_clip

                    tokenizer = open_clip.get_tokenizer("ViT-B-32")
                    txt_tokens = tokenizer([text])
                else:
                    import clip  # type: ignore

                    txt_tokens = clip.tokenize([text])

                txt_tokens = txt_tokens.to(device)
                txt_feat = model.encode_text(txt_tokens)
            else:
                return 0.0

            # Normalize and cosine similarity
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ txt_feat.T).squeeze().item()

        # sim is in [-1,1]; map to [0,1]
        return 0.5 * (sim + 1.0)
    except Exception:
        return 0.0
