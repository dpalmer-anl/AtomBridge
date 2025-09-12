"""
Materials Project API validation (preferred over MCP).

Environment:
- MP_API_KEY: your Materials Project API key.

Functions:
- mp_api_validate_composition(formula): query MP for composition summary.
- mp_api_validate_from_text(text): extract a likely formula from free text and validate.

Note: This is a thin HTTP wrapper using requests to avoid heavy deps.
"""
from __future__ import annotations
import os
from typing import Optional, Dict, Any
import re
import requests


BASE_URL = os.getenv("MP_API_BASE", "https://api.materialsproject.org")


def _headers() -> Dict[str, str]:
    key = os.getenv("MP_API_KEY", "").strip()
    h = {"Content-Type": "application/json"}
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def mp_api_validate_composition(formula: str) -> Dict[str, Any]:
    """Validate composition against Materials Project via REST API.
    Returns a dict with status and result payload.
    """
    if not os.getenv("MP_API_KEY"):
        return {"status": "skipped", "reason": "MP_API_KEY not set"}

    try:
        url = f"{BASE_URL}/materials/summary"
        params = {"formula": formula, "fields": "material_id,formula_pretty,spacegroup.symbol,spacegroup.number,structure.lattice"}
        r = requests.get(url, headers=_headers(), params=params, timeout=20)
        if r.status_code != 200:
            return {"status": "error", "reason": f"HTTP {r.status_code}", "body": r.text}
        data = r.json()
        results = data.get("data") if isinstance(data, dict) else data
        if not results:
            return {"status": "no_match", "data": None}
        return {"status": "ok", "data": results}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def mp_api_validate_from_text(text: str) -> Dict[str, Any]:
    """Extract the first plausible chemical formula and validate via API."""
    candidates = re.findall(r"\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?(?:\d+)?)+\b", text)
    for tok in candidates:
        res = mp_api_validate_composition(tok)
        if res.get("status") in {"ok", "no_match"}:
            res["queried_formula"] = tok
            return res
    return {"status": "skipped", "reason": "no formula candidates found"}

