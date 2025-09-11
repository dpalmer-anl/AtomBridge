"""
Materials Project MCP integration (optional).

This module provides a thin wrapper to validate composition/lattice info
against Materials Project via an MCP-compatible server. To enable:

- Set env var MP_MCP_ENDPOINT (e.g., http://localhost:8000 for a local MCP)
- Optionally set MP_API_KEY if your MCP expects it (depends on implementation)

If not configured, functions return a 'skipped' result.
"""

from __future__ import annotations
import os
from typing import Optional, Dict, Any

import json
import urllib.request
import urllib.error


def _mp_endpoint() -> Optional[str]:
    return os.getenv("MP_MCP_ENDPOINT")


def mp_validate_composition(formula: str) -> Dict[str, Any]:
    """Validate a composition against Materials Project via MCP.
    Returns a dict with 'status' (ok|no_match|error|skipped) and any data.
    """
    endpoint = _mp_endpoint()
    if not endpoint:
        return {"status": "skipped", "reason": "MP_MCP_ENDPOINT not set"}

    payload = {"action": "search_composition", "formula": formula}
    req = urllib.request.Request(
        endpoint.rstrip("/") + "/query",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": os.getenv("MP_API_KEY", "")},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # Expected data shape depends on MCP; pass through
            if not data:
                return {"status": "no_match", "data": None}
            return {"status": "ok", "data": data}
    except urllib.error.HTTPError as e:
        return {"status": "error", "reason": f"HTTP {e.code}", "body": e.read().decode("utf-8", errors="ignore")}
    except Exception as e:
        return {"status": "error", "reason": str(e)}


def mp_validate_from_text(text: str) -> Dict[str, Any]:
    """Very simple heuristic: pick the first likely formula token and validate it.
    This is a placeholder; for robust parsing use an LLM or domain parser.
    """
    import re
    # Match tokens like LiCoO2, SrTiO3, Co3O4, etc.
    candidates = re.findall(r"\b[A-Z][a-z]?(?:\d+)?(?:[A-Z][a-z]?(?:\d+)?)+\b", text)
    for token in candidates:
        res = mp_validate_composition(token)
        if res.get("status") in {"ok", "no_match"}:
            res["queried_formula"] = token
            return res
    return {"status": "skipped", "reason": "no formula candidates found"}

