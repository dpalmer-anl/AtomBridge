from __future__ import annotations

from typing import Any, Dict

from tem2cif.state import S


def revise_description(state: S) -> S:
    """Stub: apply simple key:value edits to draft_description if provided.

    If `new_focus` is present in feedback, also set `focus_query` so the
    graph can branch back to linking.
    """
    fb = state.get("user_feedback", {}) or {}
    edits: Dict[str, Any] = fb.get("edits", {}) or {}
    if edits:
        draft = state.get("draft_description", {}) or {}
        dj = draft.get("description_json", {}) or {}
        dj.update(edits)
        draft["description_json"] = dj
        state["draft_description"] = draft

    if fb.get("new_focus"):
        state["focus_query"] = fb.get("new_focus")

    # Default to not accepted to keep loop alive if called
    state["user_feedback"] = {"accept": False, "edits": edits, "new_focus": fb.get("new_focus")}
    return state
