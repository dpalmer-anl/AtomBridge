from __future__ import annotations

from tem2cif.state import S
from tem2cif.llm.prompts import ASK_USER_CONFIRM_PROMPT


def ask_user_confirm(state: S) -> S:
    """Stub: auto-accept the draft for now.

    Later, this should present a UI or chat loop for user input.
    """
    state["user_feedback"] = {"accept": True, "edits": {}, "new_focus": None, "prompt": ASK_USER_CONFIRM_PROMPT.strip()}
    return state
