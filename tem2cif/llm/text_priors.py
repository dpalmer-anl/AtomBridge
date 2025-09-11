from __future__ import annotations

from tem2cif.state import S, DescriptionJSON


def text_priors(state: S) -> S:
    """Stub: extract priors from text/captions.

    For now, provide an empty or minimal DescriptionJSON compatible dict.
    """
    priors: DescriptionJSON = {}
    state["priors_text"] = priors
    return state
