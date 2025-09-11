TEXT_PRIORS_PROMPT = """
Return grounded crystallographic fields from text+captions:
formula, spacegroup, unit cell (a,b,c,alpha,beta,gamma), Z,
reported d-spacings with (hkl) if stated, and zone-axis hints.
Output JSON matching description.json with evidence.caption_quote and page. Omit ungrounded fields.
"""

SYNTHESIZE_DESCRIPTION_PROMPT = """
Fuse text priors with image-derived FFT metrics. Produce:
(1) a concise 5–6 line summary tied to the paper’s hypothesis/conclusion,
(2) an updated description.json,
(3) a bullet list of uncertainties and assumptions.
"""

ASK_USER_CONFIRM_PROMPT = """
Show summary, d/θ table, chosen panel thumbnail. Options:
(1) Accept (2) Edit fields (key:value) (3) Change focus (figure/panel or keywords).
"""
