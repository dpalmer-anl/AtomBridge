from tem2cif.state import (
    Peak,
    ImageMetrics,
    DescriptionJSON,
    CIFScore,
    Feedback,
    S,
)
from tem2cif.llm.prompts import (
    TEXT_PRIORS_PROMPT,
    SYNTHESIZE_DESCRIPTION_PROMPT,
    ASK_USER_CONFIRM_PROMPT,
)
from tem2cif.utils.scoring import TIER1_THRESH, WEIGHTS


def test_state_typedicts_have_expected_keys():
    assert set(Peak.__annotations__.keys()) == {"d_A", "theta_deg", "intensity"}

    im_keys = set(ImageMetrics.__annotations__.keys())
    for k in ["scale_A_per_px", "zone_axis_candidates", "peaks", "uncertainties", "provenance"]:
        assert k in im_keys

    desc_keys = set(DescriptionJSON.__annotations__.keys())
    for k in [
        "formula",
        "spacegroup",
        "cell",
        "Z",
        "expected_d",
        "zone_axis",
        "phase_labels",
        "evidence",
        "assumptions",
    ]:
        assert k in desc_keys

    score_keys = set(CIFScore.__annotations__.keys())
    for k in ["path", "tier1", "tier2", "refine", "composite"]:
        assert k in score_keys

    fb_keys = set(Feedback.__annotations__.keys())
    for k in ["accept", "edits", "new_focus"]:
        assert k in fb_keys

    s_keys = set(S.__annotations__.keys())
    for k in [
        "pdf_path",
        "focus_query",
        "out_dir",
        "doc_text",
        "captions",
        "fig_images",
        "fig_meta",
        "panel_candidates",
        "focus_bundle",
        "priors_text",
        "image_metrics",
        "draft_description",
        "user_feedback",
        "final_description",
        "cif_candidates",
        "fft_scored_cifs",
        "rs_scored_cifs",
        "export",
    ]:
        assert k in s_keys


def test_prompts_are_strings_with_keywords():
    assert isinstance(TEXT_PRIORS_PROMPT, str) and "formula" in TEXT_PRIORS_PROMPT
    assert isinstance(SYNTHESIZE_DESCRIPTION_PROMPT, str) and "summary" in SYNTHESIZE_DESCRIPTION_PROMPT
    assert isinstance(ASK_USER_CONFIRM_PROMPT, str) and "Accept" in ASK_USER_CONFIRM_PROMPT


def test_scoring_constants():
    assert set(TIER1_THRESH.keys()) == {"f1", "drmse", "ang_deg"}
    # Weights should sum to 1 within tolerance
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-9
