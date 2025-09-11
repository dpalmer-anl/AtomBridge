from typing import TypedDict, List, Optional, Dict, Any


class Peak(TypedDict):
    d_A: float
    theta_deg: float
    intensity: float


class ImageMetrics(TypedDict, total=False):
    scale_A_per_px: float
    zone_axis_candidates: List[str]
    peaks: List[Peak]
    uncertainties: Dict[str, float]
    provenance: Dict[str, Any]  # fig ID, bbox, quality_score, page


class DescriptionJSON(TypedDict, total=False):
    formula: str
    spacegroup: str
    cell: Dict[str, float]  # a,b,c,alpha,beta,gamma
    Z: int
    expected_d: List[Dict[str, Any]]  # {"hkl": "111", "d_A": 2.05}
    zone_axis: str
    phase_labels: List[str]
    evidence: Dict[str, Any]  # fig, caption_quote, page
    assumptions: List[str]


class CIFScore(TypedDict, total=False):
    path: str
    tier1: Dict[str, Any]
    tier2: Dict[str, Any]
    refine: Dict[str, Any]
    composite: float


class Feedback(TypedDict, total=False):
    accept: bool
    edits: Dict[str, Any]
    new_focus: Optional[str]  # e.g., "Fig. 3b kinks"


class S(TypedDict, total=False):
    pdf_path: str
    focus_query: Optional[str]
    out_dir: str
    doc_text: str
    captions: List[str]
    fig_images: List[Any]  # PIL or np.ndarray
    fig_meta: List[Dict[str, Any]]
    panel_candidates: List[Dict[str, Any]]
    focus_bundle: Dict[str, Any]  # {"image": ..., "bbox":..., "fig_id":..., "page":..., "score":...}
    priors_text: DescriptionJSON
    image_metrics: ImageMetrics
    draft_description: Dict[str, Any]
    user_feedback: Feedback
    final_description: DescriptionJSON
    cif_candidates: List[str]
    fft_scored_cifs: List[CIFScore]
    rs_scored_cifs: List[CIFScore]
    export: Dict[str, Any]
