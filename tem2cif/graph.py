from langgraph.graph import StateGraph, END
from tem2cif.state import S
from tem2cif.io.pdf_ingest import ingest_paper
from tem2cif.vision.link_text_to_images import link_text_to_images
from tem2cif.vision.choose_best_image import choose_best_image
from tem2cif.llm.text_priors import text_priors
from tem2cif.vision.segment_tem import segment_tem
from tem2cif.vision.filter_clean import filter_clean
from tem2cif.vision.measure_fft import measure_fft
from tem2cif.llm.synthesize_description import synthesize_description
from tem2cif.llm.ask_user_confirm import ask_user_confirm
from tem2cif.llm.revise_description import revise_description
from tem2cif.structures.gen_structures import gen_structures
from tem2cif.structures.tier1_fft_validate import tier1_fft_validate
from tem2cif.structures.tier2_realspace_validate import tier2_realspace_validate
from tem2cif.structures.auto_refine_validate import auto_refine_validate
from tem2cif.io.export import choose_export


def build_graph():
    g = StateGraph(S)

    for name, fn in {
        "ingest_paper": ingest_paper,
        "link_text_to_images": link_text_to_images,
        "choose_best_image": choose_best_image,
        "text_priors": text_priors,
        "segment_tem": segment_tem,
        "filter_clean": filter_clean,
        "measure_fft": measure_fft,
        "synthesize_description": synthesize_description,
        "ask_user_confirm": ask_user_confirm,
        "revise_description": revise_description,
        "gen_structures": gen_structures,
        "tier1_fft_validate": tier1_fft_validate,
        "tier2_realspace_validate": tier2_realspace_validate,
        "auto_refine_validate": auto_refine_validate,
        "choose_export": choose_export,
    }.items():
        g.add_node(name, fn)

    g.set_entry_point("ingest_paper")

    # entry → link text & pick best panel; run priors in parallel
    g.add_edge("ingest_paper", "link_text_to_images")
    g.add_edge("ingest_paper", "text_priors")
    g.add_edge("link_text_to_images", "choose_best_image")
    g.add_edge("choose_best_image", "segment_tem")
    g.add_edge("segment_tem", "filter_clean")
    g.add_edge("filter_clean", "measure_fft")
    g.add_edge("measure_fft", "synthesize_description")
    g.add_edge("text_priors", "synthesize_description")

    # confirmation loop
    g.add_edge("synthesize_description", "ask_user_confirm")

    def confirmed(s: S):
        return bool(s.get("user_feedback", {}).get("accept"))

    g.add_conditional_edges(
        "ask_user_confirm", confirmed, {True: "gen_structures", False: "revise_description"}
    )

    def focus_changed(s: S):
        return s.get("user_feedback", {}).get("new_focus") is not None

    g.add_conditional_edges(
        "revise_description",
        focus_changed,
        {True: "link_text_to_images", False: "synthesize_description"},
    )

    # generation → FFT gate → real-space → auto-refine → export
    g.add_edge("gen_structures", "tier1_fft_validate")

    def tier1_pass(s: S):
        return any(c.get("tier1", {}).get("pass") for c in s.get("fft_scored_cifs", []))

    g.add_conditional_edges(
        "tier1_fft_validate", tier1_pass, {True: "tier2_realspace_validate", False: "ask_user_confirm"}
    )

    g.add_edge("tier2_realspace_validate", "auto_refine_validate")
    g.add_edge("auto_refine_validate", "choose_export")
    g.add_edge("choose_export", END)

    return g.compile()
