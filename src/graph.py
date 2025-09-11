import os
from typing import Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from .mp_mcp import mp_validate_from_text

from .utils_paper_and_code import parse_pdf, extract_code, run_code
from .create_ASE_RAG import RAG_ASE


class S(TypedDict, total=False):
    """Pipeline state for TEM→unit cell→ASE→CIF flow."""
    paper_path: str
    paper_text: str
    user_notes: Optional[str]
    target_plan: str
    rag_answer: str
    generated_code: str
    run_stdout: str
    run_stderr: str
    run_rc: int
    # model selection
    plan_model: Optional[str]
    code_model: Optional[str]


def _get_llm(model_name: str):
    return init_chat_model(model_name, model_provider="google_genai")


def load_paper(state: S) -> S:
    """Load and parse the PDF text into state.paper_text."""
    pdf = state.get("paper_path")
    if not pdf:
        raise ValueError("paper_path must be provided in state")
    text = parse_pdf(pdf)
    return {"paper_text": text}


def plan_targets(state: S) -> S:
    """Use LLM to propose target systems and structure scope from paper text."""
    paper = state.get("paper_text", "")
    user = state.get("user_notes", "")
    plan_model = (
        state.get("plan_model")
        or os.getenv("MODEL_PLAN_NAME")
        or "gemini-2.5-flash"
    )
    prompt = (
        "You are assisting with atomistic structure generation from a paper.\n"
        "Paper text follows. Identify the crystalline systems relevant to the main hypothesis.\n"
        "Propose primitive unit cells and any needed parameters (composition, lattice constants, space group).\n"
        "If information is missing, make reasonable assumptions to sweep.\n\n"
        f"User notes (optional): {user}\n\n"
        f"Paper:\n{paper[:12000]}\n\n"
        "Return a concise plan and the specific structures to generate."
    )
    llm = _get_llm(plan_model)
    msg = llm.invoke(prompt)
    return {"target_plan": msg.content}


def synthesize_code(state: S) -> S:
    """Use ASE RAG to draft Python code that builds ASE Atoms and writes CIF files."""
    plan = state.get("target_plan", "")
    code_model = (
        state.get("code_model")
        or os.getenv("MODEL_CODE_NAME")
        or "gemini-2.5-pro"
    )
    rag_tool = RAG_ASE(model_name=code_model)
    query = (
        "Write Python code that constructs ase.Atoms objects for the systems described below, "
        "creates supercells where appropriate, optionally introduces defects or grain boundaries as described, "
        "and writes each structure to a descriptive .cif file in the current working directory using ase.io.write.\n\n"
        f"Targets:\n{plan}"
    )
    rag_res = rag_tool(query)
    answer = rag_res.get("answer", "") or ""
    code = extract_code(answer)
    return {"rag_answer": answer, "generated_code": code}


def run_generated_code(state: S) -> S:
    """Execute the generated code once and capture outputs."""
    code = state.get("generated_code", "")
    if not code:
        return {"run_stdout": "", "run_stderr": "No code generated.", "run_rc": 1}
    stdout, stderr, rc = run_code(code)
    return {"run_stdout": stdout, "run_stderr": stderr, "run_rc": rc}


def build_graph():
    g = StateGraph(S)
    g.add_node("load_paper", load_paper)
    g.add_node("plan_targets", plan_targets)
    # Optional validation against Materials Project MCP
    def mp_validate(state: S) -> S:
        plan = state.get("target_plan", "")
        res = mp_validate_from_text(plan)
        return {"mp_validation": res}

    g.add_node("mp_validate", mp_validate)
    g.add_node("synthesize_code", synthesize_code)
    g.add_node("run_generated_code", run_generated_code)

    g.add_edge(START, "load_paper")
    g.add_edge("load_paper", "plan_targets")
    g.add_edge("plan_targets", "mp_validate")
    g.add_edge("mp_validate", "synthesize_code")
    g.add_edge("synthesize_code", "run_generated_code")
    g.add_edge("run_generated_code", END)

    return g.compile()


# Convenience entry to run with a provided state
def run_graph(paper_path: str, user_notes: Optional[str] = None, *, plan_model: Optional[str] = None, code_model: Optional[str] = None) -> S:
    graph = build_graph()
    init_state: S = {"paper_path": paper_path}
    if user_notes:
        init_state["user_notes"] = user_notes
    if plan_model:
        init_state["plan_model"] = plan_model
    if code_model:
        init_state["code_model"] = code_model
    out = {}
    for event in graph.stream(init_state):
        out.update(list(event.values())[-1])
    return out  # final state delta
