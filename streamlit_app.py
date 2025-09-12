import os
import time
from pathlib import Path
import streamlit as st

from src.graph import load_paper, plan_targets, synthesize_code, run_generated_code
from src.utils_paper_and_code import suggest_prompts_from_paper, extract_code, run_code, build_constraints_prompt, is_llm_quota_error
from src.create_ASE_RAG import RAG_ASE
from src.mp_api import mp_api_validate_from_text
from src.figures import extract_figures, run_tem_to_atom_coords
from src.validation import compare_image_coords_to_cif


UPLOAD_DIR = Path("_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def save_uploaded_pdf(uploaded_file) -> str:
    dest = UPLOAD_DIR / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(dest)


def list_papers_dir() -> list[str]:
    papers = []
    pdir = Path("papers")
    if pdir.exists():
        for p in sorted(pdir.glob("*.pdf")):
            papers.append(str(p))
    return papers


def run_pipeline_stepwise(pdf_path: str, notes: str | None, do_exec: bool, plan_model: str, code_model: str) -> dict:
    state: dict = {"paper_path": pdf_path, "plan_model": plan_model, "code_model": code_model}
    state.update(load_paper(state))
    if notes:
        state["user_notes"] = notes
    state.update(plan_targets(state))
    state.update(synthesize_code(state))
    if do_exec:
        state.update(run_generated_code(state))
    return state


def generate_and_fix_code(user_prompt: str, paper_text: str, code_model: str, max_iters: int = 3):
    """Use ASE RAG (code_model) to generate code. If it errors, loop with error context.
    Returns dict with keys: code, stdout, stderr, rc, iterations, history (list of dicts).
    """
    rag_tool = RAG_ASE(model_name=code_model)

    history = []
    last_code = ""
    last_stdout = ""
    last_stderr = ""
    last_rc = 1

    for i in range(max_iters):
        if i == 0:
            query = (
                "Write Python code that constructs ase.Atoms objects for the task below, "
                "creates supercells/defects/GBs as requested, and writes each to .cif via ase.io.write.\n"
                "Only return a single fenced Python code block.\n\n"
                f"Task: {user_prompt}\n\n"
                "Paper context (may contain targets and lattice hints):\n"
                f"{paper_text[:8000]}\n"
            )
        else:
            query = (
                "The previous generated code failed. Here is the error. Fix the code.\n"
                "Return only a single fenced Python code block that can run as-is.\n\n"
                f"Error:\n{last_stderr[:4000]}\n\n"
                f"Original task: {user_prompt}\n"
            )

        rag_res = rag_tool(query)
        answer = rag_res.get("answer", "") or ""
        code = extract_code(answer)
        stdout, stderr, rc = run_code(code)

        history.append({
            "iteration": i + 1,
            "query": query,
            "answer": answer,
            "code": code,
            "stdout": stdout,
            "stderr": stderr,
            "rc": rc,
        })

        last_code, last_stdout, last_stderr, last_rc = code, stdout, stderr, rc
        if rc == 0:
            break

    # Build a concise fix report
    fix_lines = []
    for h in history:
        if h["rc"] == 0:
            fix_lines.append(f"Iteration {h['iteration']}: success.")
        else:
            # include an error snippet and note that we attempted a fix
            err = (h.get("stderr") or "").splitlines()
            err_snip = "; ".join(err[:2]) if err else "error occurred"
            # simple supercell hint
            supercell_hint = "supercell" if "supercell" in (h.get("stderr") or "").lower() else None
            if supercell_hint:
                fix_lines.append(f"Iteration {h['iteration']}: error while generating supercell â€” {err_snip}. Tried to regenerate code with error context.")
            else:
                fix_lines.append(f"Iteration {h['iteration']}: error â€” {err_snip}. Tried to regenerate code with error context.")

    return {
        "code": last_code,
        "stdout": last_stdout,
        "stderr": last_stderr,
        "rc": last_rc,
        "iterations": len(history),
        "history": history,
        "fix_report": "\n".join(fix_lines) if fix_lines else "",
    }


def generate_and_fix_code_v2(user_prompt: str, paper_text: str, code_model: str, max_iters: int = 3):
    """Revised: generate code via ASE RAG and iteratively fix on errors.
    Returns dict with: code, stdout, stderr, rc, iterations, history, fix_report.
    """
    rag_tool = RAG_ASE(model_name=code_model)
    history = []
    last_code = ""
    last_stdout = ""
    last_stderr = ""
    last_rc = 1

    for i in range(max_iters):
        if i == 0:
            query = (
                "Write Python code that constructs ase.Atoms objects for the task below, "
                "creates supercells/defects/GBs as requested, and writes each to .cif via ase.io.write.\n"
                "Only return a single fenced Python code block.\n\n"
                f"Task: {user_prompt}\n\n"
                "Paper context (may contain targets and lattice hints):\n"
                f"{paper_text[:8000]}\n"
            )
        else:
            query = (
                "The previous generated code failed. Here is the error. Fix the code.\n"
                "Return only a single fenced Python code block that can run as-is.\n\n"
                f"Error:\n{last_stderr[:4000]}\n\n"
                f"Original task: {user_prompt}\n"
            )

        rag_res = rag_tool(query)
        answer = rag_res.get("answer", "") or ""
        code = extract_code(answer)
        stdout, stderr, rc = run_code(code)
        history.append({
            "iteration": i + 1,
            "query": query,
            "answer": answer,
            "code": code,
            "stdout": stdout,
            "stderr": stderr,
            "rc": rc,
        })
        last_code, last_stdout, last_stderr, last_rc = code, stdout, stderr, rc
        if rc == 0:
            break

    # Build fix report
    fix_lines = []
    for h in history:
        if h.get("rc") == 0:
            fix_lines.append(f"Iteration {h['iteration']}: success.")
        else:
            err = (h.get("stderr") or "").splitlines()
            err_snip = "; ".join(err[:2]) if err else "error occurred"
            if "supercell" in (h.get("stderr") or "").lower():
                fix_lines.append(
                    f"Iteration {h['iteration']}: error while generating supercell â€” {err_snip}. Tried to regenerate code with error context."
                )
            else:
                fix_lines.append(
                    f"Iteration {h['iteration']}: error â€” {err_snip}. Tried to regenerate code with error context."
                )

    return {
        "code": last_code,
        "stdout": last_stdout,
        "stderr": last_stderr,
        "rc": last_rc,
        "iterations": len(history),
        "history": history,
        "fix_report": "\n".join(fix_lines) if fix_lines else "",
    }


st.set_page_config(page_title="Atombridge", layout="wide")
st.title("TEM â†’ CIF")
st.caption("Run the minimal pipeline without using the terminal.")

with st.sidebar:
    st.header("Environment")
    key_present = bool(os.getenv("GOOGLE_API_KEY"))
    st.write("GOOGLE_API_KEY:", "âœ… set" if key_present else "âŒ missing")
    api_key_input = st.text_input("Enter API key (not saved)", type="password", help="Used only for this session")
    if st.button("Use key for this session"):
        if api_key_input.strip():
            os.environ["GOOGLE_API_KEY"] = api_key_input.strip()
            st.success("API key set for this session.")
            key_present = True
        else:
            st.warning("Please enter a non-empty key.")
    st.divider()
    st.markdown("- Ensure ASE is installed locally.\n- First run may build a local Chroma DB.")
    st.divider()
    st.subheader("Mode")
    mode = st.selectbox(
        "Choose a mode",
        options=["Balanced (recommended)", "Accurate (slower)", "Fast (cheaper)"],
        index=0,
        help="Balanced uses Flash for planning and Pro for code. Accurate uses Pro for both. Fast uses Flash for both.",
    )

    # Map mode â†’ models
    if mode.startswith("Balanced"):
        plan_model = "gemini-2.5-flash"
        code_model = "gemini-2.5-pro"
    elif mode.startswith("Accurate"):
        plan_model = "gemini-2.5-pro"
        code_model = "gemini-2.5-pro"
    else:  # Fast
        plan_model = "gemini-2.5-flash"
        code_model = "gemini-2.5-flash"

    with st.expander("Advanced model settings"):
        plan_model = st.selectbox(
            "Planning model",
            options=["gemini-2.5-flash", "gemini-2.5-pro"],
            index=["gemini-2.5-flash", "gemini-2.5-pro"].index(plan_model),
            help="Used for extracting targets/plan from the paper",
        )
    st.divider()
    st.subheader("Materials Project API Validation")
    st.caption("Validate via official MP API (preferred). Enter your MP API key; no server needed.")
    mp_api_key = st.text_input("MP API key", type="password")
    col_mp1, col_mp2 = st.columns(2)
    with col_mp1:
        if st.button("Use MP API key for this session"):
            if mp_api_key.strip():
                os.environ["MP_API_KEY"] = mp_api_key.strip()
                st.success("MP API key set for this session.")
            else:
                st.warning("Please enter a non-empty MP API key.")
    with col_mp2:
        validate_mp_api = st.checkbox("Validate with MP API during run", value=False)
    if st.button("Validate last result (MP API)"):
        if "last_result" in st.session_state and st.session_state.get("last_result") and os.getenv("MP_API_KEY"):
            plan_text = st.session_state.get("paper_text") or ""
            val = mp_api_validate_from_text(plan_text)
            st.session_state.last_result["mp_validation"] = val
            st.success("Validated last result (see Materials Project Validation section).")
        else:
            st.warning("No last result or MP API key not set.")
        code_model = st.selectbox(
            "Codegen model",
            options=["gemini-2.5-pro", "gemini-2.5-flash"],
            index=["gemini-2.5-pro", "gemini-2.5-flash"].index(code_model),
            help="Used by ASE RAG to generate Python code",
        )

st.subheader("Select a paper")
source = st.radio("PDF source", ["Upload", "From papers/"], horizontal=True)

pdf_path: str | None = None
if source == "Upload":
    up = st.file_uploader("Upload a PDF", type=["pdf"])
    if up is not None:
        pdf_path = save_uploaded_pdf(up)
        st.success(f"Saved to {pdf_path}")
else:
    options = list_papers_dir()
    pdf_path = st.selectbox("Choose from papers/", options) if options else None
    if not options:
        st.info("No PDFs found in papers/ directory.")

if "paper_text" not in st.session_state:
    st.session_state.paper_text = None
if "suggested_prompts" not in st.session_state:
    st.session_state.suggested_prompts = []
if "conversation" not in st.session_state:
    st.session_state.conversation = []  # list of {role: user/assistant, content: str}
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_cifs" not in st.session_state:
    st.session_state.last_cifs = []
if "last_code" not in st.session_state:
    st.session_state.last_code = ""
if "figures" not in st.session_state:
    st.session_state.figures = []
if "fig_coords" not in st.session_state:
    st.session_state.fig_coords = {}

analyze = st.button("Analyze Paper & Suggest Prompts")
if analyze:
    if not pdf_path:
        st.error("Please provide a PDF (upload or pick from papers/)")
        st.stop()
    with st.spinner("Reading paper and proposing prompts…" ):€¦"):
        try:
            # Reuse the graph helper to parse the paper
            state = {"paper_path": pdf_path}
            st.session_state.paper_text = load_paper(state)["paper_text"]
            st.session_state.suggested_prompts = suggest_prompts_from_paper(
                st.session_state.paper_text, model_name=plan_model
            )
        except Exception as e:
            st.error(f"Failed to analyze paper: {e}")
            st.stop()

if st.session_state.suggested_prompts:
    st.subheader("Suggested Prompts (click to run)")
    btn_cols = st.columns(len(st.session_state.suggested_prompts))
    for i, (c, sp) in enumerate(zip(btn_cols, st.session_state.suggested_prompts)):
        with c:
            if st.button(sp, key=f"suggest_{i}"):
                # Put into prompt box and run full pipeline automatically
                st.session_state["prompt_input"] = sp
                st.session_state["auto_exec"] = True
                st.rerun()
else:
    st.caption("Click 'Analyze Paper & Suggest Prompts' to get suggestions, or type your own below.")

st.subheader("Constraints (optional)")
with st.expander("Specify constraints from TEM/paper"):
    comp = st.text_input("Composition (formula)", placeholder="e.g., MoS2 or LiCoO2")
    sg = st.text_input("Space group (symbol or number)", placeholder="e.g., P6_3/mmc or 194")
    c1, c2, c3 = st.columns(3)
    with c1:
        a_val = st.number_input("a (Ã…)", min_value=0.0, value=0.0, step=0.01)
        alpha_val = st.number_input("alpha (Â°)", min_value=0.0, max_value=180.0, value=0.0, step=0.1)
    with c2:
        b_val = st.number_input("b (Ã…)", min_value=0.0, value=0.0, step=0.01)
        beta_val = st.number_input("beta (Â°)", min_value=0.0, max_value=180.0, value=0.0, step=0.1)
    with c3:
        c_val = st.number_input("c (Ã…)", min_value=0.0, value=0.0, step=0.01)
        gamma_val = st.number_input("gamma (Â°)", min_value=0.0, max_value=180.0, value=0.0, step=0.1)

    d_text = st.text_area("d-spacings (Ã…, comma or space separated)", placeholder="e.g., 2.46, 1.42, 1.23")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        scx = st.number_input("Supercell Nx", min_value=1, value=1, step=1)
    with sc2:
        scy = st.number_input("Supercell Ny", min_value=1, value=1, step=1)
    with sc3:
        scz = st.number_input("Supercell Nz", min_value=1, value=1, step=1)

    defect_opts = ["vacancy", "substitution", "interstitial", "antisite", "dopant"]
    defects = st.multiselect("Defects", options=defect_opts, default=[])
    defect_details = st.text_area("Defect details", placeholder="e.g., S vacancy at 1% in 2x2 supercell")

    gb = st.checkbox("Include grain boundary")
    gb_desc = st.text_input("Grain boundary description", disabled=not gb, placeholder="e.g., tilt GB ~5Â° along [10-10]")

notes = st.text_area(
    "Your prompt (optional)",
    key="prompt_input",
    placeholder="e.g., Extract all structures and write CIFs; include 2x2 supercells and S vacancy variants.",
)

# Build constraints string
parsed_d = []
if d_text.strip():
    try:
        # split by comma or whitespace
        for tok in d_text.replace(",", " ").split():
            parsed_d.append(float(tok))
    except Exception:
        parsed_d = [s for s in d_text.replace(",", " ").split() if s]

constraints = {
    "composition": comp.strip() or None,
    "space_group": sg.strip() or None,
    "a": a_val or None,
    "b": b_val or None,
    "c": c_val or None,
    "alpha": alpha_val or None,
    "beta": beta_val or None,
    "gamma": gamma_val or None,
    "d_spacings": parsed_d,
    "supercell": (int(scx), int(scy), int(scz)),
    "defects": defects,
    "defect_details": defect_details.strip() or None,
    "grain_boundary": bool(gb),
    "gb_description": gb_desc.strip() if gb else None,
}
constraints_text = build_constraints_prompt(constraints)

final_prompt = (notes or "").strip()
if constraints_text:
    final_prompt = (final_prompt + "\n\nConstraints:\n" + constraints_text).strip()
if not final_prompt:
    st.caption("Provide a prompt or analyze the paper to pick a suggestion, and optionally add constraints.")

col1, col2 = st.columns(2)
with col1:
    dry = st.button("Plan & Generate Code (no exec)")
with col2:
    full = st.button("Run Full Pipeline (exec code)")

auto_exec = bool(st.session_state.get("auto_exec"))
if auto_exec and not pdf_path:
    st.error("Please provide a PDF (upload or pick from papers/) before using a suggested prompt.")
    st.session_state["auto_exec"] = False

if (dry or full or auto_exec):
    if not pdf_path:
        st.error("Please provide a PDF (upload or pick from papers/)")
        st.stop()
    do_exec = bool(full or auto_exec)
    start_ts = time.time()
    with st.spinner("Running pipeline… this may take a moment on first run"):
        try:
            # If user gave a direct prompt, use generate_and_fix_code; otherwise run the graph path
            if final_prompt:
                # Plan summary (optional) using plan_targets, to enrich context
                state = {"paper_path": pdf_path}
                state.update(load_paper(state))
                if st.session_state.paper_text is None:
                    st.session_state.paper_text = state.get("paper_text")
                # Use iterative codegen + fix
                result = generate_and_fix_code_v2(
                    user_prompt=final_prompt,
                    paper_text=st.session_state.paper_text or "",
                    code_model=code_model,
                    max_iters=3 if do_exec else 1,
                )
                # Optional MP API validation
                if 'validate_mp_api' in locals() and validate_mp_api and os.getenv("MP_API_KEY"):
                    mp_res = mp_api_validate_from_text(st.session_state.paper_text or "")
                    result["mp_validation"] = mp_res
                # If not executing, do not run; just show the code produced in first iteration
                if not do_exec:
                    result["rc"] = None
            else:
                result = run_pipeline_stepwise(pdf_path, None, do_exec, plan_model, code_model)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()
    # Clear auto-exec after run
    if auto_exec:
        st.session_state["auto_exec"] = False

    st.success("Done")
    st.session_state.last_result = result
    if "target_plan" in result:
        st.subheader("Target Plan")
        st.write(result.get("target_plan", "(none)"))

    code = (result.get("generated_code") or result.get("code") or "")
    st.session_state.last_code = code

    if do_exec:
        st.subheader("Execution Output")
        st.write("Return code:", result.get("run_rc") if "run_rc" in result else result.get("rc"))
        with st.expander("STDOUT"):
            st.text(result.get("run_stdout") or result.get("stdout") or "")
        with st.expander("STDERR"):
            st.text(result.get("run_stderr") or result.get("stderr") or "")
        if result.get("fix_report"):
            st.subheader("Fix Attempts Summary")
            st.text(result["fix_report"]) 

        st.subheader("Generated CIF Files (recent)")
        # List CIFs modified after start_ts
        cif_files = []
        for p in Path(".").glob("*.cif"):
            try:
                if os.path.getmtime(p) >= start_ts - 1:
                    cif_files.append(p)
            except Exception:
                continue
        st.session_state.last_cifs = [str(p) for p in cif_files]
        if cif_files:
            for p in cif_files:
                with st.expander(p.name):
                    try:
                        txt = Path(p).read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        txt = "(binary or unreadable)"
                    st.code(txt[:20000], language="text")
                    with open(p, "rb") as f:
                        st.download_button(
                            f"Download {p.name}", data=f, file_name=p.name, mime="chemical/x-cif"
                        )
        else:
            st.info("No new CIF files detected. Check STDERR for issues.")

    # Code expander
    if code:
        with st.expander("Show Generated Code"):
            st.code(code, language="python")
            st.download_button(
                "Download generated_ase.py",
                data=code,
                file_name="generated_ase.py",
                mime="text/x-python",
            )
    else:
        st.info("No code generated.")

    if result.get("mp_validation"):
        st.subheader("Materials Project Validation")
        st.json(result["mp_validation"])

# If we have last results but not running now, still show for convenience
elif st.session_state.last_result:
    lr = st.session_state.last_result
    st.info("Showing last run results. Adjust options and run again as needed.")
    if lr.get("target_plan"):
        st.subheader("Target Plan (last run)")
        st.write(lr.get("target_plan"))
    if st.session_state.last_cifs:
        st.subheader("Generated CIF Files (last run)")
        for p_str in st.session_state.last_cifs:
            p = Path(p_str)
            if p.exists():
                with st.expander(p.name):
                    try:
                        txt = p.read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        txt = "(binary or unreadable)"
                    st.code(txt[:20000], language="text")
                    with open(p, "rb") as f:
                        st.download_button(
                            f"Download {p.name}", data=f, file_name=p.name, mime="chemical/x-cif"
                        )
    if st.session_state.last_code:
        with st.expander("Show Generated Code (last run)"):
            st.code(st.session_state.last_code, language="python")

st.divider()
st.subheader("Conversation")
for m in st.session_state.conversation:
    role = m.get("role", "assistant")
    st.markdown(f"**{role.title()}:** {m.get('content','')}")

user_msg = st.text_input("Ask a follow-up or refine the request")
colA, colB = st.columns(2)
with colA:
    send = st.button("Send")
with colB:
    regen = st.button("Regenerate Code from Conversation")

if send and user_msg.strip():
    st.session_state.conversation.append({"role": "user", "content": user_msg.strip()})

if regen:
    # Build a consolidated prompt from the conversation and any suggested choice
    convo_text = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.conversation])
    extra = ("\n\nConstraints:\n" + constraints_text) if constraints_text else ""
    combined_prompt = (final_prompt or "Generate CIFs from the paper") + extra + "\n" + convo_text
    with st.spinner("Regenerating code using conversation contextâ€¦"):
        result = generate_and_fix_code_v2(
            user_prompt=combined_prompt,
            paper_text=st.session_state.paper_text or "",
            code_model=code_model,
            max_iters=3,
        )
    st.session_state.conversation.append({
        "role": "assistant",
        "content": f"Regenerated code (rc={result.get('rc')}), see above for outputs."
    })

st.divider()
st.subheader("Analyze Figures from Paper")
colF1, colF2 = st.columns(2)
with colF1:
    if st.button("Extract Figures"):
        if not pdf_path:
            st.error("Select or upload a PDF above first.")
        else:
            with st.spinner("Extracting figures and captions..."):
                try:
                    figs = extract_figures(pdf_path)
                    st.session_state.figures = figs
                    st.success(f"Found {len(figs)} figures.")
                except Exception as e:
                    st.error(f"Figure extraction failed: {e}")
with colF2:
    selected_cif = None
    if st.session_state.last_cifs:
        selected_cif = st.selectbox("CIF to compare", st.session_state.last_cifs)
    else:
        st.caption("Run a pipeline first to generate CIFs, or drop a CIF into repo root.")

if st.session_state.figures:
    options = [f"Page {f.page_index+1}: {Path(f.image_path).name}" for f in st.session_state.figures]
    idx = st.selectbox("Pick a figure", list(range(len(options))), format_func=lambda i: options[i])
    fig = st.session_state.figures[idx]
    st.image(fig.image_path, caption=fig.caption or "(no caption)", use_column_width=True)

    if st.button("Detect atoms in figure"):
        with st.spinner("Detecting atomic coordinates in imageâ€¦"):
            try:
                coords = run_tem_to_atom_coords(fig.image_path)
                st.session_state.fig_coords[fig.image_path] = coords
                st.success(f"Detected {len(coords)} candidate atomic sites.")
            except Exception as e:
                st.error(f"Detection failed: {e}")

    coords = st.session_state.fig_coords.get(fig.image_path)
    if coords and selected_cif:
        if st.button("Compare detected atoms to selected CIF"):
            with st.spinner("Comparing image-derived coordinates to CIFâ€¦"):
                try:
                    res = compare_image_coords_to_cif(selected_cif, coords)
                    st.subheader("Image vs CIF comparison")
                    st.json(res)
                    if res.get("pass"):
                        st.success("Within margin of error.")
                    else:
                        st.warning("Outside margin of error â€” consider refining constraints or code.")
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
# Default ""search all"" prompt text for non-coders
DEFAULT_SEARCH_ALL_PROMPT = (
    "I am interested in investigating the structures defined in this paper using atomistic simulation. "
    "This paper contains TEM results with structural information. Determine the structures of interest in this paper. "
    "The structures of interest should be related to the main hypothesis of the paper. "
    "Next, construct ase.atoms objects for the systems of interest. "
    "Write a python script that constructs the ase.atoms objects for the systems of interest. "
    "This python script should create each ase.atoms object and write the object to a CIF file in the current working directory with a descriptive filename and .cif extension using the ase.io.write(<filename>,atoms_object,format='vasp'). "
    "Note these ase.atoms objects will be the starting point of a simulation (either DFT or MD). "
    "In the event there are certain degrees of freedom that are unclear or poorly defined in the paper, it may be useful to produce structures that sweep over several reasonable values."
)

if \"prompt_input\" not in st.session_state:
    st.session_state.prompt_input = DEFAULT_SEARCH_ALL_PROMPT








