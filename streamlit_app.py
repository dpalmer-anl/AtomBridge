import os
import time
from pathlib import Path
import streamlit as st
# Avoid cross-drive file watcher exceptions on Windows when external libs write temp files
try:
    st.set_option("server.fileWatcherType", "poll")
except Exception:
    pass
# Compatibility shim for streamlit_drawable_canvas on Streamlit>=1.49:
# Older versions of the canvas lib call streamlit.elements.image.image_to_url,
# which no longer exists. We provide a drop-in that stores the image in
# Streamlit's MediaFileManager and returns a relative URL (e.g., /media/...).
try:
    import streamlit.elements.image as _st_image_mod  # type: ignore
    if not hasattr(_st_image_mod, "image_to_url"):
        import io, hashlib
        from typing import Any
        import PIL.Image as _PILImage
        import numpy as _np
        from streamlit.runtime import get_instance as _get_rt

        def _to_png_bytes(img: Any) -> bytes:
            # Try treating it as a PIL image first
            try:
                bio = io.BytesIO()
                img.save(bio, format="PNG")
                return bio.getvalue()
            except Exception:
                pass
            # Try converting from numpy-like
            try:
                arr = _np.array(img)
                if arr.ndim == 2:
                    pil = _PILImage.fromarray(arr)
                elif arr.ndim == 3:
                    if arr.dtype != _np.uint8:
                        arr = _np.clip(arr, 0, 255).astype(_np.uint8)
                    # Drop alpha if present for PNG RGB
                    if arr.shape[2] == 4:
                        arr = arr[:, :, :3]
                    pil = _PILImage.fromarray(arr)
                else:
                    raise TypeError(f"Unsupported array shape: {arr.shape}")
                bio = io.BytesIO()
                pil.save(bio, format="PNG")
                return bio.getvalue()
            except Exception as _e:
                raise TypeError(f"Unsupported image type for image_to_url shim: {type(img)}") from _e

        def image_to_url(img, width, clamp, channels, output_format, image_id):  # type: ignore
            data = _to_png_bytes(img)
            coord = image_id or f"drawable-canvas-bg-{hashlib.md5(data).hexdigest()}"
            rt = _get_rt()
            url = rt.media_file_mgr.add(data, "image/png", coord)
            return url

        _st_image_mod.image_to_url = image_to_url  # type: ignore[attr-defined]
except Exception:
    pass

from src.graph import load_paper, plan_targets, synthesize_code, run_generated_code
from src.utils_paper_and_code import suggest_prompts_from_paper, extract_code, run_code, build_constraints_prompt, is_llm_quota_error
from src.create_ASE_RAG import RAG_ASE
from src.mp_api import mp_api_validate_from_text
from src.figures import extract_figures, run_tem_to_atom_coords, crop_image, split_into_grid, parse_subfigure_labels, overlay_points, heatmap_overlay
import streamlit.components.v1 as components
from src.validation import compare_image_coords_to_cif


UPLOAD_DIR = Path("_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Default "search all" prompt text for non-coders
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
                fix_lines.append(f"Iteration {h['iteration']}: error while generating supercell - {err_snip}. Tried to regenerate code with error context.")
            else:
                fix_lines.append(f"Iteration {h['iteration']}: error - {err_snip}. Tried to regenerate code with error context.")

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
                    f"Iteration {h['iteration']}: error while generating supercell - {err_snip}. Tried to regenerate code with error context."
                )
            else:
                fix_lines.append(
                    f"Iteration {h['iteration']}: error - {err_snip}. Tried to regenerate code with error context."
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
st.title("Atombridge")
st.caption("STEM to CIF")

with st.sidebar:
    st.header("Environment")
    key_present = bool(os.getenv("GOOGLE_API_KEY"))
    st.write("GOOGLE_API_KEY:", "set" if key_present else "missing")
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
    st.subheader("Model")
    selected_model = st.selectbox(
        "Choose model",
        options=["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
        help="Pick Pro or Flash. Planning uses Flash to save quota.",
    )
    # Effective models
    plan_model = "gemini-2.5-flash"
    code_model = selected_model
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
if "figure_idx" not in st.session_state:
    st.session_state.figure_idx = 0

analyze = st.button("Analyze Paper & Suggest Prompts")
if analyze:
    if not pdf_path:
        st.error("Please provide a PDF (upload or pick from papers/)")
        st.stop()
    with st.spinner("Reading paper and proposing prompts..."):
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
    st.subheader('Suggested Prompts (click to run)')
    # Build extended prompts list: only compositions tied to TEM figures
    try:
        from src.utils_paper_and_code import extract_candidates_from_texts as _extract
        tem_texts = []
        for f in st.session_state.figures:
            if getattr(f, 'is_tem', False):
                try:
                    tem_texts.append((f.caption or '') + '\n' + (getattr(f, 'page_text', '') or ''))
                except Exception:
                    continue
        tem_candidates = _extract(tem_texts) if tem_texts else []
    except Exception:
        tem_candidates = []

    # Filter original LLM suggestions to those that mention any TEM-tied formula
    def _mentions_any_formula(text: str, formulas: list[str]) -> bool:
        t = (text or '').lower()
        for f in formulas:
            if f.lower() in t:
                return True
        return False

    tem_formulas = [c.get('formula') for c in tem_candidates if c.get('formula')]
    filtered_suggestions = list(st.session_state.suggested_prompts)
    if tem_formulas:
        filtered_suggestions = [p for p in st.session_state.suggested_prompts if _mentions_any_formula(p, tem_formulas)]

    # Compose extended prompts from filtered suggestions + all TEM-tied candidates
    extended = list(filtered_suggestions)
    for cand in tem_candidates:
        form = cand.get('formula')
        if not form:
            continue
        kws = cand.get('keywords') or []
        kw_text = ' '.join(sorted(set(kws) & {'spinel','layered','rocksalt','perovskite'})).strip()
        desc = (' ' + kw_text) if kw_text else ''
        prompt = f"Generate CIFs for {form}{desc} consistent with the paper's TEM analysis. Include reasonable variants if parameters are ambiguous."
        if all(prompt.lower() != p.lower() for p in extended):
            extended.append(prompt)

    # Batch-generate all TEM-linked prompts
    def _slugify(t: str) -> str:
        import re
        s = t.lower()
        s = re.sub('[^a-z0-9]+', '-', s).strip('-')
        return s[:40] if len(s) > 40 else s

    cola, colb = st.columns([2,3])
    with cola:
        append_constraints = st.checkbox(
            'Append constraints to all',
            value=False,
            help='Append current constraints (composition, lattice, defects) to each batch prompt.'
        )
    with colb:
        if st.button('Generate TEM-linked CIFs (all)'):
            if not extended:
                st.warning('No TEM-linked prompts available. Extract figures first or analyze paper.')
            else:
                if st.session_state.paper_text is None:
                    st.error('Please analyze the paper first to load text context.')
                else:
                    gen_count, errors = 0, []
                    seen_tags = set()
                    for j, prompt_text in enumerate(extended):
                        tag = _slugify(prompt_text)
                        if tag in seen_tags:
                            continue
                        seen_tags.add(tag)
                        prompt_full = prompt_text
                        if append_constraints:
                            ctext = st.session_state.get('constraints_text')
                            if ctext:
                                prompt_full = (prompt_full + '\n\nConstraints:\n' + ctext).strip()
                        suffix = (
                            '\n\nInstructions: For each structure, write to a distinct CIF file with a descriptive,'
                            f" hyphenated name that includes '{tag}' to avoid collisions."
                        )
                        try:
                            with st.spinner(f'Generating CIFs for prompt {j+1}...'):
                                _ = generate_and_fix_code_v2(
                                    user_prompt=(prompt_full + suffix),
                                    paper_text=st.session_state.paper_text or '',
                                    code_model=code_model,
                                    max_iters=3,
                                )
                                gen_count += 1
                        except Exception as e:
                            errors.append(str(e))
                    new_cifs = [str(p) for p in Path('.').glob('*.cif')]
                    st.session_state.last_cifs = new_cifs
                    # Run quick overlap check on generated CIFs
                    try:
                        from ase.io import read as ase_read
                        from src.structure_checks import check_atom_distances, distance_summary
                        bads, goods = [], []
                        for p in new_cifs:
                            try:
                                atoms = ase_read(p)
                                check_atom_distances(atoms)
                                goods.append((p, distance_summary(atoms)))
                            except Exception as ee:
                                bads.append((p, str(ee)))
                        if bads:
                            st.error(f"Atom-overlap check failed for {len(bads)}/{len(new_cifs)} CIF(s). See details below.")
                            with st.expander('Warnings: Potential atom overlaps detected'):
                                for p, msg in bads:
                                    st.write(p)
                                    st.code(msg)
                        else:
                            st.success("Atom-overlap check passed for all generated CIFs.")
                            with st.expander('Overlap check details (min distances)'):
                                for p, summ in goods:
                                    st.write(f"{Path(p).name}: min distance = {summ['min_distance']:.3f} Å (cutoff {summ['cutoff']:.2f} Å)")
                    except Exception:
                        pass
                    st.success(f'Completed generation for {gen_count} TEM-linked prompts. Found {len(new_cifs)} CIFs in working directory.')
                    # Show CIF visuals like single-run
                    st.subheader('Generated CIF Files (recent)')
                    if new_cifs:
                        for p in new_cifs:
                            pth = Path(p)
                            with st.expander(pth.name):
                                try:
                                    txt = pth.read_text(encoding='utf-8', errors='ignore')
                                except Exception:
                                    txt = '(binary or unreadable)'
                                st.code(txt[:20000], language='text')
                                if st.button(f"Show 3D unit cell ({pth.name})", key=f"show3d_recent_{pth.name}"):
                                    render_cif_3d(str(pth), width=700, height=500, style='ballstick')
                                with open(pth, 'rb') as f:
                                    st.download_button(
                                        f'Download {pth.name}', data=f, file_name=pth.name, mime='chemical/x-cif'
                                    )
                        with st.expander('Optional: Validate stable structure (M3GNET, slow)'):
                            if st.button('Validate stable structure (batch)', key='m3g_batch_recent'):
                                try:
                                    from src.structure_checks import validate_m3gnet
                                    for p in new_cifs:
                                        try:
                                            e, _ = validate_m3gnet(p)
                                            st.write(f"{Path(p).name}: {e:.6f}")
                                        except Exception as ee:
                                            st.write(f"{Path(p).name}: {ee}")
                                except Exception as e:
                                    st.info(f"M3GNET validation not available: {e}")
                    else:
                        st.info('No CIF files detected. Check STDERR for issues.')
                    if errors:
                        with st.expander('Errors while generating some prompts'):
                            for e in errors:
                                st.write(e)

    # Pager/carousel over all prompts (most relevant first)
    per_page = 4
    if 'prompt_page' not in st.session_state:
        st.session_state.prompt_page = 0
    total = len(extended)
    pages = max(1, (total + per_page - 1) // per_page)
    nav1, nav2, nav3 = st.columns([1,1,6])
    with nav1:
        if st.button('◀ Prev'):
            st.session_state.prompt_page = max(0, st.session_state.prompt_page - 1)
    with nav2:
        if st.button('Next ▶'):
            st.session_state.prompt_page = min(pages - 1, st.session_state.prompt_page + 1)
    with nav3:
        st.caption(f"Page {st.session_state.prompt_page + 1} / {pages} — {total} prompts")

    start = st.session_state.prompt_page * per_page
    end = min(total, start + per_page)
    cols = st.columns(max(1, end - start))
    for i, col in enumerate(cols, start=start):
        with col:
            sp = extended[i]
            if st.button(sp, key=f'prompt_ext_{i}'):
                st.session_state['prompt_input'] = sp
                st.session_state['auto_exec'] = True
                st.rerun()
else:
    st.caption("Click 'Analyze Paper & Suggest Prompts' to get suggestions, or type your own below.")

st.subheader("Constraints (optional)")
with st.expander("Specify constraints from TEM/paper"):
    comp = st.text_input("Composition (formula)", placeholder="e.g., MoS2 or LiCoO2")
    sg = st.text_input("Space group (symbol or number)", placeholder="e.g., P6_3/mmc or 194")
    c1, c2, c3 = st.columns(3)
    with c1:
        a_val = st.number_input("a (Angstrom)", min_value=0.0, value=0.0, step=0.01)
        alpha_val = st.number_input("alpha (deg)", min_value=0.0, max_value=180.0, value=0.0, step=0.1)
    with c2:
        b_val = st.number_input("b (Angstrom)", min_value=0.0, value=0.0, step=0.01)
        beta_val = st.number_input("beta (deg)", min_value=0.0, max_value=180.0, value=0.0, step=0.1)
    with c3:
        c_val = st.number_input("c (Angstrom)", min_value=0.0, value=0.0, step=0.01)
        gamma_val = st.number_input("gamma (deg)", min_value=0.0, max_value=180.0, value=0.0, step=0.1)

    d_text = st.text_area("d-spacings (Angstroms, comma or space separated)", placeholder="e.g., 2.46, 1.42, 1.23")
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
    gb_desc = st.text_input("Grain boundary description", disabled=not gb, placeholder="e.g., tilt GB ~5 along [10-10]")

# Ensure prompt state exists (do not auto-fill)
if "prompt_input" not in st.session_state:
    st.session_state.prompt_input = ""

notes = st.text_area(
    "Your prompt (optional)",
    key="prompt_input",
    placeholder="e.g., Extract all structures and write CIFs; include 2x2 supercells and S vacancy variants.",
)

# Build constraints string
parsed_d = []
if d_text.strip():
    try:
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
st.session_state['constraints_text'] = constraints_text

final_prompt = (notes or "").strip()
if constraints_text:
    final_prompt = (final_prompt + "\n\nConstraints:\n" + constraints_text).strip()
if not final_prompt:
    st.caption("Provide a prompt or analyze the paper to pick a suggestion, and optionally add constraints.")

col1, col2 = st.columns(2)
with col1:
    dry = st.button("Generate ASE Code")
with col2:
    full = st.button("Generate .cif")

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
    with st.spinner("Running pipeline... this may take a moment on first run"):
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
        # Overlap check summary for this execution
        if cif_files:
            try:
                from ase.io import read as ase_read
                from src.structure_checks import check_atom_distances, distance_summary
                bads, goods = [] , []
                for p in cif_files:
                    try:
                        atoms = ase_read(p)
                        check_atom_distances(atoms)
                        goods.append((str(p), distance_summary(atoms)))
                    except Exception as ee:
                        bads.append((str(p), str(ee)))
                if bads:
                    st.error(f"Atom-overlap check failed for {len(bads)}/{len(cif_files)} CIF(s). See details below.")
                    with st.expander('Warnings: Potential atom overlaps detected'):
                        for p, msg in bads:
                            st.write(p)
                            st.code(msg)
                else:
                    st.success("Atom-overlap check passed for all generated CIFs.")
                    with st.expander('Overlap check details (min distances)'):
                        for p, summ in goods:
                            name = Path(p).name
                            st.write(f"{name}: min distance = {summ['min_distance']:.3f} Å (cutoff {summ['cutoff']:.2f} Å)")
            except Exception:
                pass
        if cif_files:
            for p in cif_files:
                with st.expander(p.name):
                    try:
                        txt = Path(p).read_text(encoding="utf-8", errors="ignore")
                    except Exception:
                        txt = "(binary or unreadable)"
                    st.code(txt[:20000], language='text')
                    if st.button(f"Show 3D unit cell ({p.name})", key=f"show3d_last_{p.name}"):
                        render_cif_3d(str(p), width=700, height=500, style='ballstick')
                    with open(p, "rb") as f:
                        st.download_button(
                            f"Download {p.name}", data=f, file_name=p.name, mime="chemical/x-cif"
                        )
            with st.expander('Optional: Validate stable structure (M3GNET, slow)'):
                if st.button('Validate stable structure (recent)', key='m3g_exec_recent'):
                    try:
                        from src.structure_checks import validate_m3gnet
                        for p in cif_files:
                            try:
                                e, _ = validate_m3gnet(str(p))
                                st.write(f"{p.name}: {e:.6f}")
                            except Exception as ee:
                                st.write(f"{p.name}: {ee}")
                    except Exception as e:
                        st.info(f"M3GNET validation not available: {e}")
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
        # Overlap check summary for last run
        try:
            from ase.io import read as ase_read
            from src.structure_checks import check_atom_distances
            bads = []
            for p_str in st.session_state.last_cifs:
                p = Path(p_str)
                if not p.exists():
                    continue
                try:
                    atoms = ase_read(str(p))
                    check_atom_distances(atoms)
                except Exception as ee:
                    bads.append((str(p), str(ee)))
            if bads:
                st.error(f"Atom-overlap check failed for {len(bads)}/{len(st.session_state.last_cifs)} CIF(s). See details below.")
                with st.expander('Warnings: Potential atom overlaps detected (last run)'):
                    for p, msg in bads:
                        st.write(p)
                        st.code(msg)
            else:
                st.success("Atom-overlap check passed for all last-run CIFs.")
        except Exception:
            pass
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
        with st.expander('Optional: Validate stable structure (M3GNET, slow)'):
            if st.button('Validate stable structure (last run)', key='m3g_last'):
                try:
                    from src.structure_checks import validate_m3gnet
                    for p_str in st.session_state.last_cifs:
                        p = Path(p_str)
                        if not p.exists():
                            continue
                        try:
                            e, _ = validate_m3gnet(str(p))
                            st.write(f"{p.name}: {e:.6f}")
                        except Exception as ee:
                            st.write(f"{p.name}: {ee}")
                except Exception as e:
                    st.info(f"M3GNET validation not available: {e}. To enable, install 'pymatgen' and 'm3gnet' in your environment.")
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
    with st.spinner("Regenerating code using conversation context..."):
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
                    # Try improved extractor first; fall back to baseline
                    try:
                        from src.figures import extract_figures_v2 as _extract_v2
                        figs = _extract_v2(pdf_path)
                        if not figs:
                            raise RuntimeError("no_figs_v2")
                    except Exception:
                        figs = extract_figures(pdf_path)
                    st.session_state.figures = figs
                    st.session_state.figure_idx = 0
                    st.session_state.figures_extracted = True
                    st.success(f"Found {len(figs)} figures.")
                except Exception as e:
                    st.error(f"Figure extraction failed: {e}")
with colF2:
    selected_cif = None
    # Prefer last run's CIFs; otherwise list all CIFs in working directory
    cif_candidates = list(st.session_state.get("last_cifs") or [])
    if not cif_candidates:
        try:
            cif_candidates = [str(p) for p in Path('.').glob('*.cif')]
        except Exception:
            cif_candidates = []
    if cif_candidates:
        selected_cif = st.selectbox("CIF to compare", sorted(cif_candidates))
    else:
        st.caption("Place a .cif in the repo root or generate one first to enable auto-select & comparison.")

    # Auto-select a figure that best matches the selected CIF (LLM text-aware only)
    if selected_cif and st.session_state.figures and st.button("Auto-select figure for CIF"):
        from src.validation import compare_image_coords_to_cif as _cmp
        # First attempt: LLM-based textual selection to avoid non-TEM picks
        try:
            from langchain.chat_models import init_chat_model as _init_chat_model
            cands = list(st.session_state.figures)
            if not cands:
                raise RuntimeError("no_figs")

            def _desc(fig) -> str:
                cap = (fig.caption or "").strip().replace("\n", " ")
                pg = (getattr(fig, "page_text", "") or "").strip().replace("\n", " ")
                pg = pg[:300]
                hint = "TEM?yes" if getattr(fig, "is_tem", False) else "TEM?no"
                return f"{hint} | caption: {cap[:300]} | page: {pg}"

            def _keywords_from_cif(path: str):
                kws = []
                try:
                    from ase.io import read as ase_read
                    atoms = ase_read(path)
                    form = atoms.get_chemical_formula()
                    if form:
                        kws.append(form)
                except Exception:
                    pass
                base = Path(path).stem
                for t in base.replace('-', '_').split('_'):
                    if t:
                        kws.append(t)
                return ", ".join(dict.fromkeys([k for k in kws if k]))

            cif_kws = _keywords_from_cif(selected_cif)
            model_name = "gemini-2.5-flash"
            _llm = _init_chat_model(model_name, model_provider="google_genai", max_retries=0)
            numbered = "\n".join([f"{i+1}. {_desc(f)}" for i, f in enumerate(cands)])
            prompt = (
                "You are assisting with selecting the most relevant figure for validating a crystal structure from a CIF file.\n"
                "Pick the SINGLE best figure that is a TEM/HRTEM/STEM micrograph (not plots/graphs/XRD/spectra) and most relevant to the given structure.\n"
                "Return only the number of the chosen item. If none are suitable TEM micrographs, return 'none'.\n\n"
                f"Structure context (from CIF/filename): {cif_kws}\n\n"
                "Candidate figures:\n" + numbered
            )
            _resp = _llm.invoke(prompt)
            _out = str(getattr(_resp, "content", _resp)).strip().lower()
            sel_idx = None
            import re as _re
            m = _re.search(r"(\d+)", _out)
            if m:
                sel_idx = max(1, int(m.group(1))) - 1
            elif "none" in _out:
                sel_idx = None
            if sel_idx is not None and 0 <= sel_idx < len(cands):
                pick = cands[sel_idx]
                if getattr(pick, "is_tem", False):
                    pool = [f for f in st.session_state.figures if getattr(f, 'is_tem', False)]
                    try:
                        tem_idx = pool.index(pick)
                        st.session_state.figure_idx = tem_idx
                        st.success(f"LLM selected figure #{sel_idx+1} as the best textual match.")
                        # Skip heuristic-geometry stage
                        raise SystemExit
                    except ValueError:
                        pass
                else:
                    st.info("LLM did not find a suitable TEM micrograph; no selection made.")
                    raise SystemExit
            else:
                st.info("LLM indicated no suitable TEM figures; no selection made.")
                raise SystemExit
        except SystemExit:
            # Selection already handled or explicitly skipped.
            pass
        except Exception as e:
            st.warning(f"Auto-select (LLM) failed: {e}")

tem_pool = [f for f in st.session_state.figures if getattr(f, 'is_tem', False)]
if tem_pool:
    options = [f"Page {f.page_index+1}: {Path(f.image_path).name}" for f in tem_pool]
    idx = st.selectbox("Pick a figure", list(range(len(options))), index=min(st.session_state.figure_idx, len(options)-1), format_func=lambda i: options[i])
    fig = tem_pool[idx]
    # Show selected figure image
    st.image(fig.image_path, caption=fig.caption or "(no caption)", use_container_width=True)
else:
    st.info("No relevant TEM figures found to analyze.")
    fig = None

st.subheader("Crop region to analyze")
crop_path = None
if fig is not None:
    # Slider-based crop (robust, no canvas)
    from PIL import Image as PILImage
    im = PILImage.open(fig.image_path)
    W, H = im.size
    x = st.slider("x", 0, max(1, W - 1), 0)
    y = st.slider("y", 0, max(1, H - 1), 0)
    w = st.slider("w", 1, W, min(200, W))
    h = st.slider("h", 1, H, min(200, H))
    # Optional overlay of crop box on full image
    show_overlay = st.checkbox("Show crop box on full image", value=True)
    if show_overlay:
        try:
            from PIL import ImageDraw
            overlay = im.copy()
            draw = ImageDraw.Draw(overlay)
            draw.rectangle([int(x), int(y), int(x + w), int(y + h)], outline="red", width=3)
            st.image(overlay, caption=f"Crop box x={x}, y={y}, w={w}, h={h}", use_container_width=True)
        except Exception:
            pass
    if st.button("Preview crop"):
        cpath = crop_image(fig.image_path, (int(x), int(y), int(w), int(h)))
        st.session_state["crop_preview_path"] = cpath
    if st.session_state.get("crop_preview_path"):
        cpath = st.session_state["crop_preview_path"]
        st.image(cpath, caption="Cropped preview", use_container_width=True)
        crop_path = cpath

    if crop_path and st.button("Detect atoms in selected region"):
        with st.spinner("Detecting atomic coordinates in image..."):
            try:
                coords = run_tem_to_atom_coords(crop_path)
                st.session_state.fig_coords[crop_path] = coords
                st.success(f"Detected {len(coords)} candidate atomic sites.")
                # Overlays
                if coords:
                    pts = overlay_points(crop_path, coords)
                    st.image(pts, caption="Detections overlay", use_container_width=True)
                    if st.checkbox("Show detections heatmap", value=False):
                        hm = heatmap_overlay(crop_path, coords)
                        if hm:
                            st.image(hm, caption="Detections heatmap", use_container_width=True)
            except Exception as e:
                st.error(f"Detection failed: {e}")

    # STEM lattice analysis using peak-based method (nm per pixel input)
    if crop_path:
        st.subheader("STEM lattice analysis")
        nm_per_px = st.number_input("Scale (nm per pixel)", min_value=0.0001, max_value=10.0, value=0.05, step=0.005, format="%.4f")
        c1, c2 = st.columns(2)
        with c1:
            run_lattice = st.button("Analyze lattice (STEM)")
        with c2:
            quick_min_cif = st.button("Write minimal CIF from lattice")
        if run_lattice or quick_min_cif:
            try:
                from src.stem_analysis import measure_lattice_vectors, minimal_cif_from_lattice
                res_lat = measure_lattice_vectors(crop_path, nm_per_px)
                st.session_state["last_lattice"] = res_lat
                st.success(f"a={res_lat['a_nm']:.4f} nm, b={res_lat['b_nm']:.4f} nm, gamma={res_lat['gamma_deg']:.2f} deg; atoms={res_lat['n_atoms']}")
                if res_lat.get("overlay_path"):
                    st.image(res_lat["overlay_path"], caption="Lattice detection overlay", use_container_width=True)
                if quick_min_cif:
                    out_name = Path(crop_path).with_suffix("").name + "_lattice.cif"
                    out_path = minimal_cif_from_lattice(res_lat['a_nm'], res_lat['b_nm'], res_lat['gamma_deg'], out_name)
                    st.success(f"Wrote minimal CIF: {out_path}")
                    with open(out_path, "rb") as f:
                        st.download_button(f"Download {Path(out_path).name}", data=f, file_name=Path(out_path).name, mime="chemical/x-cif")
            except Exception as e:
                st.warning(f"Lattice analysis failed: {e}")

    coords = st.session_state.fig_coords.get(crop_path or (fig.image_path if fig else None))
    # Button to create CIF from cropped region context
    if crop_path and st.button("Create CIF"):
        # Build focused prompt using caption and page text
        context = (fig.caption or "") + "\n\n" + (fig.page_text or "")
        lattice_hint = ""
        if st.session_state.get("last_lattice"):
            lat = st.session_state["last_lattice"]
            lattice_hint = (f"\n\nImage-derived lattice parameters (nm): a={lat['a_nm']:.4f}, b={lat['b_nm']:.4f}, "
                            f"gamma={lat['gamma_deg']:.2f} deg. Use these as strong constraints.")
        focus_prompt = (
            "Focus on the structure shown in the selected subfigure/crop. "
            "Use the caption and surrounding page text as context:\n" + context[:4000] + lattice_hint
        )
        user_prompt = (final_prompt + "\n\n" + focus_prompt).strip()
        with st.spinner("Generating ASE code and CIF from selected region..."):
            result2 = generate_and_fix_code_v2(
                user_prompt=user_prompt,
                paper_text=st.session_state.paper_text or "",
                code_model=code_model,
                max_iters=3,
            )
        st.session_state.last_result = result2
        # Refresh CIF list
        new_cifs = [str(p) for p in Path('.').glob('*.cif')]
        st.session_state.last_cifs = new_cifs
        # Quick overlap checks
        try:
            from ase.io import read as ase_read
            from src.structure_checks import check_atom_distances
            bads = []
            for p in new_cifs:
                try:
                    atoms = ase_read(p)
                    check_atom_distances(atoms)
                except Exception as ee:
                    bads.append((p, str(ee)))
            if bads:
                st.error(f"Atom-overlap check failed for {len(bads)}/{len(new_cifs)} CIF(s). See details below.")
                with st.expander('Warnings: Potential atom overlaps detected'):
                    for p, msg in bads:
                        st.write(p)
                        st.code(msg)
            else:
                st.success("Atom-overlap check passed for all generated CIFs.")
        except Exception:
            pass
        st.session_state.last_code = result2.get("code") or ""
        st.success("CIF generation attempt finished.")

    if coords and selected_cif:
        if st.button("Compare detected atoms to selected CIF"):
            with st.spinner("Comparing image-derived coordinates to CIF..."):
                try:
                    res = compare_image_coords_to_cif(selected_cif, coords)
                    st.subheader("Image vs CIF comparison")
                    st.json(res)
                    if res.get("pass"):
                        st.success("Within margin of error.")
                    else:
                        st.warning("Outside margin of error - consider refining constraints or code.")
                except Exception as e:
                    st.error(f"Comparison failed: {e}")

    # Optional M3GNET validation on last CIFs
    if st.session_state.get('last_cifs'):
        with st.expander('Optional: Validate last CIFs with M3GNET (slow)'):
            if st.button('Run M3GNET relaxation on last CIFs'):
                try:
                    from src.structure_checks import validate_m3gnet
                    energies = []
                    for p in st.session_state['last_cifs']:
                        try:
                            e, _ = validate_m3gnet(p)
                            energies.append((p, e))
                        except Exception as ee:
                            st.write(f"{p}: {ee}")
                    if energies:
                        st.subheader('M3GNET energies (lower is better)')
                        for p, e in energies:
                            st.write(f"{Path(p).name}: {e:.6f}")
                except Exception as e:
                    st.info(f"M3GNET validation not available: {e}")

    # Combined analysis to paper and MP
    if crop_path and st.button("Analyze to Paper and MP"):
        analysis_text = (fig.caption or "") + "\n\n" + (fig.page_text or "")
        from src.mp_api import mp_api_validate_from_text as _mp_validate
        mp_res = _mp_validate(analysis_text)
        st.subheader("Materials Project API validation")
        st.json(mp_res)
        coords2 = st.session_state.fig_coords.get(crop_path)
        if coords2 and st.session_state.last_cifs:
            st.subheader("Image vs last generated CIFs")
            for p in st.session_state.last_cifs:
                try:
                    metrics = compare_image_coords_to_cif(p, coords2)
                    st.write(p)
                    st.json(metrics)
                except Exception as e:
                    st.warning(f"Compare failed for {p}: {e}")

# 3D CIF viewers using py3Dmol
def render_cif_3d(cif_path: str, width: int = 600, height: int = 400, style: str = "stick", show_unit_cell: bool = True):
    try:
        import py3Dmol
        txt = Path(cif_path).read_text(encoding="utf-8", errors="ignore")
        view = py3Dmol.view(width=width, height=height)
        view.addModel(txt, 'cif')
        if style == "stick":
            view.setStyle({"stick": {}})
        elif style == "ballstick":
            view.setStyle({"sphere": {"scale": 0.2}, "stick": {}})
        else:
            view.setStyle({"line": {}})
        if show_unit_cell:
            try:
                view.addUnitCell()
            except Exception:
                pass
        view.zoomTo()
        html = view._make_html()
        components.html(html, height=height)
    except Exception as e:
        st.warning(f"3D viewer unavailable: {e}")
