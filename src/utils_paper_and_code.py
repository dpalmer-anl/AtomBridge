import re
import subprocess
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import init_chat_model


def parse_pdf(pdf_filename: str) -> str:
    """Load a PDF using PyPDFLoader and return concatenated page text."""
    loader = PyPDFLoader(pdf_filename)
    docs = loader.load()
    return "\n\n".join([d.page_content for d in docs])


def extract_code(text: str) -> str:
    """Extract the first Python code block from a markdown string; fallback to raw text."""
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def run_code(code: str) -> tuple[str, str, int]:
    """Run code in a temp .py file; return (stdout, stderr, exit_code)."""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write(code)
        f.flush()
        result = subprocess.run(["python", f.name], capture_output=True, text=True)
    # Ensure outputs are handled as UTF-8 to avoid charmap errors on Windows
    try:
        out = result.stdout.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
        err = result.stderr.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        out, err = result.stdout, result.stderr
    return out, err, result.returncode


def suggest_prompts_from_paper(paper_text: str, model_name: str = "gemini-2.5-flash") -> list[str]:
    """Use an LLM to propose 3 likely prompts based on the paper.
    Returns a list of 3 concise, user-facing prompts.
    """
    llm = init_chat_model(model_name, model_provider="google_genai", max_retries=0)
    prompt = (
        "You analyze a materials science paper containing TEM-based structure discussion.\n"
        "Propose three practical user prompts to generate CIFs with ASE based on this paper.\n"
        "Make them short, action-oriented, and distinct (e.g., extract all structures, specific systems, variations).\n"
        "Return only a numbered list of three prompts.\n\n"
        f"Paper excerpt:\n{paper_text[:6000]}"
    )
    msg = llm.invoke(prompt)
    # naive parse: splitlines and take first 3 non-empty
    lines = [l.strip("- ") for l in msg.content.splitlines() if l.strip()]
    out = []
    for l in lines:
        # strip leading numbering
        out.append(l.split(" ", 1)[1] if l[:2].isdigit() and " " in l else l)
        if len(out) == 3:
            break
    # ensure 3 outputs
    while len(out) < 3:
        out.append("Extract all structures referenced in the paper to CIF files.")
    return out[:3]


def is_llm_quota_error(exc: Exception | str) -> bool:
    """Heuristically detect quota/rate-limit/billing errors from LLM providers."""
    msg = str(exc).lower()
    triggers = [
        "quota", "rate limit", "429", "resource has been exhausted",
        "insufficient", "billing", "out of tokens", "credit", "exceeded"
    ]
    return any(t in msg for t in triggers)


def extract_candidates_from_texts(texts: list[str]) -> list[dict]:
    """Heuristically extract composition formulas and structure keywords from texts.
    Returns list of dicts: {formula, keywords, count, weight} sorted by weight desc.
    """
    import re
    from collections import Counter, defaultdict

    # Regex: tokens like LiCoO2, Co3O4, SrTiO3, etc. At least 2 elements
    formula_pat = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b")
    keyword_list = [
        "spinel", "layered", "rocksalt", "perovskite", "monolayer",
        "vacancy", "defect", "grain boundary", "GB", "FFT"
    ]

    formula_hits = Counter()
    keyword_hits = Counter()
    formula_in_text = defaultdict(set)  # formula -> set(index)

    for idx, t in enumerate(texts):
        lower = (t or "").lower()
        for m in formula_pat.findall(t or ""):
            formula_hits[m] += 1
            formula_in_text[m].add(idx)
        for kw in keyword_list:
            if kw in lower:
                keyword_hits[kw] += 1

    candidates = []
    for f, c in formula_hits.items():
        # Keywords near formulas are unknown; approximate using global keyword frequency
        kws = [kw for kw, kc in keyword_hits.items() if kc > 0]
        # weight: formula frequency + emphasis for appearing in many separate texts (figures/pages)
        spread = len(formula_in_text[f])
        weight = c + 0.5 * spread + 0.25 * len(kws)
        candidates.append({"formula": f, "keywords": kws, "count": c, "weight": weight})

    # sort by weight desc, then count desc, then formula
    candidates.sort(key=lambda d: (-d["weight"], -d["count"], d["formula"]))
    return candidates


def build_constraints_prompt(constraints: dict) -> str:
    """Format a constraints dictionary into a concise prompt snippet.
    Expected keys (all optional):
    - composition: str
    - space_group: str|int
    - a,b,c: float (Ã…)
    - alpha,beta,gamma: float (deg)
    - d_spacings: list[float]
    - supercell: tuple[int,int,int]
    - defects: list[str]
    - defect_details: str
    - grain_boundary: bool
    - gb_description: str
    """
    lines = []
    if constraints.get("composition"):
        lines.append(f"Composition: {constraints['composition']}")
    if constraints.get("space_group"):
        lines.append(f"Space group: {constraints['space_group']}")
    # lattice
    abc = [constraints.get("a"), constraints.get("b"), constraints.get("c")]
    angles = [constraints.get("alpha"), constraints.get("beta"), constraints.get("gamma")]
    if any(v for v in abc):
        a, b, c = abc
        parts = []
        if a: parts.append(f"a={a} Ã…")
        if b: parts.append(f"b={b} Ã…")
        if c: parts.append(f"c={c} Ã…")
        if parts:
            lines.append("Lattice constants: " + ", ".join(parts))
    if any(v for v in angles):
        al, be, ga = angles
        parts = []
        if al: parts.append(f"alpha={al}Â°")
        if be: parts.append(f"beta={be}Â°")
        if ga: parts.append(f"gamma={ga}Â°")
        if parts:
            lines.append("Lattice angles: " + ", ".join(parts))
    # d-spacings
    dvals = constraints.get("d_spacings") or []
    if dvals:
        try:
            d_fmt = ", ".join(f"{float(x)} Ã…" for x in dvals)
        except Exception:
            d_fmt = ", ".join(map(str, dvals))
        lines.append(f"Observed d-spacings: {d_fmt}")
    # supercell
    if constraints.get("supercell"):
        sc = constraints["supercell"]
        lines.append(f"Supercell: {sc[0]}x{sc[1]}x{sc[2]}")
    # defects
    if constraints.get("defects"):
        lines.append("Defects: " + ", ".join(constraints["defects"]))
    if constraints.get("defect_details"):
        lines.append("Defect details: " + constraints["defect_details"])
    # GB
    if constraints.get("grain_boundary"):
        desc = constraints.get("gb_description") or "yes"
        lines.append("Grain boundary: " + desc)

    if not lines:
        return ""
    return "\n".join(lines)

