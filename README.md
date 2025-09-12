# AtomBridge — STEM/TEM to CIF

Atombridge is a Streamlit app that helps you go from a PDF (with STEM/TEM figures) or a STEM/TEM image crop to valid crystal structures in CIF format. It combines:

- Figure extraction and LLM‑assisted figure selection from papers
- Slider‑based image cropping with lattice analysis from the crop
- ASE RAG–guided code generation to produce structures and CIFs
- Built‑in validation: 3D viewer, atom‑overlap checks, and optional M3GNET relaxation


## Quick start

1) Create an environment (Windows/macOS/Linux; Python 3.10 strongly recommended)

```bash
python -m venv .venv
. .venv/Scripts/activate    # Windows: .\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2) Optional: set your Google GenAI key for Gemini

```bash
set GOOGLE_API_KEY=your_key_here   # PowerShell: $env:GOOGLE_API_KEY='…'
```

3) Run the app

```bash
streamlit run streamlit_app.py
```

Open the URL printed by Streamlit (usually http://localhost:8501).


## What you can do

- Select a paper (upload or from `papers/`) and click “Extract Figures”.
- Pick a CIF and click “Auto‑select figure for CIF” to let the LLM choose the most relevant STEM/TEM micrograph (not plots/graphs).
- Crop a region using sliders and preview the crop.
- Click “Analyze lattice (STEM)” and enter a scale (nm per pixel). The app estimates two in‑plane lattice vectors and the inter‑vector angle.
- Click “Create CIF” to generate structures guided by paper text and (if present) your measured lattice constraints.
- Visualize in 3D (py3Dmol), check atom overlaps, and (optionally) validate stability with M3GNET.


## Key features

- LLM‑guided figure selection: Chooses only STEM/TEM micrographs using caption/page text and CIF hints.
- Robust cropping + analysis: Simple slider crop, CLAHE enhancement, multi‑strategy peak detection, and DBSCAN clustering. If that fails, a Fourier‑domain fallback recovers lattice vectors from FFT peaks.
- Improved figure extraction: `extract_figures_v2` merges nearby image regions and segments subfigures with adaptive thresholding.
- Codegen via ASE RAG: Uses Gemini with a small retrieval set from the ASE source tree to produce solid‑starter Python code and CIFs.
- Validation built‑in:
  - 3D unit‑cell viewer (py3Dmol)
  - Atom‑overlap check with pass/fail banners and per‑file min‑distance details
  - Optional M3GNET relaxation (“Validate stable structure”) with per‑file energies


## Credentials and models

- Google GenAI (Gemini): The app uses LangChain’s `init_chat_model` with provider `google_genai`. Supply `GOOGLE_API_KEY` in your environment or in the app sidebar. Default model is `gemini-2.5-flash`; you can choose `gemini-2.5-pro`.
- Materials Project API (optional): Enter your MP API key in the sidebar to enable MP validation tools.


## Environment notes

- Python: 3.10 recommended (for best compatibility with optional scientific stacks).
- Requirements: see `requirements.txt`. The core app avoids heavy scientific deps by default.
- Optional M3GNET: If you click “Validate stable structure,” the app attempts a one‑time `conda install --no-deps m3gnet` into your active conda env (requires Conda to be available). If that’s not possible, the UI will instruct you to run the command manually.
- Windows: The app forces UTF‑8 I/O to avoid charmap errors when handling non‑ASCII text (e.g., “≈”). The file watcher is set to “poll” to avoid cross‑drive errors.


## How it works (modules)

- `streamlit_app.py`: Main UI and orchestration.
- `src/figures.py`:
  - `extract_figures_v2`: Merged‑bbox figure discovery + subfigure segmentation + caption association.
  - Heuristics for `Figure.is_tem` and TEM relevance scoring.
- `src/stem_analysis.py`:
  - `measure_lattice_vectors`: CLAHE enhancement → peak detection (skimage/OpenCV/morphology) → neighbor clustering → lattice vectors; FFT fallback when needed.
  - `minimal_cif_from_lattice`: Writes a minimal 2D CIF with the measured lattice.
- `src/create_ASE_RAG.py`: ASE RAG builder; retrieves relevant ASE snippets and queries Gemini to synthesize code.
- `src/utils_paper_and_code.py`: PDF parsing, code extraction, and robust subprocess execution.
- `src/structure_checks.py`:
  - `check_atom_distances` + `distance_summary`: Per‑CIF overlap validation and min‑distance reporting.
  - `validate_m3gnet`: Optional M3GNET relaxation. Tries to auto‑install `m3gnet` into the active conda env (no deps) if missing.


## Typical workflow

1) Load a paper and extract figures.
2) If you already have a CIF, use “Auto‑select figure for CIF”. Otherwise, proceed to cropping a likely STEM region.
3) Crop with sliders and preview.
4) Analyze lattice (enter nm/pixel). If DBSCAN fails, the FFT fallback usually succeeds on periodic images.
5) Generate CIFs. The app injects your measured lattice constraints into the LLM prompt.
6) Inspect results:
   - View in 3D
   - Overlap check (green success or red details expander)
   - Optional: Validate stable structure (M3GNET)


## Troubleshooting

- “Atom‑overlap check failed …”
  - Open the warnings expander to see which atoms are too close. Regenerate with better constraints or adjust structure code.
- “M3GNET validation unavailable…”
  - Ensure you are running in a conda environment and have `conda` in PATH. The app attempts `conda install --no-deps m3gnet -p <your_env>` when you first validate. If it fails, run that command manually.
- “Lattice analysis failed: Could not find two primary lattice directions.”
  - Provide a slightly larger crop with consistent contrast and a correct scale. The FFT fallback will kick in automatically in most cases.
- Encoding errors like `charmap`
  - Handled by the app’s UTF‑8 settings; if you see any, report where it occurred.


## Contributing

Issues and PRs are welcome! Please include screenshots or sample PDFs/images when reporting figure‑extraction or lattice‑analysis problems so we can reproduce and tune the detectors.


## License

See `LICENSE`.
