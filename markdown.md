# TEM → Unit Cell → ASE Supercell → CIF Pipeline

This repo contains a minimal LangGraph-based pipeline aligned with the current local code (no external Materials Project integration yet). It follows the agreed flow:

- Text from paper (+ optional TEM info) → Primitive unit cell proposal via LLM
- Generate ASE code to build Atoms objects (RAG over ASE source)
- Create supercells and optionally modify (defects, GBs) in code
- Write outputs as `.cif` files (DFT/post-processing can happen later)

## Components

- `src/create_ASE_RAG.py`: Builds a local RAG tool on the ASE source tree to help synthesize ASE code.
- `src/utils_paper_and_code.py`: Helpers to parse PDFs (`PyPDFLoader`), extract python code blocks, and run code safely in a temp file.
- `src/graph.py`: Minimal LangGraph with nodes: load paper → plan targets → synthesize code (RAG) → run code → end.

## Assumptions

- PDF is local and readable.
- LLM (Google GenAI) is available via `langchain` and you set `GOOGLE_API_KEY`.
- ASE is installed locally; otherwise `create_ASE_RAG` can clone ASE (requires git and network).
- This graph focuses on structure generation; DFT and Materials Project enrichment are TODOs.

## Quick Start (conceptual)

1. Set environment variables:
   - `GOOGLE_API_KEY`
2. Use Python 3.11 (recommended) for best package compatibility.
   - PowerShell: `py -3.11 -m venv .venv` then `./.venv/Scripts/Activate`
   - Install deps: `pip install -r requirements.txt`
3. Prepare a local PDF under `papers/` and note the path.
4. Use the graph entry (`run_graph`) from `src/graph.py` to execute with a state: `{ "paper_path": "papers/your.pdf" }`.
5. Generated `.cif` files (if code runs successfully) are written by the generated script in the working directory.

## Next Steps (future work)

- Materials Project lookup to validate/fill lattice parameters and composition.
- Automated supercell selection and systematic defect/GB templates.
- FFT- or real-space validation and optional iterative refinement.
- DFT job preparation and queuing (run after CIF generation).
