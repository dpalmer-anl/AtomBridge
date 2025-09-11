# TEM → CIF Pipeline (LangGraph, ASE + spglib + abTEM)

This project adds a new package `tem2cif` to build a **LangGraph-based workflow** for reconstructing crystallographic structures from TEM papers.  
It is designed for Cursor + Codex development, with modular nodes that can be iteratively filled.

---

## Why ASE (not pymatgen)

- **ASE** → structure build/edit, CIF I/O, strains/supercells (primary library here).  
- **spglib** → symmetry finding & cell standardization (works directly with ASE).  
- **abTEM** → diffraction + HRTEM simulation.  
- **OpenCV / SciPy** → image FFT + peak detection.  
- **EasyOCR/Tesseract** → remove overlays/scale-bar text.  
- **CLIP (optional)** → caption ↔ panel linking.

---

## Project Structure

tem2cif/
init.py
config.py
state.py # TypedDict schema for shared state
graph.py # LangGraph DAG assembly
run.py # CLI (typer)

io/
pdf_ingest.py # ingest_paper
export.py # choose_export

vision/
link_text_to_images.py
choose_best_image.py
segment_tem.py
filter_clean.py
measure_fft.py

llm/
prompts.py
text_priors.py
synthesize_description.py
ask_user_confirm.py
revise_description.py

structures/
gen_structures.py
tier1_fft_validate.py
tier2_realspace_validate.py
auto_refine_validate.py

utils/
scoring.py
plotting.py
ocr.py
clip.py

tests/
test_schemas.py


---

## Workflow Outline

1. **Ingest** paper PDF → extract text, captions, and figures.  
2. **Link text ↔ images** → match references (e.g. “Fig. 3b”) to panels using regex, CLIP, OCR keywords.  
3. **Choose best panel** → rank by annotation penalty + lattice signal strength.  
4. **Text priors** → extract composition, SG, cell, d-spacings (LLM).  
5. **Image metrics** → FFT peak detection → d, θ, zone-axis candidates.  
6. **Synthesize description** → NL summary + `description.json`.  
7. **User confirm loop** → accept / edit / refocus.  
8. **Generate structures (ASE)** → candidate CIFs with small lattice strains.  
9. **Tier-1 validation (FFT)** → abTEM diffraction vs experimental FFT peaks.  
10. **Tier-2 validation (real-space)** → abTEM HRTEM vs experimental fringes/kinks.  
11. **Auto refine** → spglib symmetry, density/stoich check, quick powder XRD.  
12. **Export** → `final.cif`, `report.md`, `trace.json`.

---

## Data Contracts

### image_metrics.json
```json
{
  "scale_A_per_px": 0.020,
  "zone_axis_candidates": ["[110]", "[111]"],
  "peaks": [
    {"d_A": 2.04, "theta_deg": 31.0, "intensity": 1.0},
    {"d_A": 1.77, "theta_deg": 91.2, "intensity": 0.6}
  ],
  "uncertainties": {"d_A": 0.04, "theta_deg": 2.0},
  "provenance": {"fig": "3b", "panel_bbox": [100,60,512,512], "quality_score": 0.86}
}

**description.json**
'''json
{
  "formula": "Co",
  "spacegroup": "Fm-3m",
  "cell": {"a": 3.54, "b": 3.54, "c": 3.54, "alpha": 90, "beta": 90, "gamma": 90},
  "Z": 4,
  "expected_d": [{"hkl": "111", "d_A": 2.05}, {"hkl": "200", "d_A": 1.77}],
  "zone_axis": "[110]",
  "phase_labels": ["twisted plane near surface"],
  "evidence": {"fig": "3b", "caption_quote": "Twisted Co plane...kinks", "page": 5},
  "assumptions": ["fcc Co", "surface twist not in bulk cell"]
}
