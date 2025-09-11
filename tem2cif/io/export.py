from __future__ import annotations

import json
import os
import shutil
from typing import Any, Dict, List

from tem2cif.state import S


def _ensure_dir(d: str) -> str:
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return d


def _write_report_md(path: str, best: Dict[str, Any], state: S):
    lines = []
    lines.append("# TEM → CIF Report\n")
    lines.append("\n## Selected Candidate\n")
    lines.append(f"Path: {best.get('path','')}  \n")
    t1 = best.get("tier1", {}) or {}
    t2 = best.get("tier2", {}) or {}
    lines.append("\n## Tier-1 FFT Metrics\n")
    for k in ["precision", "recall", "f1", "drmse", "ang_deg"]:
        if k in t1:
            lines.append(f"- {k}: {t1[k]}\n")
    lines.append("\n## Tier-2 Real-space Metrics\n")
    for k in ["ssim", "fringe_err", "thickness_nm", "defocus_nm"]:
        if k in t2:
            lines.append(f"- {k}: {t2[k]}\n")
    lines.append("\n## Notes\n")
    lines.append("TEM-only validation (no XRD).\n")
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(lines)
    except Exception:
        pass


def choose_export(state: S) -> S:
    """Pick the top composite-scored CIF and write outputs."""
    scored: List[Dict[str, Any]] = state.get("rs_scored_cifs") or state.get("fft_scored_cifs") or []
    best: Dict[str, Any] = {}
    if scored:
        best = sorted(scored, key=lambda x: x.get("composite", 0.0), reverse=True)[0]
    out_dir = _ensure_dir(state.get("out_dir", "out"))

    # Write final.cif
    cif_dst = os.path.join(out_dir, "final.cif")
    src = best.get("path") if best else None
    if src and os.path.isfile(src):
        try:
            shutil.copyfile(src, cif_dst)
        except Exception:
            # fallback: write minimal CIF if copy fails
            with open(cif_dst, "w", encoding="utf-8") as f:
                f.write("data_generated\n_symmetry_space_group_name_H-M 'P1'\n")
    else:
        with open(cif_dst, "w", encoding="utf-8") as f:
            f.write("data_generated\n_symmetry_space_group_name_H-M 'P1'\n")

    # Write report.md
    rpt_path = os.path.join(out_dir, "report.md")
    _write_report_md(rpt_path, best, state)

    # Write trace.json
    trace_path = os.path.join(out_dir, "trace.json")
    trace = {
        "provenance": {
            "image_metrics": state.get("image_metrics", {}),
            "draft_description": state.get("draft_description", {}),
            "final_description": state.get("final_description", {}),
            "user_feedback": state.get("user_feedback", {}),
        },
        "scores": scored,
        "focus": state.get("focus_bundle", {}),
    }
    try:
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace, f, indent=2)
    except Exception:
        pass

    state["export"] = {"cif": cif_dst, "report": rpt_path, "trace": trace_path}
    return state
