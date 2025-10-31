import argparse
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from .graph import run_graph, load_paper, plan_targets, synthesize_code
except ImportError:
    # Allow execution as a standalone script when src/ isn't installed as a package.
    project_root = Path(__file__).resolve().parent.parent
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))
    from src.graph import run_graph, load_paper, plan_targets, synthesize_code  # type: ignore


def _slugify_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return cleaned.strip("_") or "paper"


def _prepare_run_directory(pdf_path: str) -> Path:
    base_dir = Path(
        os.environ.get(
            "ATOMBRIDGE_RUN_ROOT",
            Path(__file__).resolve().parent.parent / "runs",
        )
    )
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    paper_stem = Path(pdf_path).stem
    slug = _slugify_name(paper_stem)
    candidate = f"{timestamp}_{slug}"
    run_dir = base_dir / candidate
    counter = 2
    while run_dir.exists():
        run_dir = base_dir / f"{candidate}_{counter:02d}"
        counter += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Run TEM->ASE->CIF pipeline")
    parser.add_argument("--pdf", required=True, help="Path to the paper PDF")
    parser.add_argument("--notes", default=None, help="Optional user notes/preferences")
    parser.add_argument("--skip-exec", action="store_true", help="Stop after code synthesis (do not execute)")
    parser.add_argument("--plan-model", default=None, help="Model for planning (default env MODEL_PLAN_NAME or gemini-2.5-flash)")
    parser.add_argument("--code-model", default=None, help="Model for codegen (default env MODEL_CODE_NAME or gemini-2.5-pro)")
    parser.add_argument("--save-code", default=None, help="Optional path to save generated code (e.g., generated_ase.py)")
    parser.add_argument("--mp-validate", action="store_true", help="Run Materials Project validation (requires MP_API_KEY)")
    args = parser.parse_args()

    if args.skip_exec:
        state = {"paper_path": args.pdf}
        state.update(load_paper(state))
        if args.notes:
            state["user_notes"] = args.notes
        if args.plan_model:
            state["plan_model"] = args.plan_model
        state.update(plan_targets(state))
        if args.mp_validate:
            from .mp_api import mp_api_validate_from_text
            state["mp_validation"] = mp_api_validate_from_text(state.get("target_plan",""))
        if args.code_model:
            state["code_model"] = args.code_model
        state.update(synthesize_code(state))
        print("--- Target Plan ---\n", state.get("target_plan", ""))
        print("\n--- Generated Code (first 400 chars) ---\n", state.get("generated_code", "")[:400])
        if args.save_code and state.get("generated_code"):
            with open(args.save_code, "w", encoding="utf-8") as f:
                f.write(state["generated_code"])
            print(f"Saved generated code to {args.save_code}")
        return

    run_dir: Optional[Path] = None
    if not args.skip_exec:
        run_dir = _prepare_run_directory(args.pdf)
        os.environ["ATOMBRIDGE_CIF_OUTPUT_DIR"] = str(run_dir)
        print(f"Writing generated .cif files to {run_dir}")

    final: dict = {}
    try:
        final = run_graph(args.pdf, args.notes, plan_model=args.plan_model, code_model=args.code_model)
    finally:
        if run_dir:
            os.environ.pop("ATOMBRIDGE_CIF_OUTPUT_DIR", None)

    print("Return code:", final.get("run_rc"))
    if args.mp_validate:
        print("MP validation:", final.get("mp_validation"))
    if final.get("generated_code") and args.save_code:
        with open(args.save_code, "w", encoding="utf-8") as f:
            f.write(final["generated_code"])
        print(f"Saved generated code to {args.save_code}")
    if run_dir:
        cif_files = sorted(run_dir.glob("*.cif"))
        if cif_files:
            print("\nSaved CIF files:")
            for path in cif_files:
                print(f" - {path.name}")
        else:
            print(f"\nNo .cif files were generated in {run_dir}")
    print("\nSTDOUT (truncated):\n", (final.get("run_stdout") or "")[:1000])
    print("\nSTDERR (truncated):\n", (final.get("run_stderr") or "")[:1000])


if __name__ == "__main__":
    main()
