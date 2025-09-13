import argparse
import json
import os
from typing import Optional

from .graph import run_graph, load_paper, plan_targets, synthesize_code


def main():
    parser = argparse.ArgumentParser(description="Run TEM→ASE→CIF pipeline")
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

    final = run_graph(args.pdf, args.notes, plan_model=args.plan_model, code_model=args.code_model)
    print("Return code:", final.get("run_rc"))
    if args.mp_validate:
        print("MP validation:", final.get("mp_validation"))
    if final.get("generated_code") and args.save_code:
        with open(args.save_code, "w", encoding="utf-8") as f:
            f.write(final["generated_code"])
        print(f"Saved generated code to {args.save_code}")
    print("\nSTDOUT (truncated):\n", (final.get("run_stdout") or "")[:1000])
    print("\nSTDERR (truncated):\n", (final.get("run_stderr") or "")[:1000])


if __name__ == "__main__":
    main()
