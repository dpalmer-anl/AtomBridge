from __future__ import annotations

import os
import tempfile
import textwrap
from pathlib import Path

from src.utils_paper_and_code import (
    build_constraints_prompt,
    extract_candidates_from_texts,
    extract_code,
    is_llm_quota_error,
    run_code,
)


def test_extract_code_prefers_python_block():
    sample = textwrap.dedent(
        """
        Some context

        ```python
        print("hello")
        ```

        ```python
        print("second")
        ```
        """
    )
    assert extract_code(sample) == 'print("hello")'


def test_extract_code_falls_back_to_raw_text():
    sample = "No fenced block here"
    assert extract_code(sample) == sample


def test_run_code_executes_and_captures_stdout():
    # run_code writes to a NamedTemporaryFile under the system temp dir.
    # Ensure subprocess outputs propagate back to the caller.
    stdout, stderr, rc = run_code('print("atombridge")')
    assert stdout.strip() == "atombridge"
    assert stderr.strip() == ""
    assert rc == 0


def test_run_code_respects_cif_output_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "cif_run"
        os.environ["ATOMBRIDGE_CIF_OUTPUT_DIR"] = str(run_dir)
        try:
            code = textwrap.dedent(
                """
                import pathlib

                cif_path = pathlib.Path("example.cif")
                cif_path.write_text("test")
                print(pathlib.Path.cwd())
                """
            )
            stdout, stderr, rc = run_code(code)
        finally:
            os.environ.pop("ATOMBRIDGE_CIF_OUTPUT_DIR", None)
        assert rc == 0
        assert stderr.strip() == ""
        assert stdout.strip() == str(run_dir.resolve())
        assert (run_dir / "example.cif").is_file()


def test_is_llm_quota_error_detects_common_messages():
    for message in [
        "Quota exceeded for this project",
        "Rate limit hit (HTTP 429)",
        "Billing issue detected",
    ]:
        assert is_llm_quota_error(RuntimeError(message))


def test_build_constraints_prompt_formats_available_fields():
    prompt = build_constraints_prompt(
        {
            "composition": "LiCoO2",
            "space_group": "R-3m",
            "a": 2.81,
            "b": 2.81,
            "c": 14.05,
            "alpha": 90,
            "beta": 90,
            "gamma": 120,
            "d_spacings": [2.47, 4.80],
            "supercell": (2, 2, 1),
            "defects": ["oxygen vacancy"],
            "grain_boundary": True,
            "gb_description": "tilt boundary",
        }
    )
    assert "Composition: LiCoO2" in prompt
    assert "Space group: R-3m" in prompt
    assert "Lattice constants: a=2.81" in prompt
    assert "b=2.81" in prompt
    assert "c=14.05" in prompt
    assert "Lattice angles: alpha=90" in prompt
    assert "Observed d-spacings: 2.47" in prompt
    assert "Supercell: 2x2x1" in prompt
    assert "Defects: oxygen vacancy" in prompt
    assert "Grain boundary: tilt boundary" in prompt


def test_extract_candidates_from_texts_ranks_by_weight():
    texts = [
        "The LiCoO2 layered structure exhibits defects.",
        "Co3O4 spinel is also observed alongside LiCoO2.",
        "LiCoO2 appears again with vacancy ordering.",
    ]
    candidates = extract_candidates_from_texts(texts)
    assert candidates[0]["formula"] == "LiCoO2"
    assert candidates[0]["count"] >= candidates[1]["count"]
