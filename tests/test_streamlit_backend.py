from __future__ import annotations

import types

import streamlit_app as app


class DummyRAG:
    def __init__(self, answers: list[str]):
        self._answers = answers
        self.calls: list[str] = []
        self._idx = 0

    def __call__(self, query: str) -> dict[str, str]:
        self.calls.append(query)
        answer = self._answers[min(self._idx, len(self._answers) - 1)]
        self._idx += 1
        return {"answer": answer}


def test_generate_and_fix_code_v2_returns_on_first_success(monkeypatch):
    rag = DummyRAG(["```python\nprint('ok')\n```"])
    monkeypatch.setattr(app, "RAG_ASE", lambda model_name: rag)

    def fake_run_code(code: str):
        assert code == "print('ok')"
        return ("ok\n", "", 0)

    monkeypatch.setattr(app, "run_code", fake_run_code)

    result = app.generate_and_fix_code_v2("build LiCoO2", "paper text", "gemini")

    assert result["rc"] == 0
    assert result["stdout"].strip() == "ok"
    assert result["iterations"] == 1
    assert result["history"][0]["rc"] == 0
    assert "success" in result["fix_report"]


def test_generate_and_fix_code_v2_retries_on_failure(monkeypatch):
    rag = DummyRAG(
        [
            "```python\nraise RuntimeError('fail')\n```",
            "```python\nprint('fixed')\n```",
        ]
    )
    monkeypatch.setattr(app, "RAG_ASE", lambda model_name: rag)

    outcomes = [
        ("", "Traceback: failure", 1),
        ("fixed\n", "", 0),
    ]

    def fake_run_code(code: str):
        idx = fake_run_code.call_count
        fake_run_code.call_count += 1
        return outcomes[idx]

    fake_run_code.call_count = 0  # type: ignore[attr-defined]
    monkeypatch.setattr(app, "run_code", fake_run_code)

    result = app.generate_and_fix_code_v2("task", "paper", "gemini")

    assert result["rc"] == 0
    assert result["iterations"] == 2
    assert result["history"][0]["rc"] == 1
    assert any("error" in line.lower() for line in result["fix_report"].splitlines())


def test_run_pipeline_stepwise_composes_state(monkeypatch):
    observed = []

    def fake_load(state):
        observed.append(("load", dict(state)))
        return {"paper_text": "pdf content"}

    def fake_plan(state):
        assert state["paper_text"] == "pdf content"
        observed.append(("plan", dict(state)))
        return {"target_plan": "make LiCoO2"}

    def fake_synthesize(state):
        assert state["target_plan"] == "make LiCoO2"
        observed.append(("synth", dict(state)))
        return {"rag_answer": "answer", "generated_code": "print('hi')"}

    def fake_run(state):
        assert state["generated_code"] == "print('hi')"
        observed.append(("run", dict(state)))
        return {"run_stdout": "hi\n", "run_stderr": "", "run_rc": 0}

    monkeypatch.setattr(app, "load_paper", fake_load)
    monkeypatch.setattr(app, "plan_targets", fake_plan)
    monkeypatch.setattr(app, "synthesize_code", fake_synthesize)
    monkeypatch.setattr(app, "run_generated_code", fake_run)

    result = app.run_pipeline_stepwise("paper.pdf", "notes", True, "plan-model", "code-model")

    assert result["paper_path"] == "paper.pdf"
    assert result["paper_text"] == "pdf content"
    assert result["target_plan"] == "make LiCoO2"
    assert result["generated_code"] == "print('hi')"
    assert result["run_rc"] == 0
    assert [step for step, _ in observed] == ["load", "plan", "synth", "run"]
