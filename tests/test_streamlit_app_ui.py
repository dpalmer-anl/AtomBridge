from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from streamlit.testing.v1 import AppTest

import langchain.chat_models as chat_models
import src.create_ASE_RAG as rag_module
import src.graph as graph
import src.utils_paper_and_code as upc


def _patch_cif_glob(monkeypatch):
    original_glob = Path.glob

    def fake_glob(self, pattern):
        if self == Path(".") and pattern == "*.cif":
            return []
        return original_glob(self, pattern)

    monkeypatch.setattr(Path, "glob", fake_glob)


def _patch_llm(monkeypatch, numbered_response: str):
    def fake_init_chat_model(model_name, model_provider="google_genai", max_retries=0):
        message = SimpleNamespace(content=numbered_response)
        return SimpleNamespace(invoke=lambda prompt: message)

    monkeypatch.setattr(chat_models, "init_chat_model", fake_init_chat_model)
    monkeypatch.setattr("src.utils_paper_and_code.init_chat_model", fake_init_chat_model, raising=False)
    monkeypatch.setattr("src.graph.init_chat_model", fake_init_chat_model, raising=False)


def test_analyze_button_populates_suggestions(monkeypatch):
    suggestions = ["1. Prompt A", "2. Prompt B", "3. Prompt C"]

    monkeypatch.setattr(graph, "load_paper", lambda state: {"paper_text": "stub content"})
    _patch_llm(monkeypatch, "1. Prompt A\n2. Prompt B\n3. Prompt C")
    _patch_cif_glob(monkeypatch)

    at = AppTest.from_file("streamlit_app.py")
    at.run(timeout=10)

    at.radio[0].set_value("From papers/").run(timeout=10)
    pdf_select = next(sb for sb in at.selectbox if sb.label == "Choose from papers/")
    pdf_option = pdf_select.options[0]
    pdf_select.set_value(pdf_option).run(timeout=10)
    analyze_btn = next(b for b in at.button if b.label == "Analyze Paper & Suggest Prompts")
    analyze_btn.click().run(timeout=10)

    assert at.session_state["paper_text"] == "stub content"
    assert at.session_state["suggested_prompts"] == suggestions


def test_generate_code_button_uses_prompt_and_backend_stub(monkeypatch):
    monkeypatch.setattr(graph, "load_paper", lambda state: {"paper_text": "integration text"})
    _patch_llm(monkeypatch, "1. Prompt X\n2. Prompt Y\n3. Prompt Z")

    captured_code: dict[str, str] = {}

    def fake_run_code(code: str):
        captured_code["code"] = code
        return ("stdout", "", 0)

    monkeypatch.setattr(upc, "run_code", fake_run_code)

    rag_calls: list[str] = []

    class DummyRAG:
        def __call__(self, query: str) -> dict[str, str]:
            rag_calls.append(query)
            return {"answer": "```python\nprint('integration test')\n```"}

    monkeypatch.setattr(rag_module, "RAG_ASE", lambda model_name=None: DummyRAG())
    _patch_cif_glob(monkeypatch)

    at = AppTest.from_file("streamlit_app.py")
    at.run(timeout=10)

    at.radio[0].set_value("From papers/").run(timeout=10)
    pdf_select = next(sb for sb in at.selectbox if sb.label == "Choose from papers/")
    pdf_option = pdf_select.options[0]
    pdf_select.set_value(pdf_option).run(timeout=10)
    prompt_area = next(ta for ta in at.text_area if ta.label.startswith("Your prompt"))
    prompt_area.set_value("Create LiCoO2 structure").run(timeout=10)

    generate_btn = next(b for b in at.button if b.label == "Generate ASE Code")
    generate_btn.click().run(timeout=10)

    assert captured_code["code"] == "print('integration test')"
    result = at.session_state["last_result"]
    assert result["code"] == "print('integration test')"
    assert result["rc"] is None
    assert len(rag_calls) == 1
