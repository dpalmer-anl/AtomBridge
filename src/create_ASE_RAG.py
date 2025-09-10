"""
langgraph_rag_ase_local.py

If ASE is installed locally, use its source directory for RAG.
Otherwise, clone from GitLab:
https://gitlab.com/ase/ase/-/tree/master/ase?ref_type=heads

Steps:
- detect ase.__file__
- load text files from source tree
- chunk and embed with Google GenAI
- store in Chroma
- expose a LangGraph-style tool (rag_tool)
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Dict
import ase.io
from ase import Atoms
from langchain_core.tools import tool

# ---- Config ----
GIT_URL = "https://gitlab.com/ase/ase.git"
CLONE_DIR = Path("./_tmp_ase_repo")
CHROMA_PERSIST_DIR = "./chroma_ase"
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
TOP_K = 15

# ---- Environment / credentials ----
if "GOOGLE_API_KEY" not in os.environ:
    print("Warning: GOOGLE_API_KEY not set. Set it before running.", file=sys.stderr)

# ---- Detect ASE source ----
def get_ase_source_path() -> Path:
    try:
        import ase
        src_file = Path(ase.__file__).resolve()
        ase_dir = src_file.parent
        print(f"Found ASE installed at {ase_dir}")
        return ase_dir
    except ImportError:
        print("ASE not installed, cloning repo instead...")
        if not CLONE_DIR.exists():
            subprocess.check_call(["git", "clone", "--depth", "1", GIT_URL, str(CLONE_DIR)])
        return CLONE_DIR / "ase"

# ---- Read text files ----
INCLUDE_EXTS = {".py",  ".txt"}

def load_text_documents(base_dir: Path) -> List[Dict]:
    docs = []
    for p in base_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in INCLUDE_EXTS:
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                continue
            if text.strip():
                docs.append({"path": str(p.relative_to(base_dir)), "text": text})
    return docs

# ---- Build retriever ----
def build_vector_store_and_retriever(docs: List[Dict]):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_chroma import Chroma

    texts, metadatas = [], []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    for d in docs:
        chunks = splitter.split_text(d["text"])
        for i, c in enumerate(chunks):
            texts.append(c)
            metadatas.append({"source": d["path"], "chunk": i})

    if not texts:
        raise RuntimeError("No texts found to index.")

    #embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001") #requires too many credits for free version
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") #runs locally
    if Path(CHROMA_PERSIST_DIR).exists():
        print(f"Reusing cached Chroma DB at {CHROMA_PERSIST_DIR}")
        vectordb = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    else:
        print("Building new Chroma DB...")
        vectordb = Chroma.from_texts(
            texts, embedding=embeddings, metadatas=metadatas, persist_directory=CHROMA_PERSIST_DIR
        )
        vectordb.persist()
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": TOP_K})
    return retriever

# ---- QA chain ----
def make_qa_chain(retriever):
    from langchain.chat_models import init_chat_model
    from langchain.chains import RetrievalQA
    llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
    )
    return qa

# ---- Tool wrapper ----
def make_langgraph_tool(qa_chain):
    def rag_tool(query: str) -> Dict:
        out = qa_chain.invoke({"query": query})
        answer = out.get("result") or out.get("answer") or out.get("output_text")
        sources = []
        for doc in out.get("source_documents", []):
            sources.append({
                "source": doc.metadata.get("source"),
                "chunk": doc.metadata.get("chunk"),
                "preview": doc.page_content[:300]
            })
        return {"answer": answer, "sources": sources}
    return rag_tool

@tool
def run_ase_script(code: str, filename: str) -> None:
    """Run Python code that defines an ASE Atoms object as `atoms` and writes the atoms object to an extxyz file with a descriptive and unique name.
    Args:
        code: Python code to execute
        filename: descriptive and unique filename. 
        """
    local_env = {}
    exec(code, {}, local_env)
    atoms = local_env.get("atoms")
    if not isinstance(atoms, Atoms):
        raise ValueError("Script did not define an ASE Atoms object named `atoms`.")
    else:
        ase.io.write(filename,atoms,format="vasp")

# ---- Main ----
def main():
    ase_src = get_ase_source_path()
    if not ase_src.exists():
        raise FileNotFoundError(f"ASE source not found: {ase_src}")
    print("Loading ASE files...")
    docs = load_text_documents(ase_src)
    print(f"Indexed {len(docs)} files.")
    retriever = build_vector_store_and_retriever(docs)
    qa_chain = make_qa_chain(retriever)
    rag_tool = make_langgraph_tool(qa_chain)

    # Example run
    q = "Write a python script that constructs an ase.atoms object of a MoS2 monolayer with a S vacancy. \
        define the ase.atoms object as atoms and nothing else"
    res = rag_tool(q)
    print("Answer:\n", res["answer"])
    print("Sources:")
    for s in res["sources"]:
        print("-", s["source"], "chunk", s["chunk"])

    return rag_tool

if __name__ == "__main__":
    main()



