from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from create_ASE_RAG import RAG_ASE
import ase.io
from ase import Atoms
from langchain_experimental.tools import PythonREPLTool
import re
import subprocess
import tempfile

def extract_code(text: str) -> str:
    """Extract the first Python code block from a markdown string."""
    match = re.search(r"```python(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

def run_code(code: str) -> tuple[str, str, int]:
    """Run code in a temp .py file and capture stdout, stderr, exit code."""
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w") as f:
        f.write(code)
        f.flush()
        result = subprocess.run(
            ["python", f.name], capture_output=True, text=True
        )
    return result.stdout, result.stderr, result.returncode

def parse_pdf(pdf_filename: str) -> None:
    """load the pdf with the pdf_filename using PyPDFLoader. 
    Args:
        pdf_filename: filename of the pdf
    return:
        text of paper
        """
    loader = PyPDFLoader(pdf_filename)
    docs = loader.load()
    paper_text = "\n\n".join([d.page_content for d in docs])
    return paper_text


memory = MemorySaver()
model = init_chat_model("gemini-2.5-pro", model_provider="google_genai")
tools = [RAG_ASE]
agent_executor = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}

pdf_filename = "../papers/lee-et-al-2020-deep-learning-enabled-strain-mapping-of-single-atom-defects-in-two-dimensional-transition-metal.pdf"
paper_text = parse_pdf(pdf_filename)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant."),
    ("human", "Here is the paper:\n{paper}\n\nQuestion: {question}")
])

chain = prompt | model
"""response = chain.invoke({"paper": paper_text, "question": "I am interested in investigating the \
                     structures defined in this paper using atomistic simulation. \
                     This paper contains TEM results with structural information. \
                     Determine the structures of interest in this paper. \
                     The structures of interest should be related to the main hypothesis of the paper. \
                     Next, construct ase.atoms objects for the systems of interest. \
                     Write a python script that constructs the ase.atoms objects for the systems of interest\
                     This python script should create each ase.atoms object and write the object to a CIF file in the current working directory with a descriptive filename and .cif extension using the \
                     ase.io.write(<filename>,atoms_object,format='vasp').\
                     Note these ase.atoms objects will be the starting point of a simulation (either DFT or MD). \
                     once the python script is written, execute the python script using the PythonREPLTool.\
                     check to see if the .cif files were actually written to the current directory\
                     In the event there are certain degrees of freedom that are unclear or poorly defined in the paper, \
                     it may be useful to produce structures that sweep over several reasonable values."})
print(response.content)"""
input_message = {"paper": paper_text, "question": "I am interested in investigating the \
                     structures defined in this paper using atomistic simulation. \
                     This paper contains TEM results with structural information. \
                     Determine the structures of interest in this paper. \
                     The structures of interest should be related to the main hypothesis of the paper. \
                     Next, construct ase.atoms objects for the systems of interest. \
                     Write a python script that constructs the ase.atoms objects for the systems of interest\
                     This python script should create each ase.atoms object and write the object to a CIF file in the current working directory with a descriptive filename and .cif extension using the \
                     ase.io.write(<filename>,atoms_object,format='vasp').\
                     Note these ase.atoms objects will be the starting point of a simulation (either DFT or MD). \
                     In the event there are certain degrees of freedom that are unclear or poorly defined in the paper, \
                     it may be useful to produce structures that sweep over several reasonable values."}

for i in range(5):  # allow a few retries
    response = chain.invoke(input_message)
    code = extract_code(response.content)
    stdout, stderr, rc = run_code(code)

    if rc == 0:
        print("✅ Success:\n", stdout)
        break
    else:
        print("❌ Error:\n", stderr)
        query = f"The code failed with error:\n{stderr}\nPlease fix it."

