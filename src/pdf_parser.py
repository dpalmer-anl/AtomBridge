from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

@tool
def parse_pdf(pdf_filename: str) -> None:
    """load the pdf with the pdf_filename using PyPDFLoader. 
    Args:
        code: Python code to execute
        filename: descriptive and unique filename. The filename must end with the .extxyz extension
    return:
        text of paper
        """
    loader = PyPDFLoader(pdf_filename)
    docs = loader.load()
    paper_text = "\n\n".join([d.page_content for d in docs])
    return paper_text

memory = MemorySaver()
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

pdf_filename = "bendersky-et-al-2016-microscopy-study-of-structural-evolution-in-epitaxial-licoo2-positive-electrode-films-during.pdf"
paper_text = parse_pdf(pdf_filename)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a research assistant."),
    ("human", "Here is the paper:\n{paper}\n\nQuestion: {question}")
])

chain = prompt | model
response = chain.invoke({"paper": paper_text, "question": "Summarize the system of interest mentioned in this paper"})
print(response.content)