from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

def multiply(a:float,b:float) -> float:
    """Multiply two floats.

    Args:
        a: First float
        b: Second float
    """
    return a*b

memory = MemorySaver()
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
search = TavilySearch(max_results=2)
tools = [search, multiply]
agent_executor = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}
