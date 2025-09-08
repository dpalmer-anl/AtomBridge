# Import relevant functionality
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


input_message = {
    "role": "user",
    "content": "Hi, I'm Dan and I live in Champaign.",
}
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()

input_message = {
    "role": "user",
    "content": "What's the weather like in Champaign?",
}

for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()

input_message = {
    "role": "user",
    "content": "What is 3.4 * 8.5?",
}

for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()