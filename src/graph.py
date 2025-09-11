import os
from typing import Annotated
import json
from typing_extensions import TypedDict
from IPython.display import Image, display
from typing import Annotated

from langchain_tavily import TavilySearch
from langchain_core.tools import tool

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt
from langchain.chat_models import init_chat_model
from create_ASE_RAG import RAG_ASE

class State(TypedDict):
    messages: Annotated[list, add_messages]

#nodes
def evaluate_paper(state: State):
    """First LLM call to parse pdf and suggest structures"""
    msg = llm.invoke(f"I am interested in investigating the \
                     structures defined in this paper using atomistic simulation. \
                     This paper contains TEM results with structural information. \
                     Determine the structures of interest in this paper. \
                     The structures of interest should be related to the main hypothesis of the paper. \
                     Next, construct ase.atoms objects for the systems of interest. \
                     Note these ase.atoms objects will be the starting point of a simulation (either DFT or MD). \
                     In the event there are certain degrees of freedom that are unclear or poorly defined in the paper, \
                     it may be useful to produce structures that sweep over several reasonable values.")
    return {"structure_suggestion":msg.content}



@tool
def human_assistance(query: str) -> str:
    """Request assistance from a human."""
    human_response = interrupt({"query": query})
    return human_response["data"]

if os.environ["GOOGLE_API_KEY"] is None:
    print("Warning: set GOOGLE_API_KEY in environment")

llm = init_chat_model("google_genai:gemini-2.0-flash")
graph_builder = StateGraph(State)


tool = TavilySearch(max_results=2)
tools = [tool, human_assistance,RAG_ASE,pdf_parser]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    assert(len(message.tool_calls) <= 1)
    return {"messages": [message]}

graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

#nodes we need
graph_builder.add_node("evaluate_paper",evaluate_paper) #determine the structures of interest from PDF plus TEM image plus user request/preference
graph_builder.add_node("refine_structure_description",refine_structure_description) #add more context/user preference for how structures are created
graph_builder.add_node("generate_structures",generate_structures) #use ase rag tool to generate confirmed structures


"""graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")"""

#edges we need
graph_builder.add_edge(START,"evaluate_paper")
graph_builder.add_conditional_edges("evaluate_paper",confirm_structures,{"Fail":"refine_structure_description","Pass":"generate_structures"}) #confirm with users which structures to generate and that structures correspond with paper.
graph_builder.add_edge("generate_structures",END) #generate structures and organize into folder tree
graph = graph_builder.compile()


###################################################

# Extra stuff that is useful

###################################################

#visualize the graph
try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

#run the chatbot
def stream_graph_updates(user_input: str):
    for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break