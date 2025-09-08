from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
import ase.io
from ase import Atoms


@tool
def run_ase_script(code: str, filename: str) -> None:
    """Run Python code that defines an ASE Atoms object as `atoms` and writes the atoms object to an extxyz file with a descriptive and unique name.
    Args:
        code: Python code to execute
        filename: descriptive and unique filename. The filename must end with the .extxyz extension
        """
    local_env = {}
    exec(code, {}, local_env)
    atoms = local_env.get("atoms")
    if not isinstance(atoms, Atoms):
        raise ValueError("Script did not define an ASE Atoms object named `atoms`.")
    else:
        ase.io.write(filename,atoms,format="extxyz")
    
memory = MemorySaver()
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

tools = [run_ase_script]
agent_executor = create_react_agent(model, tools, checkpointer=memory)
config = {"configurable": {"thread_id": "abc123"}}


input_message = {
    "role": "user",
    "content": "Write and execute a python script that constructs an ase.atoms object of a graphene monolayer. "
    "the lattice constant should be 2.46 angstroms. Write the atoms object to an extxyz file, and label it with a descriptive name and a unique identifies.",
}

#define the ase.atoms object as atoms and nothing else
for step in agent_executor.stream(
    {"messages": [input_message]}, config, stream_mode="values"
):
    step["messages"][-1].pretty_print()
