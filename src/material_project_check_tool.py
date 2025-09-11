from langchain.tools import BaseTool
from mp_api.client import MPRester
from typing import Optional, Dict
import os

class MPStructureSymmetryTool(BaseTool):
    name: str = "materials_project_structure_symmetry"
    description: str = (
        "Given a Materials Project material_id (e.g. 'mp-149'), returns the lattice parameters "
        "and symmetry info: crystal system, space group number & symbol, point group, etc."
    )
    mp_api_key: str

    def __init__(self, mp_api_key: str):
        self.mp_api_key = mp_api_key

    def _run(self, material_id: str) -> Dict:
        """
        material_id: str, e.g. 'mp-123', must be a valid Materials Project ID.
        Returns a dict with keys:
          - lattice: {a, b, c, alpha, beta, gamma}
          - symmetry: {crystal_system, space_group_symbol, space_group_number, point_group, etc.}
        """
        # Input validation
        if not material_id.startswith("mp-"):
            return {"error": f"material_id '{material_id}' does not look like an MP ID."}

        try:
            with MPRester(self.mp_api_key) as mpr:
                docs = mpr.materials.summary.search(
                    material_ids=[material_id],
                    fields=["structure", "symmetry"]
                )
        except Exception as e:
            return {"error": f"API request failed: {e}"}

        if not docs:
            return {"error": f"No data found for material_id '{material_id}'."}

        doc = docs[0]
        # Structure includes lattice vectors etc.
        struct = doc.structure
        sym = doc.symmetry

        # Extract lattice parameters from structure
        lattice = struct.lattice
        lattice_params = {
            "a": lattice.a,
            "b": lattice.b,
            "c": lattice.c,
            "alpha": lattice.alpha,
            "beta": lattice.beta,
            "gamma": lattice.gamma,
        }

        # Extract symmetry info
        symmetry_info = {
            "crystal_system": sym.crystal_system,
            "space_group_symbol": sym.symbol,
            "space_group_number": sym.number,
            "point_group": sym.point_group,
            # maybe other fields, depending on sym dict
        }

        return {
            "material_id": material_id,
            "lattice": lattice_params,
            "symmetry": symmetry_info
        }

    async def _arun(self, material_id: str) -> Dict:
        # for async support if needed
        return self._run(material_id)
    
if __name__=="__main__":
    from langchain.chat_models import init_chat_model
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import tool

    mp_tool = MPStructureSymmetryTool(mp_api_key=os.environ["MP_API_KEY"])
    tools = [mp_tool]
    crystal_id = 'mp-1821'

    memory = MemorySaver()
    model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
    agent_executor = create_react_agent(model, tools, checkpointer=memory)

    config = {"configurable": {"thread_id": "abc123"}}


    input_message = {
        "role": "user",
        "content": "What are the lattice parameters and space group of "+crystal_id+"?",
    }

    for step in agent_executor.stream(
        {"messages": [input_message]}, config, stream_mode="values"
    ):
        step["messages"][-1].pretty_print()