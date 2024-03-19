from langchain.tools import Tool
from tools.cypher import get_cypher_qa

def get_tools_for(llm):
    return [
        Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=llm.invoke,
        return_direct=False,
        handle_parsing_errors=True
        ),
        Tool.from_function(
            name="Graph Cypher QA Chain",
            description="Provides information about Movies including their Actors, Directors and Budget",
            func=get_cypher_qa(llm),
            return_direct=False,
            handle_parsing_errors=True
        ),
    ]