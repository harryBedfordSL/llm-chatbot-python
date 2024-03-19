from langchain.chains import GraphCypherQAChain
from graph import graph
from prompts import cypher_prompt

def get_cypher_qa(llm):
    return GraphCypherQAChain.from_llm(
    llm=llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt
)