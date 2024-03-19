from langchain.agents import AgentExecutor, create_react_agent
from llm import openai, mistral_large
from tools.tool_getter import get_tools_for
from prompts import agent_prompt
from memory import memory
from constants import LLM

class BaseAgent:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        pass

    def generate_response(self, prompt):
        raise NotImplementedError("Subclasses must implement get_response method")

class OpenAiAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        tools = get_tools_for(openai)
        self.executor = AgentExecutor(
            agent=create_react_agent(openai, tools, agent_prompt),
            tools=tools,
            memory=memory,
            verbose=True
            )

    def generate_response(self, prompt):
        response = self.executor.invoke({ "input": prompt})
        return response['output']

class MistralAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        tools = get_tools_for(mistral_large)
        self.executor = AgentExecutor(
            agent=create_react_agent(mistral_large, tools, agent_prompt),
            tools=tools,
            memory=memory,
            verbose=True
            )

    def generate_response(self, prompt):
        response = self.executor.invoke({ "input": prompt})
        return response['output']

agent_map = {
    LLM.openai: OpenAiAgent,
    LLM.mistral_large: MistralAgent
}

def get_agent(agent_type):
    agent = agent_map.get(agent_type)

    if agent is None:
        raise ValueError("Invalid agent type")
    
    return agent()
