from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from llm import llm
from tools.vector import kg_qa
from tools.cypher import cypher_qa
from requests import post
import streamlit as st

memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=llm.invoke,
        return_direct=False,
        handle_parsing_errors=True
        ),
    Tool.from_function(
        name="Vector Search Index",
        description="Provides information about movie plots using Vector Search",
        func=kg_qa,
        return_direct=False,
        handle_parsing_errors=True
    ),
    Tool.from_function(
        name="Graph Cypher QA Chain",
        description="Provides information about Movies including their Actors, Directors and User reviews",
        func = cypher_qa,
        return_direct=False,
        handle_parsing_errors=True
    ),
]

agent_prompt = PromptTemplate.from_template("""
You are a movie expert providing information about movies.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate to movies, actors or directors.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")
agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
    )

# messages = [{
#     "role": "system",
#     "content": "You are an AI Assistant."
# }]

def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """
    # Using Mistral Large directly in Azure:
    # question = {
    #     "role": "user",
    #     "content": prompt
    # }
    # messages.append(question)

    # headers = { "Authorization": f"Bearer {st.secrets['MISTRAL_API_KEY']}" }
    # body = {
    #     "messages": messages,
    #     "temperature": 0.8,
    #     "max_tokens": 512
    # }
    # response = post(f"{st.secrets['MISTRAL_API_TARGET']}/v1/chat/completions", json=body, headers=headers)
    # answer = response.json()["choices"][0]["message"]
    # messages.append(answer)

    # return answer["content"]

    response = agent_executor.invoke({"input": prompt})

    return response['output']