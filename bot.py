import streamlit as st
from utils import write_message
from agent_factory import get_agent
from constants import LLM, LLM_Label

# tag::setup[]
# Page Config
st.set_page_config("Ebert", page_icon=":movie_camera:")
# end::setup[]

# tag::session[]
# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the GraphAcademy Chatbot!  How can I help you?"},
    ]
# end::session[]

# tag::submit[]
# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        try:
            response = get_agent(st.session_state.llm).generate_response(message)
            write_message('assistant', response)
        except Exception as e:
            write_message('assistant', repr(e), isError=True)
# end::submit[]


# tag::chat[]
# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False, isError=message.get('isError', False))

# Handle any user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', prompt)

    # Generate a response
    handle_submit(prompt)
# end::chat[]

llm_label_mapping = {
    LLM_Label.OpenAI.name: LLM.openai,
    LLM_Label.Mistral.name: LLM.mistral_large
}

with st.sidebar:
    selected_llm = st.radio(
        "Select LLM:",
        (LLM_Label.OpenAI.name, LLM_Label.Mistral.name)
    )

    st.session_state.llm = llm_label_mapping[selected_llm]
