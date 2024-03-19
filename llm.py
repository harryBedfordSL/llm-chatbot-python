import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_mistralai.chat_models import ChatMistralAI

openai = ChatOpenAI(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    model=st.secrets["OPENAI_MODEL"],
    openai_organization=st.secrets["OPENAI_ORGANIZATION"],
)

mistral_large = ChatMistralAI(
    endpoint=st.secrets["MISTRAL_API_TARGET"],
    mistral_api_key=st.secrets["MISTRAL_API_KEY"],
)