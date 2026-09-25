import os

import streamlit as st
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv()


# --------------------------------------------------
# LangSmith Tracking
# --------------------------------------------------
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv(
    "LANGCHAIN_PROJECT",
    "LangChain-Ollama-Demo"
)


# --------------------------------------------------
# Prompt Template
# --------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. "
            "Please respond to the question asked."
        ),
        (
            "user",
            "Question: {question}"
        )
    ]
)


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("LangChain Demo With Gemma Model")

input_text = st.text_input(
    "What question do you have in mind?"
)


# --------------------------------------------------
# Ollama Chat Model
# --------------------------------------------------
llm = ChatOllama(
    model="gemma:2b",
    temperature=0
)


# --------------------------------------------------
# Output Parser
# --------------------------------------------------
output_parser = StrOutputParser()


# --------------------------------------------------
# LangChain LCEL Chain
# --------------------------------------------------
chain = prompt | llm | output_parser


# --------------------------------------------------
# Invoke Chain
# --------------------------------------------------
if input_text:
    response = chain.invoke(
        {
            "question": input_text
        }
    )

    st.write(response)