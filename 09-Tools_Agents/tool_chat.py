import os

import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import (
    ArxivQueryRun,
    WikipediaQueryRun,
    DuckDuckGoSearchRun,
)
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_classic.callbacks import StreamlitCallbackHandler
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
load_dotenv()

st.set_page_config(
    page_title="LangChain - Chat with Search",
    page_icon="🔎",
    layout="wide",
)

st.title("🔎 LangChain - Chat with Search")
st.caption("Ask questions and use web search, Wikipedia, or arXiv research tools.")

# ------------------------------------------------------------
# Sidebar settings
# ------------------------------------------------------------
st.sidebar.title("Settings")

default_model = "openai/gpt-oss-20b"
model_name = st.sidebar.text_input(
    "Groq model",
    value=os.getenv("GROQ_MODEL", default_model),
    help="Use a model available to your Groq account that supports tool calling.",
)

env_api_key = os.getenv("GROQ_API_KEY", "")
api_key = st.sidebar.text_input(
    "Enter your Groq API key",
    value=env_api_key,
    type="password",
)

st.sidebar.caption(
    "Your API key is used to make requests to Groq. Avoid sharing it or committing it to source control."
)

if st.sidebar.button("Clear chat", use_container_width=True):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm a chatbot who can search the web. How can I help you?",
        }
    ]
    st.rerun()

# ------------------------------------------------------------
# Initialize chat state
# ------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm a chatbot who can search the web. How can I help you?",
        }
    ]

# Render existing messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ------------------------------------------------------------
# Build tools and agent
# ------------------------------------------------------------
def build_tools():
    """Create the search tools used by the agent."""
    arxiv_wrapper = ArxivAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=200,
    )
    arxiv_tool = ArxivQueryRun(
        name="arxiv_search",
        description=(
            "Search arXiv for academic papers. Use for scientific, technical, "
            "machine learning, and AI research questions."
        ),
        api_wrapper=arxiv_wrapper,
    )

    wikipedia_wrapper = WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=200,
    )
    wikipedia_tool = WikipediaQueryRun(
        name="wikipedia_search",
        description=(
            "Search Wikipedia for concise background information about people, "
            "places, concepts, organizations, and events."
        ),
        api_wrapper=wikipedia_wrapper,
    )

    web_search_tool = DuckDuckGoSearchRun(
        name="web_search",
        description=(
            "Search the web for current information, websites, and general "
            "questions that need online search."
        ),
    )

    return [web_search_tool, arxiv_tool, wikipedia_tool]


def build_agent(api_key_value: str, selected_model: str):
    """Create a tool-calling agent and executor."""
    llm = ChatGroq(
        groq_api_key=api_key_value,
        model_name=selected_model,
        temperature=0,
        streaming=True,
    )

    tools = build_tools()

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful research assistant. "
                "Use the available tools when they help answer the user's question. "
                "Do not claim you searched a source unless you actually used its tool. "
                "Summarize results clearly and mention uncertainty when sources are limited.",
            ),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(
        llm=llm,
        tools=tools,
        prompt=prompt_template,
    )

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )

    return executor


def to_langchain_history(messages):
    """Convert stored Streamlit chat messages to LangChain message objects."""
    converted = []

    for message in messages:
        role = message.get("role")
        content = message.get("content", "")

        if role == "user":
            converted.append(HumanMessage(content=content))
        elif role == "assistant":
            converted.append(AIMessage(content=content))

    return converted


# ------------------------------------------------------------
# Handle new user question
# ------------------------------------------------------------
user_question = st.chat_input("What would you like to know?")

if user_question:
    # Display and save the new user message.
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )
    with st.chat_message("user"):
        st.markdown(user_question)

    if not api_key:
        st.warning(
            "Enter your Groq API key in the sidebar. You can also set "
            "GROQ_API_KEY in a local .env file."
        )
        st.stop()

    with st.chat_message("assistant"):
        callback_container = st.container()
        callback_handler = StreamlitCallbackHandler(
            callback_container,
            expand_new_thoughts=False,
        )

        try:
            # Exclude the latest question from history because it is passed
            # separately in the "input" field.
            previous_messages = st.session_state.messages[:-1]
            chat_history = to_langchain_history(previous_messages)

            search_agent = build_agent(api_key, model_name)

            with st.spinner("Thinking and searching when needed..."):
                result = search_agent.invoke(
                    {
                        "input": user_question,
                        "chat_history": chat_history,
                    },
                    config={"callbacks": [callback_handler]},
                )

            answer = result.get("output", "I couldn't generate an answer.")
            st.markdown(answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

        except Exception as exc:
            st.error(
                "The request failed. Check your Groq model, API key, package "
                "versions, and network connection."
            )
            st.exception(exc)
