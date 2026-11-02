
import validators
import streamlit as st

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)
from langchain_community.document_loaders import (
    YoutubeLoader,
    UnstructuredURLLoader
)
from langchain_core.prompts import ChatPromptTemplate

# Streamlit APP
st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜"
)

st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")

# Groq API Key
with st.sidebar:
    groq_api_key = st.text_input(
        "Groq API Key",
        value="",
        type="password"
    )

if not groq_api_key.strip():
    st.info("Please enter your Groq API Key in the sidebar.")
    st.stop()

# LLM Model
llm = ChatGroq(
    api_key=groq_api_key,
    model="openai/gpt-oss-120b",
    streaming=True
)

# Prompt Template


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert summarization assistant. Summarize the provided content in approximately 500 words."),
    ("human", "Summarize the following content:\n\n{context}")
])


# URL Input
generic_url = st.text_input(
    "URL",
    label_visibility="collapsed"
)

if st.button("Summarize the content YT or Website"):

    if not generic_url.strip():
        st.error("Please provide a URL to get started.")

    elif not validators.url(generic_url):
        st.error("Please enter a valid URL.")

    else:
        try:
            with st.spinner("Waiting..."):

                # Load YouTube or Website data
                if "youtube.com" in generic_url or "youtu.be" in generic_url:

                    loader = YoutubeLoader.from_youtube_url(
                        generic_url,
                        add_video_info=False
                    )

                else:
                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0"
                        }
                    )

                docs = loader.load()
                st.write("Number of documents:", len(docs))

                for i, doc in enumerate(docs):
                    st.write(f"Document {i + 1} length:", len(doc.page_content))
                    st.write(doc.page_content[:500])

                if not docs:
                    st.warning("No content found at the provided URL.")
                    st.stop()

                # New Runnable-based Summarization Chain
                chain = create_stuff_documents_chain(
                    llm,
                    prompt
                )

                output_summary = chain.invoke({
                    "context": docs
                })

                st.write(output_summary)
        except Exception as e:
            st.error(f"Exception: {str(e)}")
