from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langserve import add_routes

import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
model=ChatGroq(model="openai/gpt-oss-120b",groq_api_key=groq_api_key)

# Prompt template
generic_template = "Translate the following in to {language}:"
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generic_template),
        ("user", "{text}")
    ]
)

# Parser
parser = StrOutputParser()


# Chain
chain = prompt | model | parser

# App definition
app = FastAPI(title="Langchain Server", version="1.0",
              description="A simple APi server using Langchain")
add_routes(app, 
           chain,
            path="/api/v1/langchain")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)