from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()

import asyncio

async def main():
    client =  MultiServerMCPClient(
        {
            "Math":{
                "command":"python",
                "args":["mathserver.py"],
                "transport":"stdio"
            },
            "Weather":{
                "url":"http://127.0.0.1:8000/mcp",
                "transport":"streamable-http"
            }
        }
    )


    import os
    os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

    tools = await client.get_tools()
    model = ChatGroq(model="openai/gpt-oss-20b")
    agent = create_agent(model=model,tools=tools)

    math_response = await agent.ainvoke(
        {"messages" : [{"role":"user", "content" : "What's (3 +5) * 7?"}]}
        )

    print("Math response:",math_response["messages"][-1].content)

    weather_response = await agent.ainvoke(
            {"messages" : [{"role":"user", "content" : "What's the weather in Los Angeles city in California?"}]}
            )
    
    print("Weather response:",weather_response["messages"][-1].content)

asyncio.run(main())