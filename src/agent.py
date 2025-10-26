import asyncio
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API = os.getenv("GROQ_API")

async def create_langchain_docs_agent():
    # Connect to LangChain docs MCP server
    client = MultiServerMCPClient({
        "docs": {
            "transport": "streamable_http",
            "url": "https://docs.langchain.com/mcp",
        }
    })

    # Get tools from MCP server
    tools = await client.get_tools()

    # Initialize LLM
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        api_key=GROQ_API,
        max_tokens=200,
        temperature=0
    )

    # Create the agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=(
            "You are an AI assistant that answers questions using LangChain documentation. "
            "If you do not know the answer, use the MCP server tools to search the docs."
        )
    )
    return agent

# Export for LangGraph
# agent = asyncio.run(create_langchain_docs_agent())