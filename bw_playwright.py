import os
import asyncio
import warnings
from dotenv import load_dotenv

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from langchain_groq import ChatGroq

warnings.filterwarnings("ignore", category=ResourceWarning)

load_dotenv()

async def safe_invoke(agent, session, task, retries=2):
    for attempt in range(retries):
        try:
            result = await agent.ainvoke(task)

            await asyncio.sleep(0.5)

            try:
                await session.call_tool("browser_wait_for", {"time": 5})
            except Exception:
                pass

            return result

        except Exception as e:
            msg = str(e)
            if "ClosedResourceError" in msg:
                
                print("Ignored stream close error.")
                return result
            print(f"Attempt {attempt+1} failed: {e}")
            await asyncio.sleep(3)

    raise RuntimeError("All retries failed for task")

async def main():
    client = MultiServerMCPClient({
        "browser": {
            "transport": "streamable_http",
            "url": "http://localhost:8931/mcp",
        }
    })

    async with client.session("browser") as session:
        tools = await load_mcp_tools(session)

        llm = ChatGroq(
            model="openai/gpt-oss-120b",
            api_key=os.getenv("GROQ_API"),
            temperature=0
        )

        agent = create_agent(llm, tools)

        SYSTEM_PROMPT = """
        You are an autonomous AI agent connected to a Model Context Protocol (MCP) server.
        Use MCP tools like browser_navigate, browser_click, browser_type, browser_wait_for,
        browser_evaluate, browser_extract, and browser_snapshot to interact with the web.

        Rules:
        - After navigation or DOM changes, wait 5s.
        - Retry once if tool fails.
        - Return concise structured results (JSON or plain text).
        - Never refuse browsing tasks.
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Go to amazon sweden and click on one product you like and just get the name of it"}
        ]

        result = await safe_invoke(agent, session, {"messages": messages})
        print("\nFinal Result:\n", result)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\ntopped manually.")
