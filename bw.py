from browser_use import Agent, ChatBrowserUse
from dotenv import load_dotenv
import asyncio

load_dotenv()

async def main():
    llm = ChatBrowserUse()
    task = "Go to online eletric store in sweden and find the best price for a lg Oled tv"
    agent = Agent(task=task, llm=llm)
    
    await agent.run()

if __name__ == "__main__":
    asyncio.run(main())