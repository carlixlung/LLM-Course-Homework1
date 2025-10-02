from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
import asyncio
from contextlib import AsyncExitStack
import os

brave_api = ""
system_message = ""
notion_api = ""

#Dealing with APIs
with open('BraveSearchAPI.txt', 'r') as file:
    brave_api = file.read()  # Reads the entire content

if brave_api == "":
    raise ValueError("Brave API is not set")

with open('SystemMessage.txt', 'r') as file:
    system_message = file.read()

if system_message == "":
    raise ValueError("System message is not set")

with open('Notion_API.txt', 'r') as file:
    notion_api = file.read()  # Reads the entire content

if notion_api == "":
    raise ValueError("Notion API is not set")

with open('Github_API.txt', 'r') as file:
    github_api = file.read()  # Reads the entire content

if github_api == "":
    raise ValueError("Github API is not set")


llm = ChatOpenAI(
        model="llama3.1:8b",
        temperature=0.1,
        max_retries=2,
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        streaming=True
    )

filesystem_params = StdioServerParameters(
    command = "npx",
    args =[
            "-y",
            "@modelcontextprotocol/server-filesystem",
            os.path.expanduser("~")  
        ]
)

brave_params = StdioServerParameters(
    command="npx",
    args=[
        "-y",
        "@modelcontextprotocol/server-brave-search",
    ],
    env={"BRAVE_API_KEY": brave_api}
)

sequential_params = StdioServerParameters(
    command="npx",
    args=[
        "-y",
        "@modelcontextprotocol/server-sequential-thinking"
    ]
)

puppeteer_params = StdioServerParameters(
    command="npx",
    args=[
        "-y",
        "@modelcontextprotocol/server-puppeteer"
    ]
)

notion_params = StdioServerParameters(
    command="npx",
    args=[
        "-y",
        "mcp-remote",
        "https://mcp.notion.com/mcp"
    ]
    )

github_params = StdioServerParameters(
    command = "npx",
    args=[
        "-y",
        "@modelcontextprotocol/server-github"
        ],
    env = {"GITHUB_PERSONAL_ACCESS_TOKEN": github_api}
    )


#main functions to run the agent filesystem_params, brave_params, sequential_params, notion_params, github_params
async def run_agent(user_message):
    async with AsyncExitStack() as stack:
        #Load the tools
        all_tools = []
        for params in [ filesystem_params, brave_params, sequential_params, puppeteer_params, github_params]:
            try:

                server_name = params.args[-1] if params.args else params.command

                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                
                tools = await load_mcp_tools(session)

                print(f"{server_name}: Loaded {len(tools)} tools")
                for t in tools:
                    print(f"   - {t.name}")

                all_tools.extend(tools)

                filtered_tools = [
                    t for t in all_tools 
                    if t.name in [
                        'write_file',
                        'create_directory',
                        'brave_web_search',
                        'puppeteer_navigate',
                        'puppeteer_click',
                        'puppeteer_evaluate',
                        'sequentialthinking',
                        'puppeteer_screenshot',
                        'create_or_update_file',
                        'create_repository'

                    ]
                ]

            except Exception as e:
                print(e)

        agent = create_react_agent(llm, filtered_tools)
        
       


        print("Agent Output:\n")
    
        # Simple streaming - just shows the messages as they come
        async for chunk in agent.astream(
            {
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ]
            }
        ):
            # Print each chunk as it arrives
            if "agent" in chunk:
                print(chunk["agent"]["messages"][0].content)
            elif "tools" in chunk:
                print(f"\n[Tool executed: {chunk['tools']['messages'][0].name}]\n")


if __name__ == "__main__":
    research_folder = os.path.join(os.path.expanduser("~"), "LLM_research")
    prompt1 = """Create a GitHub repository called 'hello_world_ollama'. Then create a file 'hello_world.py' with this code: print('Hello World!') Use create_repository, then create_or_update_file.""" 

    result = asyncio.run(run_agent(prompt1))


